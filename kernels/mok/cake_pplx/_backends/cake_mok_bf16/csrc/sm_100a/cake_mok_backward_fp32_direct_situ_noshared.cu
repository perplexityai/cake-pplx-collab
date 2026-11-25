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


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
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

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}






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
kernel_cake_mok_backward_fp32_direct_situ_noshared(const __grid_constant__ CUtensorMap dy_r, const __grid_constant__ CUtensorMap dg_r, const __grid_constant__ CUtensorMap du_r, const __grid_constant__ CUtensorMap x_nt_r, const __grid_constant__ CUtensorMap dy_atb_r, const __grid_constant__ CUtensorMap dg_atb_r, const __grid_constant__ CUtensorMap du_atb_r, const __grid_constant__ CUtensorMap x_atb_r, const __grid_constant__ CUtensorMap h_atb_r, const __grid_constant__ CUtensorMap wg_r, const __grid_constant__ CUtensorMap wu_r, const __grid_constant__ CUtensorMap wd_r, const __grid_constant__ CUtensorMap wg_nt_r, const __grid_constant__ CUtensorMap wu_nt_r, const __grid_constant__ CUtensorMap dh_r, const __grid_constant__ CUtensorMap dx_r, const __grid_constant__ CUtensorMap gate_out_r, const __grid_constant__ CUtensorMap up_out_r, const __grid_constant__ CUtensorMap dwg_r, const __grid_constant__ CUtensorMap dwu_r, const __grid_constant__ CUtensorMap dwd_r, const __grid_constant__ CUtensorMap dh_sw_r, const __grid_constant__ CUtensorMap gate_sw_r, const __grid_constant__ CUtensorMap up_sw_r, const __grid_constant__ CUtensorMap dg_sw_r, const __grid_constant__ CUtensorMap du_sw_r, const __grid_constant__ CUtensorMap h_sw_r, __nv_bfloat16* __restrict__ x_routed_ptr, __nv_bfloat16* __restrict__ dy_routed_ptr, __nv_bfloat16* __restrict__ dy_scaled_ptr, __nv_bfloat16* __restrict__ dx_routed_ptr, float* __restrict__ weights, float* __restrict__ partials, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ dy_peers, unsigned long long* __restrict__ dx_peers, unsigned long long* __restrict__ weight_peers, unsigned long long* __restrict__ dweight_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ dh_ready, int* __restrict__ dg_ready, int* __restrict__ dy_ready, int* __restrict__ dx_ready, int* __restrict__ replay_x, int* __restrict__ replay_gu, int* __restrict__ replay_h, int* __restrict__ buffers_done, int* __restrict__ weight_ready, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float situ_beta, float situ_lb, CakeFastDivmod mini_size_div, CakeFastDivmod mini_bwd_div, CakeFastDivmod mini_replay_div, CakeFastDivmod replay_macro_tasks_div, CakeFastDivmod weight_tasks_div, CakeFastDivmod shared_wgrad_div, CakeFastDivmod dh_col_blocks_div, CakeFastDivmod dx_col_blocks_div, CakeFastDivmod replay_col_blocks_div)
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
    #define replay_arrived_addr (mbar_base + 0)
    #define swiglu_arrived_addr (mbar_base + 24)
    #define gemm_arrived_addr (mbar_base + 40)
    #define gemm_finished_addr (mbar_base + 72)
    #define output_arrived_addr (mbar_base + 104)
    #define output_finished_addr (mbar_base + 120)
    #define schedule_arrived_addr (mbar_base + 136)
    #define schedule_finished_addr (mbar_base + 144)
    #define drain_arrived_0_addr (mbar_base + 152)
    #define drain_arrived_1_addr (mbar_base + 160)
    #define drain_arrived_2_addr (mbar_base + 168)
    #define drain_arrived_3_addr (mbar_base + 176)
    #define drain_arrived_4_addr (mbar_base + 184)
    #define drain_arrived_5_addr (mbar_base + 192)
    #define drain_arrived_6_addr (mbar_base + 200)
    #define drain_arrived_7_addr (mbar_base + 208)
    #define drain_finished_0_addr (mbar_base + 216)
    #define drain_finished_1_addr (mbar_base + 224)
    #define drain_finished_2_addr (mbar_base + 232)
    #define drain_finished_3_addr (mbar_base + 240)
    #define drain_finished_4_addr (mbar_base + 248)
    #define drain_finished_5_addr (mbar_base + 256)
    #define drain_finished_6_addr (mbar_base + 264)
    #define drain_finished_7_addr (mbar_base + 272)
    #define dispatch_arrived_addr (mbar_base + 280)
    #define range_arrived_addr (mbar_base + 288)
    #define rows_arrived_addr (mbar_base + 312)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* a_nt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int a_nt_addr = smem + 1024;
    __nv_bfloat16* mn_view = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int mn_view_addr = smem + 1024;
    __nv_bfloat16* b_nt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int b_nt_addr = smem + 66560;
    __nv_bfloat16* mn_view_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int mn_view_addr_0 = smem + 66560;
    __nv_bfloat16* b_nt_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int b_nt_hi_addr = smem + 132096;
    __nv_bfloat16* mn_view_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int mn_view_addr_1 = smem + 132096;
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int d_smem_addr = smem + 206848;
    unsigned int* flat_view = reinterpret_cast<unsigned int*>(smem_raw + 206848);
    const int flat_view_addr = smem + 206848;
    __nv_bfloat16* sw_dh = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_dh_addr = smem + 1024;
    __nv_bfloat16* sw_gate = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int sw_gate_addr = smem + 66560;
    __nv_bfloat16* sw_up = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int sw_up_addr = smem + 132096;
    float* flat_view_0 = reinterpret_cast<float*>(smem_raw + 197632);
    const int flat_view_addr_0 = smem + 197632;
    __nv_bfloat16* replay_gate = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int replay_gate_addr = smem + 1024;
    __nv_bfloat16* replay_up = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int replay_up_addr = smem + 99328;
    __nv_bfloat16* replay_hidden = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int replay_hidden_addr = smem + 197632;
    __nv_bfloat16* flat_view_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int flat_view_addr_1 = smem + 1024;
    unsigned int* flat_view_2 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int flat_view_addr_2 = smem + 1024;
    float* flat_view_3 = reinterpret_cast<float*>(smem_raw + 197632);
    const int flat_view_addr_3 = smem + 197632;
    int* flat_view_4 = reinterpret_cast<int*>(smem_raw + 199168);
    const int flat_view_addr_4 = smem + 199168;
    int* flat_view_5 = reinterpret_cast<int*>(smem_raw + 197632);
    const int flat_view_addr_5 = smem + 197632;
    float* flat_view_6 = reinterpret_cast<float*>(smem_raw + 199680);
    const int flat_view_addr_6 = smem + 199680;

    // Kernel setup ops
    float k1 = __fdiv_rn(-2.88539f, situ_beta);
    float k2 = __fdiv_rn(-2.88539f, situ_lb);
    float bl;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(bl) : "f"(situ_beta), "f"(situ_lb));
    int tokens = num_tokens[0];
    int shared_down = 0;
    int shared_swiglu = 0;
    int shared_dx = 0;
    int shared_wgrad = ((intermediate / 256 * ((hidden + 512 - 1) / 512)) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (intermediate / 256 * ((hidden + 512 - 1) / 512)) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
    int shared_tasks = 0;
    int mini_down = mini_size / 256 * ((intermediate + 511) / 512);
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 16 - 1) / 16;
    int mini_dx = mini_size / 256 * ((hidden + 511) / 512);
    int mini_replay_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int mini_bwd = mini_down + mini_swiglu + mini_dx;
    int mini_replay = 2 * mini_down + mini_replay_swiglu;
    int wgrad_tasks = 3 * experts * shared_wgrad;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis = (tokens + mini_size - 1) / mini_size;
    int _min_0 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
    int saved_minis = (_min_0 + mini_size - 1) / mini_size;
    int true_compute = shared_tasks + minis * mini_bwd + (minis - saved_minis) * mini_replay + macros * wgrad_tasks;
    int weight_tasks = experts * shared_wgrad;
    int saved_tasks = saved_minis * mini_bwd + 3 * weight_tasks;
    int replay_macro_tasks = macro_size / mini_size * (mini_replay + mini_bwd) + 3 * weight_tasks;
    int comm_clusters = comm_sms / 2;
    int true_clusters = comm_clusters + true_compute;
    if (true_clusters <= bid / 2) {
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (42 barriers at smem_raw[0..336))

    if (threadIdx.x == 0) {
        // replay_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 0, 1);
        mbarrier_init(smem + 8, 1);
        mbarrier_init(smem + 16, 1);
        // swiglu_arrived: 2 barriers, init_count=1
        mbarrier_init(smem + 24, 1);
        mbarrier_init(smem + 32, 1);
        // gemm_arrived: 4 barriers, init_count=1
        mbarrier_init(smem + 40, 1);
        mbarrier_init(smem + 48, 1);
        mbarrier_init(smem + 56, 1);
        mbarrier_init(smem + 64, 1);
        // gemm_finished: 4 barriers, init_count=1
        mbarrier_init(smem + 72, 1);
        mbarrier_init(smem + 80, 1);
        mbarrier_init(smem + 88, 1);
        mbarrier_init(smem + 96, 1);
        // output_arrived: 2 barriers, init_count=1
        mbarrier_init(smem + 104, 1);
        mbarrier_init(smem + 112, 1);
        // output_finished: 2 barriers, init_count=2
        mbarrier_init(smem + 120, 2);
        mbarrier_init(smem + 128, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 136, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 144, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 152, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 160, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 168, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 176, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 184, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 192, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 200, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 208, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 216, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 224, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 232, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 240, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 248, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 256, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 264, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 272, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 280, 1);
        // range_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 288, 1);
        mbarrier_init(smem + 296, 1);
        mbarrier_init(smem + 304, 1);
        // rows_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 312, 1);
        mbarrier_init(smem + 320, 1);
        mbarrier_init(smem + 328, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 336);
    if (warp == 0) {
        int _tmem_hold = smem + 336;
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
    int cta_rank_0 = cta_rank;
    int cluster = bid / 2;
    int gemm_deep = 0;
    int gemm_wide = 0;
    int gemm_layout = 0;
    int gemm_out0 = 0;
    int gemm_out1 = 0;
    int gemm_acc_tiles = 0;
    int swiglu_seq = 0;
    int replay_seq = 0;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int dispatch_seq = 0;
        int combine_units = 0;
        if (tid == 0) {
        }
        __syncthreads();
        int offset = 0;
        int rows = ((macro_size) < (tokens - offset) ? (macro_size) : (tokens - offset));
        #pragma unroll 1
        for (int row = (cluster * 2 + cta_rank_0) * 256 + tid; row < rows; row += comm_sms * 256) {
            int peer = schedule_rank[offset + row];
            int token = schedule_token[offset + row];
            float value = 0.0f;
            if (peer >= 0) {
                value = reinterpret_cast<float*>(weight_peers[peer])[(int)token];
            }
            weights[row] = value;
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(weight_ready))), "r"(static_cast<unsigned int>(1)) : "memory");
            int poll_first;
            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_first) : "l"(weight_ready + 0) : "memory");
            int poll = poll_first;
            while (poll < comm_sms) {
                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                int poll_next;
                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_next) : "l"(weight_ready + 0) : "memory");
                poll = poll_next;
            }
            asm volatile("fence.acquire.gpu;" ::: "memory");
        }
        __syncthreads();
        int col_blocks = (hidden + 511) / 512;
        int unit_rows = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
        int block_units = (128 + unit_rows - 1) / unit_rows;
        int macro_offset = 0;
        int _min_0_1 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
        int macro_blocks = _min_0_1 / 128;
        int blocks = 0;
        if (macro_blocks > cluster * 2 + cta_rank_0) {
            blocks = (macro_blocks - (cluster * 2 + cta_rank_0) + comm_sms - 1) / comm_sms;
        }
        int units = blocks * block_units;
        int _min_0_2 = ((units) < (2) ? (units) : (2));
        #pragma unroll 1
        for (int unit = 0; unit < _min_0_2; unit++) {
            int buffer = unit % 3;
            int block = cluster * 2 + cta_rank_0 + unit / block_units * comm_sms;
            int first_row = block * 128 + unit % block_units * unit_rows;
            int rows_3 = ((unit_rows) < (block * 128 + 128 - first_row) ? (unit_rows) : (block * 128 + 128 - first_row));
            int peer_1 = -1;
            int peer_token = -1;
            if (rows_3 > tid) {
                peer_1 = schedule_rank[macro_offset + first_row + tid];
                peer_token = schedule_token[macro_offset + first_row + tid];
                flat_view_4[tid] = peer_1;
                flat_view_3[buffer * 128 + tid] = ((peer_1 >= 0) ? weights[first_row + tid] : 0.0f);
            }
            uint32_t cta_count = __syncthreads_count(peer_1 >= 0);
            if (tid == 0) {
                mbarrier_arrive_expect_tx(range_arrived_addr + (buffer) * 8, (unsigned int)(cta_count * (unsigned int)hidden * 2));
            }
            if (cta_count < (unsigned int)rows_3) {
                #pragma unroll 1
                for (int pad = 0; pad < rows_3; pad++) {
                    int pad_peer = flat_view_4[pad];
                    if (pad_peer < 0) {
                        #pragma unroll 1
                        for (int vec = tid; vec < hidden / 8; vec += 256) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_2) + (buffer * 65536 + pad * hidden * 2 + vec * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
            }
            __syncthreads();
            if (peer_1 >= 0) {
                cp_async_bulk_gmem2smem(flat_view_addr_1 + (unsigned int)((buffer * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_1])) + (unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden) * (unsigned long long)2), hidden * 2, range_arrived_addr + (buffer) * 8);
            }
        }
        #pragma unroll 1
        for (int unit_1 = 0; unit_1 < units; unit_1++) {
            int buffer_1 = unit_1 % 3;
            int block_1 = cluster * 2 + cta_rank_0 + unit_1 / block_units * comm_sms;
            int first_row_1 = block_1 * 128 + unit_1 % block_units * unit_rows;
            int rows_0 = ((unit_rows) < (block_1 * 128 + 128 - first_row_1) ? (unit_rows) : (block_1 * 128 + 128 - first_row_1));
            int _parity_range_arrived = (int)((unsigned int)unit_1 / 3 % 2);
            mbarrier_wait(range_arrived_addr + (buffer_1) * 8, _parity_range_arrived);
            __syncthreads();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (tid == 0) {
                {
                    void* _cpbulk_dst_1 = reinterpret_cast<void*>(dy_routed_ptr + (unsigned long long)first_row_1 * (unsigned long long)hidden);
                    asm volatile(
                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                        :: "l"(_cpbulk_dst_1), "r"(flat_view_addr_1 + (unsigned int)(buffer_1 * 65536)), "r"((uint32_t)((unsigned int)(rows_0 * hidden * 2)))
                        : "memory");
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            __syncthreads();
            #pragma unroll 4
            for (int word = tid; word < rows_0 * (hidden / 2); word += 256) {
                float weight = flat_view_3[buffer_1 * 128 + word / (hidden / 2)];
                int index = buffer_1 * 16384 + word;
                unsigned int pair = flat_view_2[index];
                uint32_t _bf16x2_scale_pair_2;
                {
                    uint32_t _bf16x2_pair_3 = pair;
                    float _bf16x2_lo_3;
                    float _bf16x2_hi_3;
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_3) : "h"((uint16_t)(_bf16x2_pair_3 & 0xFFFFu)));
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_3) : "h"((uint16_t)(_bf16x2_pair_3 >> 16)));
                    _bf16x2_lo_3 *= weight;
                    _bf16x2_hi_3 *= weight;
                    uint32_t _bf16x2_out_3;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_3) : "f"(_bf16x2_hi_3), "f"(_bf16x2_lo_3));
                    _bf16x2_scale_pair_2 = _bf16x2_out_3;
                }
                flat_view_2[index] = _bf16x2_scale_pair_2;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (tid == 0) {
                {
                    void* _cpbulk_dst_4 = reinterpret_cast<void*>(dy_scaled_ptr + (unsigned long long)first_row_1 * (unsigned long long)hidden);
                    asm volatile(
                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                        :: "l"(_cpbulk_dst_4), "r"(flat_view_addr_1 + (unsigned int)(buffer_1 * 65536)), "r"((uint32_t)((unsigned int)(rows_0 * hidden * 2)))
                        : "memory");
                }
                asm volatile("cp.async.bulk.commit_group;");
                if (units > unit_1 + 2) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                if (unit_1 >= 2 && (unit_1 - 2) % block_units == block_units - 1) {
                    asm volatile("cp.async.bulk.wait_group 4;");
                    int done_block = cluster * 2 + cta_rank_0 + (unit_1 - 2) / block_units * comm_sms;
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset + done_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                }
            }
            if (units > unit_1 + 2) {
                int buffer_0 = (unit_1 + 2) % 3;
                int block_0 = cluster * 2 + cta_rank_0 + (unit_1 + 2) / block_units * comm_sms;
                int first_row_0 = block_0 * 128 + (unit_1 + 2) % block_units * unit_rows;
                int rows_4 = ((unit_rows) < (block_0 * 128 + 128 - first_row_0) ? (unit_rows) : (block_0 * 128 + 128 - first_row_0));
                int peer_2 = -1;
                int peer_token_1 = -1;
                if (rows_4 > tid) {
                    peer_2 = schedule_rank[macro_offset + first_row_0 + tid];
                    peer_token_1 = schedule_token[macro_offset + first_row_0 + tid];
                    flat_view_4[tid] = peer_2;
                    flat_view_3[buffer_0 * 128 + tid] = ((peer_2 >= 0) ? weights[first_row_0 + tid] : 0.0f);
                }
                uint32_t cta_count_1 = __syncthreads_count(peer_2 >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(range_arrived_addr + (buffer_0) * 8, (unsigned int)(cta_count_1 * (unsigned int)hidden * 2));
                }
                if (cta_count_1 < (unsigned int)rows_4) {
                    #pragma unroll 1
                    for (int pad_1 = 0; pad_1 < rows_4; pad_1++) {
                        int pad_peer_1 = flat_view_4[pad_1];
                        if (pad_peer_1 < 0) {
                            #pragma unroll 1
                            for (int vec_1 = tid; vec_1 < hidden / 8; vec_1 += 256) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_2) + (buffer_0 * 65536 + pad_1 * hidden * 2 + vec_1 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                    }
                }
                __syncthreads();
                if (peer_2 >= 0) {
                    cp_async_bulk_gmem2smem(flat_view_addr_1 + (unsigned int)((buffer_0 * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_2])) + (unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden) * (unsigned long long)2), hidden * 2, range_arrived_addr + (buffer_0) * 8);
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
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset + tail_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                }
            }
        }
        #pragma unroll 1
        for (int macro = 0; macro < macros; macro++) {
            int rows_0_1 = ((tokens - macro * macro_size) < (macro_size) ? (tokens - macro * macro_size) : (macro_size));
            int unit_rows_1 = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
            int block_units_2 = (128 + unit_rows_1 - 1) / unit_rows_1;
            int macro_offset_3 = macro * macro_size;
            int _min_5 = ((macro_size) < (tokens - macro_offset_3) ? (macro_size) : (tokens - macro_offset_3));
            int macro_blocks_4 = _min_5 / 128;
            int blocks_5 = 0;
            if (macro_blocks_4 > cluster * 2 + cta_rank_0) {
                blocks_5 = (macro_blocks_4 - (cluster * 2 + cta_rank_0) + comm_sms - 1) / comm_sms;
            }
            int units_6 = blocks_5 * block_units_2;
            int _min_1 = ((units_6) < (2) ? (units_6) : (2));
            #pragma unroll 1
            for (int unit_2 = 0; unit_2 < _min_1; unit_2++) {
                int slot = (combine_units + unit_2) % 3;
                int block_2 = cluster * 2 + cta_rank_0 + unit_2 / block_units_2 * comm_sms;
                int first_row_2 = block_2 * 128 + unit_2 % block_units_2 * unit_rows_1;
                int _min_6 = ((unit_rows_1) < (block_2 * 128 + 128 - first_row_2) ? (unit_rows_1) : (block_2 * 128 + 128 - first_row_2));
                if (_min_6 > tid) {
                    flat_view_5[slot * 128 + tid] = schedule_rank[macro_offset_3 + first_row_2 + tid];
                    flat_view_5[384 + slot * 128 + tid] = schedule_token[macro_offset_3 + first_row_2 + tid];
                }
            }
            if (tid == 0) {
                int _min_2 = ((units_6) < (2) ? (units_6) : (2));
                #pragma unroll 1
                for (int unit_3 = 0; unit_3 < _min_2; unit_3++) {
                    int buffer_2 = (combine_units + unit_3) % 3;
                    int block_3 = cluster * 2 + cta_rank_0 + unit_3 / block_units_2 * comm_sms;
                    int first_row_3 = block_3 * 128 + unit_3 % block_units_2 * unit_rows_1;
                    int rows_5 = ((unit_rows_1) < (block_3 * 128 + 128 - first_row_3) ? (unit_rows_1) : (block_3 * 128 + 128 - first_row_3));
                    int mini = (macro_offset_3 + first_row_3) / mini_size;
                    int mini_rows = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                    int poll_1_first;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_first) : "l"(dx_ready + mini) : "memory");
                    int poll_1 = poll_1_first;
                    while (poll_1 < (mini_rows + 255) / 256 * (hidden / 256) * 2) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int poll_1_next;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_next) : "l"(dx_ready + mini) : "memory");
                        poll_1 = poll_1_next;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_2) * 8, (unsigned int)(rows_5 * hidden * 2));
                    cp_async_bulk_gmem2smem(flat_view_addr_1 + (unsigned int)(buffer_2 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + (unsigned long long)((unsigned long long)first_row_3 * (unsigned long long)hidden) * (unsigned long long)2), rows_5 * hidden * 2, rows_arrived_addr + (buffer_2) * 8);
                }
            }
            __syncthreads();
            #pragma unroll 1
            for (int unit_4 = 0; unit_4 < units_6; unit_4++) {
                int buffer_3 = (combine_units + unit_4) % 3;
                int block_4 = cluster * 2 + cta_rank_0 + unit_4 / block_units_2 * comm_sms;
                int first_row_4 = block_4 * 128 + unit_4 % block_units_2 * unit_rows_1;
                int rows_1 = ((unit_rows_1) < (block_4 * 128 + 128 - first_row_4) ? (unit_rows_1) : (block_4 * 128 + 128 - first_row_4));
                int _parity_rows_arrived = (int)((unsigned int)(combine_units + unit_4) / 3 % 2);
                mbarrier_wait(rows_arrived_addr + (buffer_3) * 8, _parity_rows_arrived);
                if (rows_1 > tid) {
                    int peer_3 = flat_view_5[buffer_3 * 128 + tid];
                    int token_1 = flat_view_5[384 + buffer_3 * 128 + tid];
                    if (peer_3 >= 0) {
                        {
                            float gradient = 0.0f;
                            #pragma unroll 1
                            for (int col = 0; col < intermediate / 128; col++) {
                                gradient = gradient + partials[(first_row_4 + tid) * (intermediate / 128) + col];
                            }
                            reinterpret_cast<float*>(dweight_peers[peer_3])[token_1] = gradient;
                        }
                    }
                }
                if (tid == 0) {
                    #pragma unroll 1
                    for (int row_1 = 0; row_1 < rows_1; row_1++) {
                        int row_peer = flat_view_5[buffer_3 * 128 + row_1];
                        int row_token = flat_view_5[384 + buffer_3 * 128 + row_1];
                        if (row_peer >= 0) {
                            {
                                void* _cpbulk_dst_7 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[row_peer]) + (unsigned long long)row_token * (unsigned long long)hidden);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_7), "r"(flat_view_addr_1 + (unsigned int)(buffer_3 * 65536) + (unsigned int)(row_1 * hidden * 2)), "r"((uint32_t)((unsigned int)(hidden * 2)))
                                    : "memory");
                            }
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    if (units_6 > unit_4 + 2) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        int buffer_1_1 = (combine_units + (unit_4 + 2)) % 3;
                        int block_1_1 = cluster * 2 + cta_rank_0 + (unit_4 + 2) / block_units_2 * comm_sms;
                        int first_row_1_1 = block_1_1 * 128 + (unit_4 + 2) % block_units_2 * unit_rows_1;
                        int rows_6 = ((unit_rows_1) < (block_1_1 * 128 + 128 - first_row_1_1) ? (unit_rows_1) : (block_1_1 * 128 + 128 - first_row_1_1));
                        int mini_1 = (macro_offset_3 + first_row_1_1) / mini_size;
                        int mini_rows_1 = ((mini_size) < (tokens - mini_1 * mini_size) ? (mini_size) : (tokens - mini_1 * mini_size));
                        int poll_2_first;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_first) : "l"(dx_ready + mini_1) : "memory");
                        int poll_2 = poll_2_first;
                        while (poll_2 < (mini_rows_1 + 255) / 256 * (hidden / 256) * 2) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int poll_2_next;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_next) : "l"(dx_ready + mini_1) : "memory");
                            poll_2 = poll_2_next;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_1_1) * 8, (unsigned int)(rows_6 * hidden * 2));
                        cp_async_bulk_gmem2smem(flat_view_addr_1 + (unsigned int)(buffer_1_1 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + (unsigned long long)((unsigned long long)first_row_1_1 * (unsigned long long)hidden) * (unsigned long long)2), rows_6 * hidden * 2, rows_arrived_addr + (buffer_1_1) * 8);
                    }
                }
                if (units_6 > unit_4 + 2) {
                    int slot_1 = (combine_units + (unit_4 + 2)) % 3;
                    int block_2_1 = cluster * 2 + cta_rank_0 + (unit_4 + 2) / block_units_2 * comm_sms;
                    int first_row_2_1 = block_2_1 * 128 + (unit_4 + 2) % block_units_2 * unit_rows_1;
                    int _min_8 = ((unit_rows_1) < (block_2_1 * 128 + 128 - first_row_2_1) ? (unit_rows_1) : (block_2_1 * 128 + 128 - first_row_2_1));
                    if (_min_8 > tid) {
                        flat_view_5[slot_1 * 128 + tid] = schedule_rank[macro_offset_3 + first_row_2_1 + tid];
                        flat_view_5[384 + slot_1 * 128 + tid] = schedule_token[macro_offset_3 + first_row_2_1 + tid];
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
                {
                    bool enabled_value = macros > 1;
                    if (enabled_value != 0) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
            combine_units += units_6;
            if (macros > macro + 1) {
                int minis_0 = (rows_0_1 + mini_size - 1) / mini_size;
                int required = 2 * (minis_0 * mini_bwd + wgrad_tasks) + comm_sms;
                if (tid == 0) {
                    bool enabled_value_1 = 1;
                    if (enabled_value_1 != 0) {
                        int poll_3_first;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_first) : "l"(buffers_done + macro) : "memory");
                        int poll_3 = poll_3_first;
                        while (poll_3 < required) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int poll_3_next;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_next) : "l"(buffers_done + macro) : "memory");
                            poll_3 = poll_3_next;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                    }
                }
                __syncthreads();
                int offset_1 = (macro + 1) * macro_size;
                int rows_2 = ((macro_size) < (tokens - offset_1) ? (macro_size) : (tokens - offset_1));
                #pragma unroll 1
                for (int row_2 = (cluster * 2 + cta_rank_0) * 256 + tid; row_2 < rows_2; row_2 += comm_sms * 256) {
                    int peer_4 = schedule_rank[offset_1 + row_2];
                    int token_2 = schedule_token[offset_1 + row_2];
                    float value_1 = 0.0f;
                    if (peer_4 >= 0) {
                        value_1 = reinterpret_cast<float*>(weight_peers[peer_4])[(int)token_2];
                    }
                    weights[row_2] = value_1;
                }
                __syncthreads();
                if (tid == 0) {
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(weight_ready)) + (macro + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    int poll_4_first;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_first) : "l"(weight_ready + (macro + 1)) : "memory");
                    int poll_4 = poll_4_first;
                    while (poll_4 < comm_sms) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int poll_4_next;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_next) : "l"(weight_ready + (macro + 1)) : "memory");
                        poll_4 = poll_4_next;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                }
                __syncthreads();
                int next_rows = ((tokens - (macro + 1) * macro_size) < (macro_size) ? (tokens - (macro + 1) * macro_size) : (macro_size));
                int next_tasks = next_rows / 128 * ((hidden + 511) / 512);
                #pragma unroll 1
                for (int task = cluster * 2 + cta_rank_0; task < next_tasks; task += comm_sms) {
                    int col_blocks_0 = (hidden + 511) / 512;
                    int macro_offset_1 = (macro + 1) * macro_size;
                    int macro_tokens = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                    if (task < macro_tokens / 128 * col_blocks_0) {
                        int row_3 = task / col_blocks_0 * 128;
                        int col_block = task % col_blocks_0;
                        int chunk_cols = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                        unsigned int chunk_bytes = (unsigned int)(chunk_cols * 2);
                        int peer_5 = -1;
                        int peer_token_2 = -1;
                        if (tid < 128) {
                            peer_5 = schedule_rank[macro_offset_1 + row_3 + tid];
                            peer_token_2 = schedule_token[macro_offset_1 + row_3 + tid];
                        }
                        uint32_t _cta_count_2 = __syncthreads_count(peer_5 >= 0);
                        if (tid == 0) {
                            int previous_offset = -1 * macro_size;
                            int previous_tokens = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                            if (row_3 < previous_tokens) {
                                int previous_mini = (previous_offset + row_3) / mini_size;
                                int mini_rows_2 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                                int required_0 = (mini_rows_2 + 255) / 256 * (hidden / 256) * 2;
                            }
                            mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_2 * chunk_bytes);
                        }
                        __syncthreads();
                        if (peer_5 >= 0) {
                            cp_async_bulk_gmem2smem(flat_view_addr_1 + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_5])) + (unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)) * (unsigned long long)2), chunk_cols * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_2 = 0; vec_2 < 64; vec_2++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_1) + (tid * 1024 + vec_2 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                        int _parity_dispatch_arrived = (int)((unsigned int)dispatch_seq % 2);
                        mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived);
                        dispatch_seq += 1;
                        if (tid < 128) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_9 = reinterpret_cast<void*>(x_routed_ptr + ((unsigned long long)(row_3 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_9), "r"(flat_view_addr_1 + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group 0;");
                        }
                        __syncthreads();
                        if (tid == 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_x)) + ((macro_offset_1 + row_3) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                }
                #pragma unroll 1
                for (int task_1 = cluster * 2 + cta_rank_0; task_1 < next_tasks; task_1 += comm_sms) {
                    int col_blocks_0_1 = (hidden + 511) / 512;
                    int macro_offset_1_1 = (macro + 1) * macro_size;
                    int macro_tokens_1 = ((macro_size) < (tokens - macro_offset_1_1) ? (macro_size) : (tokens - macro_offset_1_1));
                    if (task_1 < macro_tokens_1 / 128 * col_blocks_0_1) {
                        int row_4 = task_1 / col_blocks_0_1 * 128;
                        int col_block_1 = task_1 % col_blocks_0_1;
                        int chunk_cols_1 = ((512) < (hidden - col_block_1 * 512) ? (512) : (hidden - col_block_1 * 512));
                        unsigned int chunk_bytes_1 = (unsigned int)(chunk_cols_1 * 2);
                        int peer_6 = -1;
                        int peer_token_3 = -1;
                        if (tid < 128) {
                            peer_6 = schedule_rank[macro_offset_1_1 + row_4 + tid];
                            peer_token_3 = schedule_token[macro_offset_1_1 + row_4 + tid];
                        }
                        uint32_t _cta_count_3 = __syncthreads_count(peer_6 >= 0);
                        if (tid == 0) {
                            int previous_offset_1 = -1 * macro_size;
                            int previous_tokens_1 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                            if (row_4 < previous_tokens_1) {
                                int previous_mini_1 = (previous_offset_1 + row_4) / mini_size;
                                int mini_rows_3 = ((mini_size) < (tokens - previous_mini_1 * mini_size) ? (mini_size) : (tokens - previous_mini_1 * mini_size));
                                int required_0_1 = (mini_rows_3 + 255) / 256 * (hidden / 256) * 2;
                            }
                            mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_3 * chunk_bytes_1);
                        }
                        __syncthreads();
                        if (tid < 128) {
                            flat_view_6[tid] = ((peer_6 >= 0) ? weights[row_4 + tid] : 0.0f);
                        }
                        if (peer_6 >= 0) {
                            cp_async_bulk_gmem2smem(flat_view_addr_1 + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_6])) + (unsigned long long)((unsigned long long)(peer_token_3 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)) * (unsigned long long)2), chunk_cols_1 * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_3 = 0; vec_3 < 64; vec_3++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_1) + (tid * 1024 + vec_3 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                        int _parity_dispatch_arrived_1 = (int)((unsigned int)dispatch_seq % 2);
                        mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived_1);
                        dispatch_seq += 1;
                        __syncthreads();
                        if (tid < 128) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_10 = reinterpret_cast<void*>(dy_routed_ptr + ((unsigned long long)(row_4 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_10), "r"(flat_view_addr_1 + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes_1))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid < chunk_cols_1 / 2) {
                            #pragma unroll 1
                            for (int scale_row = 0; scale_row < 64; scale_row++) {
                                float weight_1 = flat_view_6[scale_row];
                                unsigned int pair_1 = flat_view_2[scale_row * 256 + tid];
                                uint32_t _bf16x2_scale_pair_11;
                                {
                                    uint32_t _bf16x2_pair_12 = pair_1;
                                    float _bf16x2_lo_12;
                                    float _bf16x2_hi_12;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_12) : "h"((uint16_t)(_bf16x2_pair_12 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_12) : "h"((uint16_t)(_bf16x2_pair_12 >> 16)));
                                    _bf16x2_lo_12 *= weight_1;
                                    _bf16x2_hi_12 *= weight_1;
                                    uint32_t _bf16x2_out_12;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_12) : "f"(_bf16x2_hi_12), "f"(_bf16x2_lo_12));
                                    _bf16x2_scale_pair_11 = _bf16x2_out_12;
                                }
                                flat_view_2[32768 + scale_row * 256 + tid] = _bf16x2_scale_pair_11;
                            }
                        }
                        if (tid < 128) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        __syncthreads();
                        if (tid < chunk_cols_1 / 2) {
                            #pragma unroll 1
                            for (int scale_row_1 = 64; scale_row_1 < 128; scale_row_1++) {
                                float weight_2 = flat_view_6[scale_row_1];
                                unsigned int pair_2 = flat_view_2[scale_row_1 * 256 + tid];
                                uint32_t _bf16x2_scale_pair_13;
                                {
                                    uint32_t _bf16x2_pair_14 = pair_2;
                                    float _bf16x2_lo_14;
                                    float _bf16x2_hi_14;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_14) : "h"((uint16_t)(_bf16x2_pair_14 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_14) : "h"((uint16_t)(_bf16x2_pair_14 >> 16)));
                                    _bf16x2_lo_14 *= weight_2;
                                    _bf16x2_hi_14 *= weight_2;
                                    uint32_t _bf16x2_out_14;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_14) : "f"(_bf16x2_hi_14), "f"(_bf16x2_lo_14));
                                    _bf16x2_scale_pair_13 = _bf16x2_out_14;
                                }
                                flat_view_2[scale_row_1 * 256 + tid] = _bf16x2_scale_pair_13;
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncthreads();
                        if (tid < 128) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_15 = reinterpret_cast<void*>(dy_scaled_ptr + ((unsigned long long)(row_4 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_15), "r"(flat_view_addr_1 + (unsigned int)(tid * 1024 + (1 - tid / 64) * 131072)), "r"((uint32_t)(chunk_bytes_1))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group 0;");
                        }
                        __syncthreads();
                        if (tid == 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset_1_1 + row_4) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                }
            }
        }
    } else {
        int iteration = 0;
        int kind = -1;
        int task_2 = 0;
        int macro_1 = 0;
        int mini_2 = 0;
        int shared = 0;
        if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
            if (shared_tasks > cluster - comm_clusters) {
                shared = 1;
                if (shared_down > cluster - comm_clusters) {
                    kind = 0;
                    task_2 = cluster - comm_clusters;
                } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                    kind = 1;
                    task_2 = cluster - comm_clusters - shared_down;
                } else {
                    if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                        kind = 2;
                        task_2 = cluster - comm_clusters - shared_down - shared_swiglu;
                    } else {
                        int weight_task = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                        uint32_t fast_div_q = (shared_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)weight_task), shared_wgrad_div.multiplier) >> shared_wgrad_div.shift_right) : (uint32_t)((unsigned int)weight_task);
                        uint32_t fast_div_r = (uint32_t)((unsigned int)weight_task) - fast_div_q * (uint32_t)(shared_wgrad_div.divisor);
                        kind = (int)fast_div_q + 3;
                        task_2 = (int)fast_div_r;
                    }
                }
            } else {
                int routed = cluster - comm_clusters - shared_tasks;
                int macro_task = routed;
                int replay_tasks = 0;
                if (routed >= saved_tasks) {
                    uint32_t fast_div_q_1 = (replay_macro_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(routed - saved_tasks)), replay_macro_tasks_div.multiplier) >> replay_macro_tasks_div.shift_right) : (uint32_t)((unsigned int)(routed - saved_tasks));
                    uint32_t fast_div_r_1 = (uint32_t)((unsigned int)(routed - saved_tasks)) - fast_div_q_1 * (uint32_t)(replay_macro_tasks_div.divisor);
                    macro_1 = (int)fast_div_q_1 + 1;
                    macro_task = (int)fast_div_r_1;
                }
                int _min_16 = ((tokens - macro_1 * macro_size) < (macro_size) ? (tokens - macro_1 * macro_size) : (macro_size));
                uint32_t fast_div_q_2 = (mini_size_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(_min_16 + mini_size - 1)), mini_size_div.multiplier) >> mini_size_div.shift_right) : (uint32_t)((unsigned int)(_min_16 + mini_size - 1));
                uint32_t fast_div_r_2 = (uint32_t)((unsigned int)(_min_16 + mini_size - 1)) - fast_div_q_2 * (uint32_t)(mini_size_div.divisor);
                int macro_minis = (int)fast_div_q_2;
                if (routed >= saved_tasks) {
                    replay_tasks = macro_minis * mini_replay;
                }
                if (macro_task < replay_tasks) {
                    uint32_t fast_div_q_3 = (mini_replay_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)macro_task), mini_replay_div.multiplier) >> mini_replay_div.shift_right) : (uint32_t)((unsigned int)macro_task);
                    uint32_t fast_div_r_3 = (uint32_t)((unsigned int)macro_task) - fast_div_q_3 * (uint32_t)(mini_replay_div.divisor);
                    mini_2 = (int)fast_div_q_3;
                    int mini_task = (int)fast_div_r_3;
                    if (mini_task < mini_down) {
                        kind = 6;
                        task_2 = mini_task;
                    } else if (mini_task < 2 * mini_down) {
                        kind = 7;
                        task_2 = mini_task - mini_down;
                    } else {
                        kind = 8;
                        task_2 = mini_task - 2 * mini_down;
                    }
                } else {
                    int bwd_task = macro_task - replay_tasks;
                    if (bwd_task < macro_minis * mini_bwd) {
                        int fill = ((macro_minis - 1) < (1) ? (macro_minis - 1) : (1));
                        int slot_down = fill * mini_down;
                        int end0 = mini_down;
                        int end1 = end0 + slot_down + mini_swiglu;
                        int _max_17 = ((macro_minis - 2) > (0) ? (macro_minis - 2) : (0));
                        int end2 = end1 + _max_17 * mini_bwd;
                        int end3 = end2 + fill * (mini_bwd - mini_down);
                        if (bwd_task < end0) {
                            kind = 0;
                            mini_2 = 0;
                            task_2 = bwd_task;
                        } else if (bwd_task < end1) {
                            int fill_task = bwd_task - end0;
                            if (fill_task < slot_down) {
                                kind = 0;
                                mini_2 = 1;
                                task_2 = fill_task;
                            } else {
                                kind = 1;
                                mini_2 = 0;
                                task_2 = fill_task - slot_down;
                            }
                        } else {
                            if (bwd_task < end2) {
                                uint32_t fast_div_q_4 = (mini_bwd_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(bwd_task - end1)), mini_bwd_div.multiplier) >> mini_bwd_div.shift_right) : (uint32_t)((unsigned int)(bwd_task - end1));
                                uint32_t fast_div_r_4 = (uint32_t)((unsigned int)(bwd_task - end1)) - fast_div_q_4 * (uint32_t)(mini_bwd_div.divisor);
                                int slot_2 = (int)fast_div_q_4 + 2;
                                int slot_task = (int)fast_div_r_4;
                                if (slot_task < mini_down) {
                                    kind = 0;
                                    mini_2 = slot_2;
                                    task_2 = slot_task;
                                } else if (slot_task < mini_down + mini_swiglu) {
                                    kind = 1;
                                    mini_2 = slot_2 - 1;
                                    task_2 = slot_task - mini_down;
                                } else {
                                    kind = 2;
                                    mini_2 = slot_2 - 2;
                                    task_2 = slot_task - mini_down - mini_swiglu;
                                }
                            } else if (bwd_task < end3) {
                                int drain_task = bwd_task - end2;
                                if (drain_task < mini_swiglu) {
                                    kind = 1;
                                    mini_2 = macro_minis - 1;
                                    task_2 = drain_task;
                                } else {
                                    kind = 2;
                                    mini_2 = macro_minis - 2;
                                    task_2 = drain_task - mini_swiglu;
                                }
                            } else {
                                kind = 2;
                                mini_2 = macro_minis - 1;
                                task_2 = bwd_task - end3;
                            }
                        }
                    } else {
                        int weight_task_1 = bwd_task - macro_minis * mini_bwd;
                        uint32_t fast_div_q_5 = (weight_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)weight_task_1), weight_tasks_div.multiplier) >> weight_tasks_div.shift_right) : (uint32_t)((unsigned int)weight_task_1);
                        uint32_t fast_div_r_5 = (uint32_t)((unsigned int)weight_task_1) - fast_div_q_5 * (uint32_t)(weight_tasks_div.divisor);
                        kind = (int)fast_div_q_5 + 3;
                        task_2 = (int)fast_div_r_5;
                    }
                }
            }
        }
        int tile_word_0 = kind;
        int tile_word_1 = task_2;
        int tile_word_2 = macro_1;
        int tile_word_3 = mini_2;
        int tile_word_4 = shared;
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
            int row_count = 2 * (intermediate / 128);
            int shared_rows = 0;
            int shared_down_0 = shared_rows * (intermediate / 256);
            if (tile_word_4 != 0) {
            } else if (tile_word_0 == 6) {
                int col_blocks_1 = (intermediate + 512 - 1) / 512;
                int x = -1;
                int y = -1;
                int expert = -1;
                int k_start = 0;
                int k_end = 0;
                int first = 0;
                int first_block = (tile_word_2 * (macro_size / mini_size) + tile_word_3) * (mini_size / 256);
                int end_block = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                int block_5 = first_block;
                uint32_t fast_div_q_6 = (replay_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), replay_col_blocks_div.multiplier) >> replay_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                uint32_t fast_div_r_6 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_6 * (uint32_t)(replay_col_blocks_div.divisor);
                block_5 = first_block + (int)fast_div_q_6;
                if (block_5 < end_block) {
                    int index_1 = counts[3 * experts + block_5];
                    int offset_2 = counts[experts + index_1] / 256;
                    int first_row_5 = ((first_block) > (offset_2) ? (first_block) : (offset_2));
                    int _min_18 = ((end_block) < (offset_2 + counts[index_1] / 256) ? (end_block) : (offset_2 + counts[index_1] / 256));
                    int rows_7 = _min_18 - first_row_5;
                    int supergroup = (tile_word_1 - (first_row_5 - first_block) * col_blocks_1) / (rows_7 * 8);
                    int full_cols = col_blocks_1 / 8 * 8;
                    int row_5 = 0;
                    int col_1 = 0;
                    if (tile_word_1 - (first_row_5 - first_block) * col_blocks_1 < rows_7 * full_cols) {
                        row_5 = (tile_word_1 - (first_row_5 - first_block) * col_blocks_1) % (rows_7 * 8) / 8;
                        col_1 = supergroup * 8 + (tile_word_1 - (first_row_5 - first_block) * col_blocks_1) % 8;
                    } else {
                        row_5 = (tile_word_1 - (first_row_5 - first_block) * col_blocks_1 - rows_7 * full_cols) / (col_blocks_1 - full_cols);
                        col_1 = full_cols + (tile_word_1 - (first_row_5 - first_block) * col_blocks_1 - rows_7 * full_cols) % (col_blocks_1 - full_cols);
                    }
                    if ((supergroup & 1) != 0) {
                        row_5 = rows_7 - row_5 - 1;
                    }
                    x = first_row_5 + row_5 - tile_word_2 * (macro_size / 256);
                    y = col_1;
                    expert = index_1;
                }
                int has_hi = 0;
                has_hi = (int)((y * 2 + 1) * 256 < intermediate);
                int global_mini = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                int macro_rows = tile_word_2 * (macro_size / 256);
                int iterations = hidden / 64;
                if (expert < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                int _min_19 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                int mini_rows_4 = ((0) > (_min_19) ? (0) : (_min_19));
                                int required_1 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                bool enabled_value_2 = 1;
                                if (enabled_value_2 != 0) {
                                    int poll_5_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_first) : "l"(replay_x + global_mini) : "memory");
                                    int poll_5 = poll_5_first;
                                    while (poll_5 < required_1) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_5_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_next) : "l"(replay_x + global_mini) : "memory");
                                        poll_5 = poll_5_next;
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
                                    :: "r"(a_nt_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&x_nt_r)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(idx), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_nt_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_nt_r)), "r"(0), "r"(y * 2 * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_nt_hi_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_nt_r)), "r"(0), "r"((y * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int head = ((3) < (iterations) ? (3) : (iterations));
                                int tail_1 = 0;
                                int _min_20 = ((3) < (iterations - head) ? (3) : (iterations - head));
                                tail_1 = _min_20;
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
                                    int _smem_desc_lo_21 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    int _smem_desc_lo_22 = (((b_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_1) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_21), "r"(_smem_desc_lo_22), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                }
                                int _stage_output_finished_1 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                int _parity_output_finished_1 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                mbarrier_wait(output_finished_addr + (_stage_output_finished_1) * 8, _parity_output_finished_1);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_2 = 0; idx_2 < head; idx_2++) {
                                    int _smem_desc_lo_23 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_2) % 4) * 1024;
                                    int _smem_desc_lo_24 = (((b_nt_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_2) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_23), "r"(_smem_desc_lo_24), "r"((tmem_accumulator + (256))), "r"(((idx_2 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8, (uint16_t)(3));
                                }
                                #pragma unroll 1
                                for (int idx_3 = head; idx_3 < iterations - tail_1; idx_3++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) % 4);
                                    int _parity_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_1) * 8, _parity_gemm_arrived_1);
                                    int _smem_desc_lo_25 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_26 = (((b_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_25), "r"(_smem_desc_lo_26), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    int _smem_desc_lo_27 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_28 = (((b_nt_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_27), "r"(_smem_desc_lo_28), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, (uint16_t)(3));
                                }
                                #pragma unroll 1
                                for (int idx_4 = iterations - tail_1; idx_4 < iterations; idx_4++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_4) % 4);
                                    int _parity_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_4) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_2) * 8, _parity_gemm_arrived_2);
                                    int _smem_desc_lo_29 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_4) % 4) * 1024;
                                    int _smem_desc_lo_30 = (((b_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_4) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_29), "r"(_smem_desc_lo_30), "r"(tmem_accumulator), "r"(((idx_4 == 0) ? 0 : 1)) : "memory");
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                #pragma unroll 1
                                for (int idx_5 = iterations - tail_1; idx_5 < iterations; idx_5++) {
                                    int _smem_desc_lo_31 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_5) % 4) * 1024;
                                    int _smem_desc_lo_32 = (((b_nt_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_5) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_31), "r"(_smem_desc_lo_32), "r"((tmem_accumulator + (256))), "r"(((idx_5 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_5) % 4)) * 8, (uint16_t)(3));
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        int _parity_output_arrived = (int)((unsigned int)gemm_out0 % 2);
                        mbarrier_wait(output_arrived_addr, _parity_output_arrived);
                        unsigned int packed[128];
                        #pragma unroll
                        for (int chunk = 0; chunk < 8; chunk++) {
                            #pragma unroll
                            for (int sub = 0; sub < 2; sub++) {
                                unsigned int address = taddr_1 + (unsigned int)(tid / 32 * 32 + sub * 16 << 16) + (unsigned int)(chunk * 32);
                                float tmem_load[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[15]))
                                    : "r"(address));
                                #pragma unroll
                                for (int pair_3 = 0; pair_3 < 8; pair_3++) {
                                    packed[chunk * 16 + sub * 8 + pair_3] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load[pair_3 * 2], tmem_load[pair_3 * 2 + 1])));
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
                            int previous_offset_2 = (tile_word_2 + 1) * macro_size;
                            int output_row = x * 256 + cta_rank_0 * 128;
                            int _min_33 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                            if (output_row < _min_33) {
                            }
                        }
                        #pragma unroll
                        for (int chunk_1 = 0; chunk_1 < 8; chunk_1++) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_0 = tid / 32;
                            int lane_1 = tid % 32;
                            #pragma unroll
                            for (int half = 0; half < 2; half++) {
                                #pragma unroll
                                for (int col_tile = 0; col_tile < 2; col_tile++) {
                                    int row_6 = warp_0 * 32 + half * 16 + lane_1 % 16;
                                    int col_2 = col_tile * 16 + lane_1 / 16 * 8;
                                    unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_6 * 32 + col_2) * 2);
                                    address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                    int offset_3 = chunk_1 * 16 + half * 8 + col_tile * 4;
                                    uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(address_1);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_3 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x * 256 + cta_rank_0 * 128, y * 2 * 8 + chunk_1, 0, 0, d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192));
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        int _parity_output_arrived_1 = (int)((unsigned int)gemm_out1 % 2);
                        mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_1);
                        if (has_hi != 0) {
                            unsigned int packed_0[128];
                            #pragma unroll
                            for (int chunk_2 = 0; chunk_2 < 8; chunk_2++) {
                                #pragma unroll
                                for (int sub_1 = 0; sub_1 < 2; sub_1++) {
                                    unsigned int address_2 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_1 * 16 << 16) + 256 + (unsigned int)(chunk_2 * 32);
                                    float tmem_load_1[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[15]))
                                        : "r"(address_2));
                                    #pragma unroll
                                    for (int pair_4 = 0; pair_4 < 8; pair_4++) {
                                        packed_0[chunk_2 * 16 + sub_1 * 8 + pair_4] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_1[pair_4 * 2], tmem_load_1[pair_4 * 2 + 1])));
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            int last_1 = 1;
                            if (last_1 != 0) {
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
                                int warp_0_1 = tid / 32;
                                int lane_2 = tid % 32;
                                #pragma unroll
                                for (int half_1 = 0; half_1 < 2; half_1++) {
                                    #pragma unroll
                                    for (int col_tile_1 = 0; col_tile_1 < 2; col_tile_1++) {
                                        int row_7 = warp_0_1 * 32 + half_1 * 16 + lane_2 % 16;
                                        int col_3 = col_tile_1 * 16 + lane_2 / 16 * 8;
                                        unsigned int address_3 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_7 * 32 + col_3) * 2);
                                        address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                        int offset_4 = chunk_3 * 16 + half_1 * 8 + col_tile_1 * 4;
                                        uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(address_3);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_4 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_out_r), 0, x * 256 + cta_rank_0 * 128, (y * 2 + 1) * 8 + chunk_3, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (has_hi == 0) {
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
                                    bool enabled_value_3 = 1;
                                    if (enabled_value_3 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows + x) * (intermediate / 256) + y * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                    if (has_hi != 0) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        bool enabled_value_0 = 1;
                                        if (enabled_value_0 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows + x) * (intermediate / 256) + y * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                int live = (int)(expert >= 0);
                gemm_wide += iterations * live;
                gemm_out0 += live;
                gemm_out1 += live;
                gemm_acc_tiles += 2 * live;
            } else {
                if (tile_word_0 == 7) {
                    int col_blocks_2 = (intermediate + 512 - 1) / 512;
                    int x_1 = -1;
                    int y_1 = -1;
                    int expert_1 = -1;
                    int k_start_1 = 0;
                    int k_end_1 = 0;
                    int first_1 = 0;
                    int first_block_1 = (tile_word_2 * (macro_size / mini_size) + tile_word_3) * (mini_size / 256);
                    int end_block_1 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int block_6 = first_block_1;
                    uint32_t fast_div_q_7 = (replay_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), replay_col_blocks_div.multiplier) >> replay_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                    uint32_t fast_div_r_7 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_7 * (uint32_t)(replay_col_blocks_div.divisor);
                    block_6 = first_block_1 + (int)fast_div_q_7;
                    if (block_6 < end_block_1) {
                        int index_2 = counts[3 * experts + block_6];
                        int offset_5 = counts[experts + index_2] / 256;
                        int first_row_6 = ((first_block_1) > (offset_5) ? (first_block_1) : (offset_5));
                        int _min_36 = ((end_block_1) < (offset_5 + counts[index_2] / 256) ? (end_block_1) : (offset_5 + counts[index_2] / 256));
                        int rows_8 = _min_36 - first_row_6;
                        int supergroup_1 = (tile_word_1 - (first_row_6 - first_block_1) * col_blocks_2) / (rows_8 * 8);
                        int full_cols_1 = col_blocks_2 / 8 * 8;
                        int row_8 = 0;
                        int col_4 = 0;
                        if (tile_word_1 - (first_row_6 - first_block_1) * col_blocks_2 < rows_8 * full_cols_1) {
                            row_8 = (tile_word_1 - (first_row_6 - first_block_1) * col_blocks_2) % (rows_8 * 8) / 8;
                            col_4 = supergroup_1 * 8 + (tile_word_1 - (first_row_6 - first_block_1) * col_blocks_2) % 8;
                        } else {
                            row_8 = (tile_word_1 - (first_row_6 - first_block_1) * col_blocks_2 - rows_8 * full_cols_1) / (col_blocks_2 - full_cols_1);
                            col_4 = full_cols_1 + (tile_word_1 - (first_row_6 - first_block_1) * col_blocks_2 - rows_8 * full_cols_1) % (col_blocks_2 - full_cols_1);
                        }
                        if ((supergroup_1 & 1) != 0) {
                            row_8 = rows_8 - row_8 - 1;
                        }
                        x_1 = first_row_6 + row_8 - tile_word_2 * (macro_size / 256);
                        y_1 = col_4;
                        expert_1 = index_2;
                    }
                    int has_hi_1 = 0;
                    has_hi_1 = (int)((y_1 * 2 + 1) * 256 < intermediate);
                    int global_mini_1 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                    int macro_rows_1 = tile_word_2 * (macro_size / 256);
                    int iterations_1 = hidden / 64;
                    if (expert_1 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_37 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                    int mini_rows_5 = ((0) > (_min_37) ? (0) : (_min_37));
                                    int required_2 = (mini_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_4 = 1;
                                    if (enabled_value_4 != 0) {
                                        int poll_6_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_first) : "l"(replay_x + global_mini_1) : "memory");
                                        int poll_6 = poll_6_first;
                                        while (poll_6 < required_2) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_6_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_next) : "l"(replay_x + global_mini_1) : "memory");
                                            poll_6 = poll_6_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                #pragma unroll 1
                                for (int idx_6 = 0; idx_6 < iterations_1; idx_6++) {
                                    int _stage_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_6) % 4);
                                    int _parity_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_6) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_1) * 8, _parity_gemm_finished_1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_nt_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&x_nt_r)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_nt_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&wu_nt_r)), "r"(0), "r"(y_1 * 2 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_1), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_nt_hi_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&wu_nt_r)), "r"(0), "r"((y_1 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_1), "r"(0),
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
                                    int _min_38 = ((3) < (iterations_1 - head_1) ? (3) : (iterations_1 - head_1));
                                    tail_2 = _min_38;
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
                                        int _smem_desc_lo_39 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_7) % 4) * 1024;
                                        int _smem_desc_lo_40 = (((b_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_7) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_39), "r"(_smem_desc_lo_40), "r"(tmem_accumulator), "r"(((idx_7 == 0) ? 0 : 1)) : "memory");
                                    }
                                    int _stage_output_finished_3 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                    int _parity_output_finished_3 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr + (_stage_output_finished_3) * 8, _parity_output_finished_3);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_8 = 0; idx_8 < head_1; idx_8++) {
                                        int _smem_desc_lo_41 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_8) % 4) * 1024;
                                        int _smem_desc_lo_42 = (((b_nt_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_8) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_41), "r"(_smem_desc_lo_42), "r"((tmem_accumulator + (256))), "r"(((idx_8 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_8) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_9 = head_1; idx_9 < iterations_1 - tail_2; idx_9++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_4 = (int)((unsigned int)(gemm_wide + idx_9) % 4);
                                        int _parity_gemm_arrived_4 = (int)((unsigned int)(gemm_wide + idx_9) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_4) * 8, _parity_gemm_arrived_4);
                                        int _smem_desc_lo_43 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                        int _smem_desc_lo_44 = (((b_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_43), "r"(_smem_desc_lo_44), "r"(tmem_accumulator), "r"(((idx_9 == 0) ? 0 : 1)) : "memory");
                                        int _smem_desc_lo_45 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                        int _smem_desc_lo_46 = (((b_nt_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_45), "r"(_smem_desc_lo_46), "r"((tmem_accumulator + (256))), "r"(((idx_9 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_10 = iterations_1 - tail_2; idx_10 < iterations_1; idx_10++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_10) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_5 = (int)((unsigned int)(gemm_wide + idx_10) % 4);
                                        int _parity_gemm_arrived_5 = (int)((unsigned int)(gemm_wide + idx_10) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_5) * 8, _parity_gemm_arrived_5);
                                        int _smem_desc_lo_47 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_10) % 4) * 1024;
                                        int _smem_desc_lo_48 = (((b_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_10) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_47), "r"(_smem_desc_lo_48), "r"(tmem_accumulator), "r"(((idx_10 == 0) ? 0 : 1)) : "memory");
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    #pragma unroll 1
                                    for (int idx_11 = iterations_1 - tail_2; idx_11 < iterations_1; idx_11++) {
                                        int _smem_desc_lo_49 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_11) % 4) * 1024;
                                        int _smem_desc_lo_50 = (((b_nt_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_11) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_49), "r"(_smem_desc_lo_50), "r"((tmem_accumulator + (256))), "r"(((idx_11 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_11) % 4)) * 8, (uint16_t)(3));
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            int _parity_output_arrived_2 = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_2);
                            unsigned int packed_1[128];
                            #pragma unroll
                            for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                #pragma unroll
                                for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                    unsigned int address_4 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                    float tmem_load_2[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[15]))
                                        : "r"(address_4));
                                    #pragma unroll
                                    for (int pair_5 = 0; pair_5 < 8; pair_5++) {
                                        packed_1[chunk_4 * 16 + sub_2 * 8 + pair_5] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_2[pair_5 * 2], tmem_load_2[pair_5 * 2 + 1])));
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            int last_2 = 1;
                            if (last_2 != 0) {
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            if (tid == 0) {
                                int previous_offset_3 = (tile_word_2 + 1) * macro_size;
                                int output_row_1 = x_1 * 256 + cta_rank_0 * 128;
                                int _min_51 = ((macro_size) < (tokens - previous_offset_3) ? (macro_size) : (tokens - previous_offset_3));
                                if (output_row_1 < _min_51) {
                                }
                            }
                            #pragma unroll
                            for (int chunk_5 = 0; chunk_5 < 8; chunk_5++) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                int warp_0_2 = tid / 32;
                                int lane_3 = tid % 32;
                                #pragma unroll
                                for (int half_2 = 0; half_2 < 2; half_2++) {
                                    #pragma unroll
                                    for (int col_tile_2 = 0; col_tile_2 < 2; col_tile_2++) {
                                        int row_9 = warp_0_2 * 32 + half_2 * 16 + lane_3 % 16;
                                        int col_5 = col_tile_2 * 16 + lane_3 / 16 * 8;
                                        unsigned int address_5 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_9 * 32 + col_5) * 2);
                                        address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                        int offset_6 = chunk_5 * 16 + half_2 * 8 + col_tile_2 * 4;
                                        uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(address_5);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_6 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_1 * 256 + cta_rank_0 * 128, y_1 * 2 * 8 + chunk_5, 0, 0, d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            int _parity_output_arrived_3 = (int)((unsigned int)gemm_out1 % 2);
                            mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_3);
                            if (has_hi_1 != 0) {
                                unsigned int packed_0_1[128];
                                #pragma unroll
                                for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                    #pragma unroll
                                    for (int sub_3 = 0; sub_3 < 2; sub_3++) {
                                        unsigned int address_6 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_6 * 32);
                                        float tmem_load_1_1[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[15]))
                                            : "r"(address_6));
                                        #pragma unroll
                                        for (int pair_6 = 0; pair_6 < 8; pair_6++) {
                                            packed_0_1[chunk_6 * 16 + sub_3 * 8 + pair_6] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_1_1[pair_6 * 2], tmem_load_1_1[pair_6 * 2 + 1])));
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_1_1 = 1;
                                if (last_1_1 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                #pragma unroll
                                for (int chunk_7 = 0; chunk_7 < 8; chunk_7++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_3 = tid / 32;
                                    int lane_4 = tid % 32;
                                    #pragma unroll
                                    for (int half_3 = 0; half_3 < 2; half_3++) {
                                        #pragma unroll
                                        for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                            int row_10 = warp_0_3 * 32 + half_3 * 16 + lane_4 % 16;
                                            int col_6 = col_tile_3 * 16 + lane_4 / 16 * 8;
                                            unsigned int address_7 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_10 * 32 + col_6) * 2);
                                            address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                            int offset_7 = chunk_7 * 16 + half_3 * 8 + col_tile_3 * 4;
                                            uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(address_7);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_7 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_out_r), 0, x_1 * 256 + cta_rank_0 * 128, (y_1 * 2 + 1) * 8 + chunk_7, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192));
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
                                        bool enabled_value_5 = 1;
                                        if (enabled_value_5 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows_1 + x_1) * (intermediate / 256) + y_1 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                        if (has_hi_1 != 0) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            bool enabled_value_1_1 = 1;
                                            if (enabled_value_1_1 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows_1 + x_1) * (intermediate / 256) + y_1 * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int live_1 = (int)(expert_1 >= 0);
                    gemm_wide += iterations_1 * live_1;
                    gemm_out0 += live_1;
                    gemm_out1 += live_1;
                    gemm_acc_tiles += 2 * live_1;
                } else if (tile_word_0 == 8) {
                    int col_blocks_3 = intermediate / 128;
                    int num_tiles = tokens / 128 * col_blocks_3;
                    int macro_row_offset = tile_word_2 * (macro_size / 128);
                    int first_tile = tile_word_1 * 6 + cta_rank_0 * 3;
                    int tile_end = num_tiles;
                    {
                        int global_mini_2 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                        int mini_tiles = mini_size / 128 * col_blocks_3;
                        first_tile = first_tile + global_mini_2 * mini_tiles;
                        int _min_54 = ((num_tiles) < ((global_mini_2 + 1) * mini_tiles) ? (num_tiles) : ((global_mini_2 + 1) * mini_tiles));
                        tile_end = _min_54;
                    }
                    if (first_tile < tile_end) {
                        int first_row_7 = first_tile / col_blocks_3;
                        int first_col = first_tile % col_blocks_3;
                        if (tid == 0) {
                            #pragma unroll
                            for (int stage = 0; stage < 3; stage++) {
                                if (tile_end > first_tile + stage) {
                                    int slot_3 = (replay_seq + stage) % 3;
                                    int row_11 = first_row_7;
                                    int col_7 = first_col + stage;
                                    if (col_7 >= col_blocks_3) {
                                        row_11 = row_11 + 1;
                                        col_7 = col_7 - col_blocks_3;
                                    }
                                    mbarrier_arrive_expect_tx(replay_arrived_addr + ((int)((unsigned int)(replay_seq + stage) % 3)) * 8, 65536);
                                    int parent = row_11 / 2 * (intermediate / 256) + col_7 / 2;
                                    int poll_7_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_first) : "l"(replay_gu + parent) : "memory");
                                    int poll_7 = poll_7_first;
                                    while (poll_7 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_7_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_next) : "l"(replay_gu + parent) : "memory");
                                        poll_7 = poll_7_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(replay_gate_addr + (unsigned int)(slot_3 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_11 - macro_row_offset) * 128), "r"(col_7 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + ((int)((unsigned int)(replay_seq + stage) % 3)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(replay_up_addr + (unsigned int)(slot_3 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_11 - macro_row_offset) * 128), "r"(col_7 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + ((int)((unsigned int)(replay_seq + stage) % 3)) * 8) : "memory");
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int stage_1 = 0; stage_1 < 3; stage_1++) {
                            if (tile_end > first_tile + stage_1) {
                                int slot_4 = (replay_seq + stage_1) % 3;
                                int _stage_replay_arrived = (int)((unsigned int)(replay_seq + stage_1) % 3);
                                int _parity_replay_arrived = (int)((unsigned int)(replay_seq + stage_1) / 3 % 2);
                                mbarrier_wait(replay_arrived_addr + (_stage_replay_arrived) * 8, _parity_replay_arrived);
                                int row_12 = first_row_7;
                                int col_8 = first_col + stage_1;
                                if (col_8 >= col_blocks_3) {
                                    row_12 = row_12 + 1;
                                    col_8 = col_8 - col_blocks_3;
                                }
                                unsigned int gate_words[32];
                                unsigned int up_words[32];
                                int warp_1 = tid / 32;
                                int local_warp = warp_1 / 4 + warp_1 % 4 * 2;
                                int lane_5 = tid % 32;
                                #pragma unroll
                                for (int tile_col = 0; tile_col < 4; tile_col++) {
                                    unsigned int packed_2[4];
                                    unsigned int address_8 = replay_gate_addr + (unsigned int)(slot_4 * 32768) + (unsigned int)(((tile_col * 16 + lane_5 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_5 % 16) * 64 + (tile_col * 16 + lane_5 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                        : "r"(address_8 ^ (address_8 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_7 = 0; pair_7 < 4; pair_7++) {
                                        gate_words[tile_col * 4 + pair_7] = packed_2[pair_7];
                                    }
                                }
                                int warp_0_4 = tid / 32;
                                int local_warp_0 = warp_0_4 / 4 + warp_0_4 % 4 * 2;
                                int lane_0 = tid % 32;
                                #pragma unroll
                                for (int tile_col_1 = 0; tile_col_1 < 4; tile_col_1++) {
                                    unsigned int packed_3[4];
                                    unsigned int address_9 = replay_up_addr + (unsigned int)(slot_4 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_0 / 16 * 8) / 64 * 128 * 64 + (local_warp_0 * 16 + lane_0 % 16) * 64 + (tile_col_1 * 16 + lane_0 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                        : "r"(address_9 ^ (address_9 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_8 = 0; pair_8 < 4; pair_8++) {
                                        up_words[tile_col_1 * 4 + pair_8] = packed_3[pair_8];
                                    }
                                }
                                float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[0]));
                                float2 v_0 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[0]));
                                unsigned int sign_0 = (gate_words[0] ^ up_words[0]) & 2147516416u;
                                float _abs_55;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(x_0.x));
                                float _abs_56;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_56) : "f"(x_0.y));
                                float2 ax = make_float2(_abs_55, _abs_56);
                                float _abs_57;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_57) : "f"(v_0.x));
                                float _abs_58;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(v_0.y));
                                float2 av = make_float2(_abs_57, _abs_58);
                                float2 _f32x2_mul_b_59 = make_float2(k1, k1);
                                float2 te;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
                                float2 _f32x2_mul_b_60 = make_float2(k2, k2);
                                float2 tw;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw) : "l"(*(const unsigned long long*)&av), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
                                float _exp2_61 = approx_exp2(te.x);
                                float _exp2_62 = approx_exp2(te.y);
                                float2 e = make_float2(_exp2_61, _exp2_62);
                                float _exp2_63 = approx_exp2(tw.x);
                                float _exp2_64 = approx_exp2(tw.y);
                                float2 w = make_float2(_exp2_63, _exp2_64);
                                float2 _f32x2_mul_b_65 = make_float2(-1.442695f, -1.442695f);
                                float2 tp;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
                                float _exp2_66 = approx_exp2_noftz(tp.x);
                                float _exp2_67 = approx_exp2_noftz(tp.y);
                                float2 p = make_float2(_exp2_66, _exp2_67);
                                float2 d1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                                float2 d2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p);
                                float2 dw = add_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                                float2 ng = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                                float2 nu = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                                float _rcp_68 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1, d2), dw).x);
                                float _rcp_69 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1, d2), dw).y);
                                float2 r_0 = make_float2(_rcp_68, _rcp_69);
                                float2 r = r_0;
                                float2 _f32x2_mul_a_70 = mul_f32x2_noftz(ng, nu);
                                float2 _f32x2_mul_b_71 = mul_f32x2_noftz(r, make_float2(bl, bl));
                                float2 m;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                                float2 ns = make_float2(((x_0.x < 0.0f) ? p.x : 1.0f), ((x_0.y < 0.0f) ? p.y : 1.0f));
                                gate_words[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m, ns))) | sign_0;
                                float2 x_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[1]));
                                float2 v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[1]));
                                unsigned int sign_1 = (gate_words[1] ^ up_words[1]) & 2147516416u;
                                float _abs_72;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_72) : "f"(x_1_1.x));
                                float _abs_73;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_73) : "f"(x_1_1.y));
                                float2 ax_0 = make_float2(_abs_72, _abs_73);
                                float _abs_74;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(v_1.x));
                                float _abs_75;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(v_1.y));
                                float2 av_0 = make_float2(_abs_74, _abs_75);
                                float2 _f32x2_mul_b_76 = make_float2(k1, k1);
                                float2 te_0;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_0) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
                                float2 _f32x2_mul_b_77 = make_float2(k2, k2);
                                float2 tw_0;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_0) : "l"(*(const unsigned long long*)&av_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
                                float _exp2_78 = approx_exp2(te_0.x);
                                float _exp2_79 = approx_exp2(te_0.y);
                                float2 e_0 = make_float2(_exp2_78, _exp2_79);
                                float _exp2_80 = approx_exp2(tw_0.x);
                                float _exp2_81 = approx_exp2(tw_0.y);
                                float2 w_0 = make_float2(_exp2_80, _exp2_81);
                                float2 _f32x2_mul_b_82 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_0;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_0) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
                                float _exp2_83 = approx_exp2_noftz(tp_0.x);
                                float _exp2_84 = approx_exp2_noftz(tp_0.y);
                                float2 p_0 = make_float2(_exp2_83, _exp2_84);
                                float2 d1_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                                float2 d2_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_0);
                                float2 dw_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                                float2 ng_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                                float2 nu_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                                float _rcp_85 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_0, d2_0), dw_0).x);
                                float _rcp_86 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_0, d2_0), dw_0).y);
                                float2 r_2 = make_float2(_rcp_85, _rcp_86);
                                float2 r_1 = r_2;
                                float2 _f32x2_mul_a_87 = mul_f32x2_noftz(ng_0, nu_0);
                                float2 _f32x2_mul_b_88 = mul_f32x2_noftz(r_1, make_float2(bl, bl));
                                float2 m_0;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
                                float2 ns_0 = make_float2(((x_1_1.x < 0.0f) ? p_0.x : 1.0f), ((x_1_1.y < 0.0f) ? p_0.y : 1.0f));
                                gate_words[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_0, ns_0))) | sign_1;
                                float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[2]));
                                float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[2]));
                                unsigned int sign_2 = (gate_words[2] ^ up_words[2]) & 2147516416u;
                                float _abs_89;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_89) : "f"(x_2.x));
                                float _abs_90;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(x_2.y));
                                float2 ax_1 = make_float2(_abs_89, _abs_90);
                                float _abs_91;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(v_2.x));
                                float _abs_92;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_92) : "f"(v_2.y));
                                float2 av_1 = make_float2(_abs_91, _abs_92);
                                float2 _f32x2_mul_b_93 = make_float2(k1, k1);
                                float2 te_1;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_1) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
                                float2 _f32x2_mul_b_94 = make_float2(k2, k2);
                                float2 tw_1;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_1) : "l"(*(const unsigned long long*)&av_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
                                float _exp2_95 = approx_exp2(te_1.x);
                                float _exp2_96 = approx_exp2(te_1.y);
                                float2 e_1 = make_float2(_exp2_95, _exp2_96);
                                float _exp2_97 = approx_exp2(tw_1.x);
                                float _exp2_98 = approx_exp2(tw_1.y);
                                float2 w_1 = make_float2(_exp2_97, _exp2_98);
                                float2 _f32x2_mul_b_99 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_1;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_1) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_99));
                                float _exp2_100 = approx_exp2_noftz(tp_1.x);
                                float _exp2_101 = approx_exp2_noftz(tp_1.y);
                                float2 p_1 = make_float2(_exp2_100, _exp2_101);
                                float2 d1_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                                float2 d2_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_1);
                                float2 dw_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                                float2 ng_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                                float2 nu_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                                float _rcp_102 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_1, d2_1), dw_1).x);
                                float _rcp_103 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_1, d2_1), dw_1).y);
                                float2 r_4 = make_float2(_rcp_102, _rcp_103);
                                float2 r_3 = r_4;
                                float2 _f32x2_mul_a_104 = mul_f32x2_noftz(ng_1, nu_1);
                                float2 _f32x2_mul_b_105 = mul_f32x2_noftz(r_3, make_float2(bl, bl));
                                float2 m_1;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
                                float2 ns_1 = make_float2(((x_2.x < 0.0f) ? p_1.x : 1.0f), ((x_2.y < 0.0f) ? p_1.y : 1.0f));
                                gate_words[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_1, ns_1))) | sign_2;
                                float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[3]));
                                float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[3]));
                                unsigned int sign_3 = (gate_words[3] ^ up_words[3]) & 2147516416u;
                                float _abs_106;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_106) : "f"(x_3.x));
                                float _abs_107;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_107) : "f"(x_3.y));
                                float2 ax_2 = make_float2(_abs_106, _abs_107);
                                float _abs_108;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_108) : "f"(v_3.x));
                                float _abs_109;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_109) : "f"(v_3.y));
                                float2 av_2 = make_float2(_abs_108, _abs_109);
                                float2 _f32x2_mul_b_110 = make_float2(k1, k1);
                                float2 te_2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_2) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
                                float2 _f32x2_mul_b_111 = make_float2(k2, k2);
                                float2 tw_2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_2) : "l"(*(const unsigned long long*)&av_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_111));
                                float _exp2_112 = approx_exp2(te_2.x);
                                float _exp2_113 = approx_exp2(te_2.y);
                                float2 e_2 = make_float2(_exp2_112, _exp2_113);
                                float _exp2_114 = approx_exp2(tw_2.x);
                                float _exp2_115 = approx_exp2(tw_2.y);
                                float2 w_2 = make_float2(_exp2_114, _exp2_115);
                                float2 _f32x2_mul_b_116 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_2) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
                                float _exp2_117 = approx_exp2_noftz(tp_2.x);
                                float _exp2_118 = approx_exp2_noftz(tp_2.y);
                                float2 p_2 = make_float2(_exp2_117, _exp2_118);
                                float2 d1_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                                float2 d2_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_2);
                                float2 dw_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                                float2 ng_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                                float2 nu_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                                float _rcp_119 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_2, d2_2), dw_2).x);
                                float _rcp_120 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_2, d2_2), dw_2).y);
                                float2 r_6 = make_float2(_rcp_119, _rcp_120);
                                float2 r_5 = r_6;
                                float2 _f32x2_mul_a_121 = mul_f32x2_noftz(ng_2, nu_2);
                                float2 _f32x2_mul_b_122 = mul_f32x2_noftz(r_5, make_float2(bl, bl));
                                float2 m_2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
                                float2 ns_2 = make_float2(((x_3.x < 0.0f) ? p_2.x : 1.0f), ((x_3.y < 0.0f) ? p_2.y : 1.0f));
                                gate_words[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_2, ns_2))) | sign_3;
                                float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[4]));
                                float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[4]));
                                unsigned int sign_4 = (gate_words[4] ^ up_words[4]) & 2147516416u;
                                float _abs_123;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_123) : "f"(x_4.x));
                                float _abs_124;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_124) : "f"(x_4.y));
                                float2 ax_3 = make_float2(_abs_123, _abs_124);
                                float _abs_125;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_125) : "f"(v_4.x));
                                float _abs_126;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(v_4.y));
                                float2 av_3 = make_float2(_abs_125, _abs_126);
                                float2 _f32x2_mul_b_127 = make_float2(k1, k1);
                                float2 te_3;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_127));
                                float2 _f32x2_mul_b_128 = make_float2(k2, k2);
                                float2 tw_3;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3) : "l"(*(const unsigned long long*)&av_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_128));
                                float _exp2_129 = approx_exp2(te_3.x);
                                float _exp2_130 = approx_exp2(te_3.y);
                                float2 e_3 = make_float2(_exp2_129, _exp2_130);
                                float _exp2_131 = approx_exp2(tw_3.x);
                                float _exp2_132 = approx_exp2(tw_3.y);
                                float2 w_3 = make_float2(_exp2_131, _exp2_132);
                                float2 _f32x2_mul_b_133 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_3;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_3) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
                                float _exp2_134 = approx_exp2_noftz(tp_3.x);
                                float _exp2_135 = approx_exp2_noftz(tp_3.y);
                                float2 p_3 = make_float2(_exp2_134, _exp2_135);
                                float2 d1_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                                float2 d2_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3);
                                float2 dw_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                                float2 ng_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                                float2 nu_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                                float _rcp_136 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_3, d2_3), dw_3).x);
                                float _rcp_137 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_3, d2_3), dw_3).y);
                                float2 r_8 = make_float2(_rcp_136, _rcp_137);
                                float2 r_7 = r_8;
                                float2 _f32x2_mul_a_138 = mul_f32x2_noftz(ng_3, nu_3);
                                float2 _f32x2_mul_b_139 = mul_f32x2_noftz(r_7, make_float2(bl, bl));
                                float2 m_3;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
                                float2 ns_3 = make_float2(((x_4.x < 0.0f) ? p_3.x : 1.0f), ((x_4.y < 0.0f) ? p_3.y : 1.0f));
                                gate_words[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_3, ns_3))) | sign_4;
                                float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[5]));
                                float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[5]));
                                unsigned int sign_5 = (gate_words[5] ^ up_words[5]) & 2147516416u;
                                float _abs_140;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_140) : "f"(x_5.x));
                                float _abs_141;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_141) : "f"(x_5.y));
                                float2 ax_4 = make_float2(_abs_140, _abs_141);
                                float _abs_142;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_142) : "f"(v_5.x));
                                float _abs_143;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_143) : "f"(v_5.y));
                                float2 av_4 = make_float2(_abs_142, _abs_143);
                                float2 _f32x2_mul_b_144 = make_float2(k1, k1);
                                float2 te_4;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_4) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_144));
                                float2 _f32x2_mul_b_145 = make_float2(k2, k2);
                                float2 tw_4;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_4) : "l"(*(const unsigned long long*)&av_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_145));
                                float _exp2_146 = approx_exp2(te_4.x);
                                float _exp2_147 = approx_exp2(te_4.y);
                                float2 e_4 = make_float2(_exp2_146, _exp2_147);
                                float _exp2_148 = approx_exp2(tw_4.x);
                                float _exp2_149 = approx_exp2(tw_4.y);
                                float2 w_4 = make_float2(_exp2_148, _exp2_149);
                                float2 _f32x2_mul_b_150 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_4;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_4) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_150));
                                float _exp2_151 = approx_exp2_noftz(tp_4.x);
                                float _exp2_152 = approx_exp2_noftz(tp_4.y);
                                float2 p_4 = make_float2(_exp2_151, _exp2_152);
                                float2 d1_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                                float2 d2_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_4);
                                float2 dw_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                                float2 ng_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                                float2 nu_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                                float _rcp_153 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_4, d2_4), dw_4).x);
                                float _rcp_154 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_4, d2_4), dw_4).y);
                                float2 r_10 = make_float2(_rcp_153, _rcp_154);
                                float2 r_9 = r_10;
                                float2 _f32x2_mul_a_155 = mul_f32x2_noftz(ng_4, nu_4);
                                float2 _f32x2_mul_b_156 = mul_f32x2_noftz(r_9, make_float2(bl, bl));
                                float2 m_4;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_155), "l"(*(const unsigned long long*)&_f32x2_mul_b_156));
                                float2 ns_4 = make_float2(((x_5.x < 0.0f) ? p_4.x : 1.0f), ((x_5.y < 0.0f) ? p_4.y : 1.0f));
                                gate_words[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_4, ns_4))) | sign_5;
                                float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[6]));
                                float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[6]));
                                unsigned int sign_6 = (gate_words[6] ^ up_words[6]) & 2147516416u;
                                float _abs_157;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_157) : "f"(x_6.x));
                                float _abs_158;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_158) : "f"(x_6.y));
                                float2 ax_5 = make_float2(_abs_157, _abs_158);
                                float _abs_159;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_159) : "f"(v_6.x));
                                float _abs_160;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_160) : "f"(v_6.y));
                                float2 av_5 = make_float2(_abs_159, _abs_160);
                                float2 _f32x2_mul_b_161 = make_float2(k1, k1);
                                float2 te_5;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_5) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
                                float2 _f32x2_mul_b_162 = make_float2(k2, k2);
                                float2 tw_5;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_5) : "l"(*(const unsigned long long*)&av_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_162));
                                float _exp2_163 = approx_exp2(te_5.x);
                                float _exp2_164 = approx_exp2(te_5.y);
                                float2 e_5 = make_float2(_exp2_163, _exp2_164);
                                float _exp2_165 = approx_exp2(tw_5.x);
                                float _exp2_166 = approx_exp2(tw_5.y);
                                float2 w_5 = make_float2(_exp2_165, _exp2_166);
                                float2 _f32x2_mul_b_167 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_5;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_5) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
                                float _exp2_168 = approx_exp2_noftz(tp_5.x);
                                float _exp2_169 = approx_exp2_noftz(tp_5.y);
                                float2 p_5 = make_float2(_exp2_168, _exp2_169);
                                float2 d1_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                                float2 d2_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_5);
                                float2 dw_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                                float2 ng_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                                float2 nu_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                                float _rcp_170 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_5, d2_5), dw_5).x);
                                float _rcp_171 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_5, d2_5), dw_5).y);
                                float2 r_12 = make_float2(_rcp_170, _rcp_171);
                                float2 r_11 = r_12;
                                float2 _f32x2_mul_a_172 = mul_f32x2_noftz(ng_5, nu_5);
                                float2 _f32x2_mul_b_173 = mul_f32x2_noftz(r_11, make_float2(bl, bl));
                                float2 m_5;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&_f32x2_mul_b_173));
                                float2 ns_5 = make_float2(((x_6.x < 0.0f) ? p_5.x : 1.0f), ((x_6.y < 0.0f) ? p_5.y : 1.0f));
                                gate_words[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_5, ns_5))) | sign_6;
                                float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[7]));
                                float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[7]));
                                unsigned int sign_7 = (gate_words[7] ^ up_words[7]) & 2147516416u;
                                float _abs_174;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_174) : "f"(x_7.x));
                                float _abs_175;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_175) : "f"(x_7.y));
                                float2 ax_6 = make_float2(_abs_174, _abs_175);
                                float _abs_176;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_176) : "f"(v_7.x));
                                float _abs_177;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_177) : "f"(v_7.y));
                                float2 av_6 = make_float2(_abs_176, _abs_177);
                                float2 _f32x2_mul_b_178 = make_float2(k1, k1);
                                float2 te_6;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_6) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_178));
                                float2 _f32x2_mul_b_179 = make_float2(k2, k2);
                                float2 tw_6;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_6) : "l"(*(const unsigned long long*)&av_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_179));
                                float _exp2_180 = approx_exp2(te_6.x);
                                float _exp2_181 = approx_exp2(te_6.y);
                                float2 e_6 = make_float2(_exp2_180, _exp2_181);
                                float _exp2_182 = approx_exp2(tw_6.x);
                                float _exp2_183 = approx_exp2(tw_6.y);
                                float2 w_6 = make_float2(_exp2_182, _exp2_183);
                                float2 _f32x2_mul_b_184 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_6;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_6) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_184));
                                float _exp2_185 = approx_exp2_noftz(tp_6.x);
                                float _exp2_186 = approx_exp2_noftz(tp_6.y);
                                float2 p_6 = make_float2(_exp2_185, _exp2_186);
                                float2 d1_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_6);
                                float2 d2_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_6);
                                float2 dw_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_6);
                                float2 ng_6 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_6);
                                float2 nu_6 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_6);
                                float _rcp_187 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_6, d2_6), dw_6).x);
                                float _rcp_188 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_6, d2_6), dw_6).y);
                                float2 r_14 = make_float2(_rcp_187, _rcp_188);
                                float2 r_13 = r_14;
                                float2 _f32x2_mul_a_189 = mul_f32x2_noftz(ng_6, nu_6);
                                float2 _f32x2_mul_b_190 = mul_f32x2_noftz(r_13, make_float2(bl, bl));
                                float2 m_6;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_189), "l"(*(const unsigned long long*)&_f32x2_mul_b_190));
                                float2 ns_6 = make_float2(((x_7.x < 0.0f) ? p_6.x : 1.0f), ((x_7.y < 0.0f) ? p_6.y : 1.0f));
                                gate_words[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_6, ns_6))) | sign_7;
                                float2 x_8 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[8]));
                                float2 v_8 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[8]));
                                unsigned int sign_8 = (gate_words[8] ^ up_words[8]) & 2147516416u;
                                float _abs_191;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(x_8.x));
                                float _abs_192;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_192) : "f"(x_8.y));
                                float2 ax_7 = make_float2(_abs_191, _abs_192);
                                float _abs_193;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_193) : "f"(v_8.x));
                                float _abs_194;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_194) : "f"(v_8.y));
                                float2 av_7 = make_float2(_abs_193, _abs_194);
                                float2 _f32x2_mul_b_195 = make_float2(k1, k1);
                                float2 te_7;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_7) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_195));
                                float2 _f32x2_mul_b_196 = make_float2(k2, k2);
                                float2 tw_7;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_7) : "l"(*(const unsigned long long*)&av_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_196));
                                float _exp2_197 = approx_exp2(te_7.x);
                                float _exp2_198 = approx_exp2(te_7.y);
                                float2 e_7 = make_float2(_exp2_197, _exp2_198);
                                float _exp2_199 = approx_exp2(tw_7.x);
                                float _exp2_200 = approx_exp2(tw_7.y);
                                float2 w_7 = make_float2(_exp2_199, _exp2_200);
                                float2 _f32x2_mul_b_201 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_7;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_7) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
                                float _exp2_202 = approx_exp2_noftz(tp_7.x);
                                float _exp2_203 = approx_exp2_noftz(tp_7.y);
                                float2 p_7 = make_float2(_exp2_202, _exp2_203);
                                float2 d1_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_7);
                                float2 d2_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_7);
                                float2 dw_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_7);
                                float2 ng_7 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_7);
                                float2 nu_7 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_7);
                                float _rcp_204 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_7, d2_7), dw_7).x);
                                float _rcp_205 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_7, d2_7), dw_7).y);
                                float2 r_16 = make_float2(_rcp_204, _rcp_205);
                                float2 r_15 = r_16;
                                float2 _f32x2_mul_a_206 = mul_f32x2_noftz(ng_7, nu_7);
                                float2 _f32x2_mul_b_207 = mul_f32x2_noftz(r_15, make_float2(bl, bl));
                                float2 m_7;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_206), "l"(*(const unsigned long long*)&_f32x2_mul_b_207));
                                float2 ns_7 = make_float2(((x_8.x < 0.0f) ? p_7.x : 1.0f), ((x_8.y < 0.0f) ? p_7.y : 1.0f));
                                gate_words[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_7, ns_7))) | sign_8;
                                float2 x_9 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[9]));
                                float2 v_9 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[9]));
                                unsigned int sign_9 = (gate_words[9] ^ up_words[9]) & 2147516416u;
                                float _abs_208;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_208) : "f"(x_9.x));
                                float _abs_209;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_209) : "f"(x_9.y));
                                float2 ax_8 = make_float2(_abs_208, _abs_209);
                                float _abs_210;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_210) : "f"(v_9.x));
                                float _abs_211;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_211) : "f"(v_9.y));
                                float2 av_8 = make_float2(_abs_210, _abs_211);
                                float2 _f32x2_mul_b_212 = make_float2(k1, k1);
                                float2 te_8;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_8) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_212));
                                float2 _f32x2_mul_b_213 = make_float2(k2, k2);
                                float2 tw_8;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_8) : "l"(*(const unsigned long long*)&av_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
                                float _exp2_214 = approx_exp2(te_8.x);
                                float _exp2_215 = approx_exp2(te_8.y);
                                float2 e_8 = make_float2(_exp2_214, _exp2_215);
                                float _exp2_216 = approx_exp2(tw_8.x);
                                float _exp2_217 = approx_exp2(tw_8.y);
                                float2 w_8 = make_float2(_exp2_216, _exp2_217);
                                float2 _f32x2_mul_b_218 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_8;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_8) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_218));
                                float _exp2_219 = approx_exp2_noftz(tp_8.x);
                                float _exp2_220 = approx_exp2_noftz(tp_8.y);
                                float2 p_8 = make_float2(_exp2_219, _exp2_220);
                                float2 d1_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_8);
                                float2 d2_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_8);
                                float2 dw_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_8);
                                float2 ng_8 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_8);
                                float2 nu_8 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_8);
                                float _rcp_221 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_8, d2_8), dw_8).x);
                                float _rcp_222 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_8, d2_8), dw_8).y);
                                float2 r_18 = make_float2(_rcp_221, _rcp_222);
                                float2 r_17 = r_18;
                                float2 _f32x2_mul_a_223 = mul_f32x2_noftz(ng_8, nu_8);
                                float2 _f32x2_mul_b_224 = mul_f32x2_noftz(r_17, make_float2(bl, bl));
                                float2 m_8;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&_f32x2_mul_b_224));
                                float2 ns_8 = make_float2(((x_9.x < 0.0f) ? p_8.x : 1.0f), ((x_9.y < 0.0f) ? p_8.y : 1.0f));
                                gate_words[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_8, ns_8))) | sign_9;
                                float2 x_10 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[10]));
                                float2 v_10 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[10]));
                                unsigned int sign_10 = (gate_words[10] ^ up_words[10]) & 2147516416u;
                                float _abs_225;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_225) : "f"(x_10.x));
                                float _abs_226;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_226) : "f"(x_10.y));
                                float2 ax_9 = make_float2(_abs_225, _abs_226);
                                float _abs_227;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_227) : "f"(v_10.x));
                                float _abs_228;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_228) : "f"(v_10.y));
                                float2 av_9 = make_float2(_abs_227, _abs_228);
                                float2 _f32x2_mul_b_229 = make_float2(k1, k1);
                                float2 te_9;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_9) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
                                float2 _f32x2_mul_b_230 = make_float2(k2, k2);
                                float2 tw_9;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_9) : "l"(*(const unsigned long long*)&av_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_230));
                                float _exp2_231 = approx_exp2(te_9.x);
                                float _exp2_232 = approx_exp2(te_9.y);
                                float2 e_9 = make_float2(_exp2_231, _exp2_232);
                                float _exp2_233 = approx_exp2(tw_9.x);
                                float _exp2_234 = approx_exp2(tw_9.y);
                                float2 w_9 = make_float2(_exp2_233, _exp2_234);
                                float2 _f32x2_mul_b_235 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_9;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_9) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_235));
                                float _exp2_236 = approx_exp2_noftz(tp_9.x);
                                float _exp2_237 = approx_exp2_noftz(tp_9.y);
                                float2 p_9 = make_float2(_exp2_236, _exp2_237);
                                float2 d1_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_9);
                                float2 d2_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_9);
                                float2 dw_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_9);
                                float2 ng_9 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_9);
                                float2 nu_9 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_9);
                                float _rcp_238 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_9, d2_9), dw_9).x);
                                float _rcp_239 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_9, d2_9), dw_9).y);
                                float2 r_20 = make_float2(_rcp_238, _rcp_239);
                                float2 r_19 = r_20;
                                float2 _f32x2_mul_a_240 = mul_f32x2_noftz(ng_9, nu_9);
                                float2 _f32x2_mul_b_241 = mul_f32x2_noftz(r_19, make_float2(bl, bl));
                                float2 m_9;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_240), "l"(*(const unsigned long long*)&_f32x2_mul_b_241));
                                float2 ns_9 = make_float2(((x_10.x < 0.0f) ? p_9.x : 1.0f), ((x_10.y < 0.0f) ? p_9.y : 1.0f));
                                gate_words[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_9, ns_9))) | sign_10;
                                float2 x_11 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[11]));
                                float2 v_11 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[11]));
                                unsigned int sign_11 = (gate_words[11] ^ up_words[11]) & 2147516416u;
                                float _abs_242;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_242) : "f"(x_11.x));
                                float _abs_243;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_243) : "f"(x_11.y));
                                float2 ax_10 = make_float2(_abs_242, _abs_243);
                                float _abs_244;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_244) : "f"(v_11.x));
                                float _abs_245;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_245) : "f"(v_11.y));
                                float2 av_10 = make_float2(_abs_244, _abs_245);
                                float2 _f32x2_mul_b_246 = make_float2(k1, k1);
                                float2 te_10;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_10) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_246));
                                float2 _f32x2_mul_b_247 = make_float2(k2, k2);
                                float2 tw_10;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_10) : "l"(*(const unsigned long long*)&av_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_247));
                                float _exp2_248 = approx_exp2(te_10.x);
                                float _exp2_249 = approx_exp2(te_10.y);
                                float2 e_10 = make_float2(_exp2_248, _exp2_249);
                                float _exp2_250 = approx_exp2(tw_10.x);
                                float _exp2_251 = approx_exp2(tw_10.y);
                                float2 w_10 = make_float2(_exp2_250, _exp2_251);
                                float2 _f32x2_mul_b_252 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_10;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_10) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_252));
                                float _exp2_253 = approx_exp2_noftz(tp_10.x);
                                float _exp2_254 = approx_exp2_noftz(tp_10.y);
                                float2 p_10 = make_float2(_exp2_253, _exp2_254);
                                float2 d1_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_10);
                                float2 d2_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_10);
                                float2 dw_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_10);
                                float2 ng_10 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_10);
                                float2 nu_10 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_10);
                                float _rcp_255 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_10, d2_10), dw_10).x);
                                float _rcp_256 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_10, d2_10), dw_10).y);
                                float2 r_22 = make_float2(_rcp_255, _rcp_256);
                                float2 r_21 = r_22;
                                float2 _f32x2_mul_a_257 = mul_f32x2_noftz(ng_10, nu_10);
                                float2 _f32x2_mul_b_258 = mul_f32x2_noftz(r_21, make_float2(bl, bl));
                                float2 m_10;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_257), "l"(*(const unsigned long long*)&_f32x2_mul_b_258));
                                float2 ns_10 = make_float2(((x_11.x < 0.0f) ? p_10.x : 1.0f), ((x_11.y < 0.0f) ? p_10.y : 1.0f));
                                gate_words[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_10, ns_10))) | sign_11;
                                float2 x_12 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[12]));
                                float2 v_12 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[12]));
                                unsigned int sign_12 = (gate_words[12] ^ up_words[12]) & 2147516416u;
                                float _abs_259;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_259) : "f"(x_12.x));
                                float _abs_260;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_260) : "f"(x_12.y));
                                float2 ax_11 = make_float2(_abs_259, _abs_260);
                                float _abs_261;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_261) : "f"(v_12.x));
                                float _abs_262;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_262) : "f"(v_12.y));
                                float2 av_11 = make_float2(_abs_261, _abs_262);
                                float2 _f32x2_mul_b_263 = make_float2(k1, k1);
                                float2 te_11;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_11) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_263));
                                float2 _f32x2_mul_b_264 = make_float2(k2, k2);
                                float2 tw_11;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_11) : "l"(*(const unsigned long long*)&av_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_264));
                                float _exp2_265 = approx_exp2(te_11.x);
                                float _exp2_266 = approx_exp2(te_11.y);
                                float2 e_11 = make_float2(_exp2_265, _exp2_266);
                                float _exp2_267 = approx_exp2(tw_11.x);
                                float _exp2_268 = approx_exp2(tw_11.y);
                                float2 w_11 = make_float2(_exp2_267, _exp2_268);
                                float2 _f32x2_mul_b_269 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_11;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_11) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_269));
                                float _exp2_270 = approx_exp2_noftz(tp_11.x);
                                float _exp2_271 = approx_exp2_noftz(tp_11.y);
                                float2 p_11 = make_float2(_exp2_270, _exp2_271);
                                float2 d1_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_11);
                                float2 d2_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_11);
                                float2 dw_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_11);
                                float2 ng_11 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_11);
                                float2 nu_11 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_11);
                                float _rcp_272 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_11, d2_11), dw_11).x);
                                float _rcp_273 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_11, d2_11), dw_11).y);
                                float2 r_24 = make_float2(_rcp_272, _rcp_273);
                                float2 r_23 = r_24;
                                float2 _f32x2_mul_a_274 = mul_f32x2_noftz(ng_11, nu_11);
                                float2 _f32x2_mul_b_275 = mul_f32x2_noftz(r_23, make_float2(bl, bl));
                                float2 m_11;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_274), "l"(*(const unsigned long long*)&_f32x2_mul_b_275));
                                float2 ns_11 = make_float2(((x_12.x < 0.0f) ? p_11.x : 1.0f), ((x_12.y < 0.0f) ? p_11.y : 1.0f));
                                gate_words[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_11, ns_11))) | sign_12;
                                float2 x_13 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[13]));
                                float2 v_13 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[13]));
                                unsigned int sign_13 = (gate_words[13] ^ up_words[13]) & 2147516416u;
                                float _abs_276;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_276) : "f"(x_13.x));
                                float _abs_277;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_277) : "f"(x_13.y));
                                float2 ax_12 = make_float2(_abs_276, _abs_277);
                                float _abs_278;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_278) : "f"(v_13.x));
                                float _abs_279;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_279) : "f"(v_13.y));
                                float2 av_12 = make_float2(_abs_278, _abs_279);
                                float2 _f32x2_mul_b_280 = make_float2(k1, k1);
                                float2 te_12;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_12) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_280));
                                float2 _f32x2_mul_b_281 = make_float2(k2, k2);
                                float2 tw_12;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_12) : "l"(*(const unsigned long long*)&av_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_281));
                                float _exp2_282 = approx_exp2(te_12.x);
                                float _exp2_283 = approx_exp2(te_12.y);
                                float2 e_12 = make_float2(_exp2_282, _exp2_283);
                                float _exp2_284 = approx_exp2(tw_12.x);
                                float _exp2_285 = approx_exp2(tw_12.y);
                                float2 w_12 = make_float2(_exp2_284, _exp2_285);
                                float2 _f32x2_mul_b_286 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_12;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_12) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_286));
                                float _exp2_287 = approx_exp2_noftz(tp_12.x);
                                float _exp2_288 = approx_exp2_noftz(tp_12.y);
                                float2 p_12 = make_float2(_exp2_287, _exp2_288);
                                float2 d1_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_12);
                                float2 d2_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_12);
                                float2 dw_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_12);
                                float2 ng_12 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_12);
                                float2 nu_12 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_12);
                                float _rcp_289 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_12, d2_12), dw_12).x);
                                float _rcp_290 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_12, d2_12), dw_12).y);
                                float2 r_26 = make_float2(_rcp_289, _rcp_290);
                                float2 r_25 = r_26;
                                float2 _f32x2_mul_a_291 = mul_f32x2_noftz(ng_12, nu_12);
                                float2 _f32x2_mul_b_292 = mul_f32x2_noftz(r_25, make_float2(bl, bl));
                                float2 m_12;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_291), "l"(*(const unsigned long long*)&_f32x2_mul_b_292));
                                float2 ns_12 = make_float2(((x_13.x < 0.0f) ? p_12.x : 1.0f), ((x_13.y < 0.0f) ? p_12.y : 1.0f));
                                gate_words[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_12, ns_12))) | sign_13;
                                float2 x_14 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[14]));
                                float2 v_14 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[14]));
                                unsigned int sign_14 = (gate_words[14] ^ up_words[14]) & 2147516416u;
                                float _abs_293;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_293) : "f"(x_14.x));
                                float _abs_294;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_294) : "f"(x_14.y));
                                float2 ax_13 = make_float2(_abs_293, _abs_294);
                                float _abs_295;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_295) : "f"(v_14.x));
                                float _abs_296;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_296) : "f"(v_14.y));
                                float2 av_13 = make_float2(_abs_295, _abs_296);
                                float2 _f32x2_mul_b_297 = make_float2(k1, k1);
                                float2 te_13;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_13) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_297));
                                float2 _f32x2_mul_b_298 = make_float2(k2, k2);
                                float2 tw_13;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_13) : "l"(*(const unsigned long long*)&av_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_298));
                                float _exp2_299 = approx_exp2(te_13.x);
                                float _exp2_300 = approx_exp2(te_13.y);
                                float2 e_13 = make_float2(_exp2_299, _exp2_300);
                                float _exp2_301 = approx_exp2(tw_13.x);
                                float _exp2_302 = approx_exp2(tw_13.y);
                                float2 w_13 = make_float2(_exp2_301, _exp2_302);
                                float2 _f32x2_mul_b_303 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_13;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_13) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_303));
                                float _exp2_304 = approx_exp2_noftz(tp_13.x);
                                float _exp2_305 = approx_exp2_noftz(tp_13.y);
                                float2 p_13 = make_float2(_exp2_304, _exp2_305);
                                float2 d1_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_13);
                                float2 d2_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_13);
                                float2 dw_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_13);
                                float2 ng_13 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_13);
                                float2 nu_13 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_13);
                                float _rcp_306 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_13, d2_13), dw_13).x);
                                float _rcp_307 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_13, d2_13), dw_13).y);
                                float2 r_28 = make_float2(_rcp_306, _rcp_307);
                                float2 r_27 = r_28;
                                float2 _f32x2_mul_a_308 = mul_f32x2_noftz(ng_13, nu_13);
                                float2 _f32x2_mul_b_309 = mul_f32x2_noftz(r_27, make_float2(bl, bl));
                                float2 m_13;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_308), "l"(*(const unsigned long long*)&_f32x2_mul_b_309));
                                float2 ns_13 = make_float2(((x_14.x < 0.0f) ? p_13.x : 1.0f), ((x_14.y < 0.0f) ? p_13.y : 1.0f));
                                gate_words[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_13, ns_13))) | sign_14;
                                float2 x_15 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[15]));
                                float2 v_15 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[15]));
                                unsigned int sign_15 = (gate_words[15] ^ up_words[15]) & 2147516416u;
                                float _abs_310;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_310) : "f"(x_15.x));
                                float _abs_311;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_311) : "f"(x_15.y));
                                float2 ax_14 = make_float2(_abs_310, _abs_311);
                                float _abs_312;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_312) : "f"(v_15.x));
                                float _abs_313;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_313) : "f"(v_15.y));
                                float2 av_14 = make_float2(_abs_312, _abs_313);
                                float2 _f32x2_mul_b_314 = make_float2(k1, k1);
                                float2 te_14;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_14) : "l"(*(const unsigned long long*)&ax_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_314));
                                float2 _f32x2_mul_b_315 = make_float2(k2, k2);
                                float2 tw_14;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_14) : "l"(*(const unsigned long long*)&av_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_315));
                                float _exp2_316 = approx_exp2(te_14.x);
                                float _exp2_317 = approx_exp2(te_14.y);
                                float2 e_14 = make_float2(_exp2_316, _exp2_317);
                                float _exp2_318 = approx_exp2(tw_14.x);
                                float _exp2_319 = approx_exp2(tw_14.y);
                                float2 w_14 = make_float2(_exp2_318, _exp2_319);
                                float2 _f32x2_mul_b_320 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_14;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_14) : "l"(*(const unsigned long long*)&ax_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_320));
                                float _exp2_321 = approx_exp2_noftz(tp_14.x);
                                float _exp2_322 = approx_exp2_noftz(tp_14.y);
                                float2 p_14 = make_float2(_exp2_321, _exp2_322);
                                float2 d1_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_14);
                                float2 d2_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_14);
                                float2 dw_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_14);
                                float2 ng_14 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_14);
                                float2 nu_14 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_14);
                                float _rcp_323 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_14, d2_14), dw_14).x);
                                float _rcp_324 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_14, d2_14), dw_14).y);
                                float2 r_30 = make_float2(_rcp_323, _rcp_324);
                                float2 r_29 = r_30;
                                float2 _f32x2_mul_a_325 = mul_f32x2_noftz(ng_14, nu_14);
                                float2 _f32x2_mul_b_326 = mul_f32x2_noftz(r_29, make_float2(bl, bl));
                                float2 m_14;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_325), "l"(*(const unsigned long long*)&_f32x2_mul_b_326));
                                float2 ns_14 = make_float2(((x_15.x < 0.0f) ? p_14.x : 1.0f), ((x_15.y < 0.0f) ? p_14.y : 1.0f));
                                gate_words[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_14, ns_14))) | sign_15;
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                __syncthreads();
                                int warp_1_1 = tid / 32;
                                int local_warp_1 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                                int lane_1_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_2 = 0; tile_col_2 < 4; tile_col_2++) {
                                    unsigned int packed_4[4];
                                    #pragma unroll
                                    for (int pair_9 = 0; pair_9 < 4; pair_9++) {
                                        packed_4[pair_9] = gate_words[tile_col_2 * 4 + pair_9];
                                    }
                                    unsigned int address_10 = replay_hidden_addr + (unsigned int)(((tile_col_2 * 16 + lane_1_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_1_1 % 16) * 64 + (tile_col_2 * 16 + lane_1_1 / 16 * 8) % 64) * 2);
                                    uint32_t _stmatrix_addr_327 = static_cast<uint32_t>(address_10 ^ (address_10 & 1023) >> 7 << 4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_327), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                        : "memory");
                                }
                                int warp_2 = tid / 32;
                                int local_warp_2 = warp_2 / 4 + warp_2 % 4 * 2;
                                int lane_2_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_3 = 4; tile_col_3 < 8; tile_col_3++) {
                                    unsigned int packed_5[4];
                                    unsigned int address_11 = replay_gate_addr + (unsigned int)(slot_4 * 32768) + (unsigned int)(((tile_col_3 * 16 + lane_2_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_2_1 % 16) * 64 + (tile_col_3 * 16 + lane_2_1 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_5[0]), "=r"(packed_5[1]), "=r"(packed_5[2]), "=r"(packed_5[3])
                                        : "r"(address_11 ^ (address_11 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_10 = 0; pair_10 < 4; pair_10++) {
                                        gate_words[tile_col_3 * 4 + pair_10] = packed_5[pair_10];
                                    }
                                }
                                int warp_3 = tid / 32;
                                int local_warp_3 = warp_3 / 4 + warp_3 % 4 * 2;
                                int lane_3_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_4 = 4; tile_col_4 < 8; tile_col_4++) {
                                    unsigned int packed_6[4];
                                    unsigned int address_12 = replay_up_addr + (unsigned int)(slot_4 * 32768) + (unsigned int)(((tile_col_4 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_3 * 16 + lane_3_1 % 16) * 64 + (tile_col_4 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_6[0]), "=r"(packed_6[1]), "=r"(packed_6[2]), "=r"(packed_6[3])
                                        : "r"(address_12 ^ (address_12 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_11 = 0; pair_11 < 4; pair_11++) {
                                        up_words[tile_col_4 * 4 + pair_11] = packed_6[pair_11];
                                    }
                                }
                                float2 x_16 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[16]));
                                float2 v_16 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[16]));
                                unsigned int sign_16 = (gate_words[16] ^ up_words[16]) & 2147516416u;
                                float _abs_328;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_328) : "f"(x_16.x));
                                float _abs_329;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_329) : "f"(x_16.y));
                                float2 ax_15 = make_float2(_abs_328, _abs_329);
                                float _abs_330;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_330) : "f"(v_16.x));
                                float _abs_331;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_331) : "f"(v_16.y));
                                float2 av_15 = make_float2(_abs_330, _abs_331);
                                float2 _f32x2_mul_b_332 = make_float2(k1, k1);
                                float2 te_15;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_15) : "l"(*(const unsigned long long*)&ax_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_332));
                                float2 _f32x2_mul_b_333 = make_float2(k2, k2);
                                float2 tw_15;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_15) : "l"(*(const unsigned long long*)&av_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_333));
                                float _exp2_334 = approx_exp2(te_15.x);
                                float _exp2_335 = approx_exp2(te_15.y);
                                float2 e_15 = make_float2(_exp2_334, _exp2_335);
                                float _exp2_336 = approx_exp2(tw_15.x);
                                float _exp2_337 = approx_exp2(tw_15.y);
                                float2 w_15 = make_float2(_exp2_336, _exp2_337);
                                float2 _f32x2_mul_b_338 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_15;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_15) : "l"(*(const unsigned long long*)&ax_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_338));
                                float _exp2_339 = approx_exp2_noftz(tp_15.x);
                                float _exp2_340 = approx_exp2_noftz(tp_15.y);
                                float2 p_15 = make_float2(_exp2_339, _exp2_340);
                                float2 d1_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_15);
                                float2 d2_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_15);
                                float2 dw_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_15);
                                float2 ng_15 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_15);
                                float2 nu_15 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_15);
                                float _rcp_341 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_15, d2_15), dw_15).x);
                                float _rcp_342 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_15, d2_15), dw_15).y);
                                float2 r_32 = make_float2(_rcp_341, _rcp_342);
                                float2 r_31 = r_32;
                                float2 _f32x2_mul_a_343 = mul_f32x2_noftz(ng_15, nu_15);
                                float2 _f32x2_mul_b_344 = mul_f32x2_noftz(r_31, make_float2(bl, bl));
                                float2 m_15;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_343), "l"(*(const unsigned long long*)&_f32x2_mul_b_344));
                                float2 ns_15 = make_float2(((x_16.x < 0.0f) ? p_15.x : 1.0f), ((x_16.y < 0.0f) ? p_15.y : 1.0f));
                                gate_words[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_15, ns_15))) | sign_16;
                                float2 x_17 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[17]));
                                float2 v_17 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[17]));
                                unsigned int sign_17 = (gate_words[17] ^ up_words[17]) & 2147516416u;
                                float _abs_345;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_345) : "f"(x_17.x));
                                float _abs_346;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_346) : "f"(x_17.y));
                                float2 ax_16 = make_float2(_abs_345, _abs_346);
                                float _abs_347;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_347) : "f"(v_17.x));
                                float _abs_348;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_348) : "f"(v_17.y));
                                float2 av_16 = make_float2(_abs_347, _abs_348);
                                float2 _f32x2_mul_b_349 = make_float2(k1, k1);
                                float2 te_16;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_16) : "l"(*(const unsigned long long*)&ax_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_349));
                                float2 _f32x2_mul_b_350 = make_float2(k2, k2);
                                float2 tw_16;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_16) : "l"(*(const unsigned long long*)&av_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_350));
                                float _exp2_351 = approx_exp2(te_16.x);
                                float _exp2_352 = approx_exp2(te_16.y);
                                float2 e_16 = make_float2(_exp2_351, _exp2_352);
                                float _exp2_353 = approx_exp2(tw_16.x);
                                float _exp2_354 = approx_exp2(tw_16.y);
                                float2 w_16 = make_float2(_exp2_353, _exp2_354);
                                float2 _f32x2_mul_b_355 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_16;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_16) : "l"(*(const unsigned long long*)&ax_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_355));
                                float _exp2_356 = approx_exp2_noftz(tp_16.x);
                                float _exp2_357 = approx_exp2_noftz(tp_16.y);
                                float2 p_16 = make_float2(_exp2_356, _exp2_357);
                                float2 d1_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_16);
                                float2 d2_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_16);
                                float2 dw_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_16);
                                float2 ng_16 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_16);
                                float2 nu_16 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_16);
                                float _rcp_358 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_16, d2_16), dw_16).x);
                                float _rcp_359 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_16, d2_16), dw_16).y);
                                float2 r_34 = make_float2(_rcp_358, _rcp_359);
                                float2 r_33 = r_34;
                                float2 _f32x2_mul_a_360 = mul_f32x2_noftz(ng_16, nu_16);
                                float2 _f32x2_mul_b_361 = mul_f32x2_noftz(r_33, make_float2(bl, bl));
                                float2 m_16;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_360), "l"(*(const unsigned long long*)&_f32x2_mul_b_361));
                                float2 ns_16 = make_float2(((x_17.x < 0.0f) ? p_16.x : 1.0f), ((x_17.y < 0.0f) ? p_16.y : 1.0f));
                                gate_words[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_16, ns_16))) | sign_17;
                                float2 x_18 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[18]));
                                float2 v_18 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[18]));
                                unsigned int sign_18 = (gate_words[18] ^ up_words[18]) & 2147516416u;
                                float _abs_362;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_362) : "f"(x_18.x));
                                float _abs_363;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_363) : "f"(x_18.y));
                                float2 ax_17 = make_float2(_abs_362, _abs_363);
                                float _abs_364;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_364) : "f"(v_18.x));
                                float _abs_365;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_365) : "f"(v_18.y));
                                float2 av_17 = make_float2(_abs_364, _abs_365);
                                float2 _f32x2_mul_b_366 = make_float2(k1, k1);
                                float2 te_17;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_17) : "l"(*(const unsigned long long*)&ax_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_366));
                                float2 _f32x2_mul_b_367 = make_float2(k2, k2);
                                float2 tw_17;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_17) : "l"(*(const unsigned long long*)&av_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_367));
                                float _exp2_368 = approx_exp2(te_17.x);
                                float _exp2_369 = approx_exp2(te_17.y);
                                float2 e_17 = make_float2(_exp2_368, _exp2_369);
                                float _exp2_370 = approx_exp2(tw_17.x);
                                float _exp2_371 = approx_exp2(tw_17.y);
                                float2 w_17 = make_float2(_exp2_370, _exp2_371);
                                float2 _f32x2_mul_b_372 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_17;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_17) : "l"(*(const unsigned long long*)&ax_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_372));
                                float _exp2_373 = approx_exp2_noftz(tp_17.x);
                                float _exp2_374 = approx_exp2_noftz(tp_17.y);
                                float2 p_17 = make_float2(_exp2_373, _exp2_374);
                                float2 d1_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_17);
                                float2 d2_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_17);
                                float2 dw_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_17);
                                float2 ng_17 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_17);
                                float2 nu_17 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_17);
                                float _rcp_375 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_17, d2_17), dw_17).x);
                                float _rcp_376 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_17, d2_17), dw_17).y);
                                float2 r_36 = make_float2(_rcp_375, _rcp_376);
                                float2 r_35 = r_36;
                                float2 _f32x2_mul_a_377 = mul_f32x2_noftz(ng_17, nu_17);
                                float2 _f32x2_mul_b_378 = mul_f32x2_noftz(r_35, make_float2(bl, bl));
                                float2 m_17;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_377), "l"(*(const unsigned long long*)&_f32x2_mul_b_378));
                                float2 ns_17 = make_float2(((x_18.x < 0.0f) ? p_17.x : 1.0f), ((x_18.y < 0.0f) ? p_17.y : 1.0f));
                                gate_words[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_17, ns_17))) | sign_18;
                                float2 x_19 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[19]));
                                float2 v_19 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[19]));
                                unsigned int sign_19 = (gate_words[19] ^ up_words[19]) & 2147516416u;
                                float _abs_379;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_379) : "f"(x_19.x));
                                float _abs_380;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_380) : "f"(x_19.y));
                                float2 ax_18 = make_float2(_abs_379, _abs_380);
                                float _abs_381;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_381) : "f"(v_19.x));
                                float _abs_382;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_382) : "f"(v_19.y));
                                float2 av_18 = make_float2(_abs_381, _abs_382);
                                float2 _f32x2_mul_b_383 = make_float2(k1, k1);
                                float2 te_18;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_18) : "l"(*(const unsigned long long*)&ax_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_383));
                                float2 _f32x2_mul_b_384 = make_float2(k2, k2);
                                float2 tw_18;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_18) : "l"(*(const unsigned long long*)&av_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_384));
                                float _exp2_385 = approx_exp2(te_18.x);
                                float _exp2_386 = approx_exp2(te_18.y);
                                float2 e_18 = make_float2(_exp2_385, _exp2_386);
                                float _exp2_387 = approx_exp2(tw_18.x);
                                float _exp2_388 = approx_exp2(tw_18.y);
                                float2 w_18 = make_float2(_exp2_387, _exp2_388);
                                float2 _f32x2_mul_b_389 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_18;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_18) : "l"(*(const unsigned long long*)&ax_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_389));
                                float _exp2_390 = approx_exp2_noftz(tp_18.x);
                                float _exp2_391 = approx_exp2_noftz(tp_18.y);
                                float2 p_18 = make_float2(_exp2_390, _exp2_391);
                                float2 d1_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_18);
                                float2 d2_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_18);
                                float2 dw_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_18);
                                float2 ng_18 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_18);
                                float2 nu_18 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_18);
                                float _rcp_392 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_18, d2_18), dw_18).x);
                                float _rcp_393 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_18, d2_18), dw_18).y);
                                float2 r_38 = make_float2(_rcp_392, _rcp_393);
                                float2 r_37 = r_38;
                                float2 _f32x2_mul_a_394 = mul_f32x2_noftz(ng_18, nu_18);
                                float2 _f32x2_mul_b_395 = mul_f32x2_noftz(r_37, make_float2(bl, bl));
                                float2 m_18;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_394), "l"(*(const unsigned long long*)&_f32x2_mul_b_395));
                                float2 ns_18 = make_float2(((x_19.x < 0.0f) ? p_18.x : 1.0f), ((x_19.y < 0.0f) ? p_18.y : 1.0f));
                                gate_words[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_18, ns_18))) | sign_19;
                                float2 x_20 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[20]));
                                float2 v_20 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[20]));
                                unsigned int sign_20 = (gate_words[20] ^ up_words[20]) & 2147516416u;
                                float _abs_396;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_396) : "f"(x_20.x));
                                float _abs_397;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_397) : "f"(x_20.y));
                                float2 ax_19 = make_float2(_abs_396, _abs_397);
                                float _abs_398;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_398) : "f"(v_20.x));
                                float _abs_399;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_399) : "f"(v_20.y));
                                float2 av_19 = make_float2(_abs_398, _abs_399);
                                float2 _f32x2_mul_b_400 = make_float2(k1, k1);
                                float2 te_19;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_19) : "l"(*(const unsigned long long*)&ax_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_400));
                                float2 _f32x2_mul_b_401 = make_float2(k2, k2);
                                float2 tw_19;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_19) : "l"(*(const unsigned long long*)&av_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_401));
                                float _exp2_402 = approx_exp2(te_19.x);
                                float _exp2_403 = approx_exp2(te_19.y);
                                float2 e_19 = make_float2(_exp2_402, _exp2_403);
                                float _exp2_404 = approx_exp2(tw_19.x);
                                float _exp2_405 = approx_exp2(tw_19.y);
                                float2 w_19 = make_float2(_exp2_404, _exp2_405);
                                float2 _f32x2_mul_b_406 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_19;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_19) : "l"(*(const unsigned long long*)&ax_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_406));
                                float _exp2_407 = approx_exp2_noftz(tp_19.x);
                                float _exp2_408 = approx_exp2_noftz(tp_19.y);
                                float2 p_19 = make_float2(_exp2_407, _exp2_408);
                                float2 d1_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_19);
                                float2 d2_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_19);
                                float2 dw_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_19);
                                float2 ng_19 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_19);
                                float2 nu_19 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_19);
                                float _rcp_409 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_19, d2_19), dw_19).x);
                                float _rcp_410 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_19, d2_19), dw_19).y);
                                float2 r_40 = make_float2(_rcp_409, _rcp_410);
                                float2 r_39 = r_40;
                                float2 _f32x2_mul_a_411 = mul_f32x2_noftz(ng_19, nu_19);
                                float2 _f32x2_mul_b_412 = mul_f32x2_noftz(r_39, make_float2(bl, bl));
                                float2 m_19;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_411), "l"(*(const unsigned long long*)&_f32x2_mul_b_412));
                                float2 ns_19 = make_float2(((x_20.x < 0.0f) ? p_19.x : 1.0f), ((x_20.y < 0.0f) ? p_19.y : 1.0f));
                                gate_words[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_19, ns_19))) | sign_20;
                                float2 x_21 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[21]));
                                float2 v_21 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[21]));
                                unsigned int sign_21 = (gate_words[21] ^ up_words[21]) & 2147516416u;
                                float _abs_413;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_413) : "f"(x_21.x));
                                float _abs_414;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_414) : "f"(x_21.y));
                                float2 ax_20 = make_float2(_abs_413, _abs_414);
                                float _abs_415;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_415) : "f"(v_21.x));
                                float _abs_416;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_416) : "f"(v_21.y));
                                float2 av_20 = make_float2(_abs_415, _abs_416);
                                float2 _f32x2_mul_b_417 = make_float2(k1, k1);
                                float2 te_20;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_20) : "l"(*(const unsigned long long*)&ax_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_417));
                                float2 _f32x2_mul_b_418 = make_float2(k2, k2);
                                float2 tw_20;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_20) : "l"(*(const unsigned long long*)&av_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_418));
                                float _exp2_419 = approx_exp2(te_20.x);
                                float _exp2_420 = approx_exp2(te_20.y);
                                float2 e_20 = make_float2(_exp2_419, _exp2_420);
                                float _exp2_421 = approx_exp2(tw_20.x);
                                float _exp2_422 = approx_exp2(tw_20.y);
                                float2 w_20 = make_float2(_exp2_421, _exp2_422);
                                float2 _f32x2_mul_b_423 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_20;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_20) : "l"(*(const unsigned long long*)&ax_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_423));
                                float _exp2_424 = approx_exp2_noftz(tp_20.x);
                                float _exp2_425 = approx_exp2_noftz(tp_20.y);
                                float2 p_20 = make_float2(_exp2_424, _exp2_425);
                                float2 d1_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_20);
                                float2 d2_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_20);
                                float2 dw_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_20);
                                float2 ng_20 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_20);
                                float2 nu_20 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_20);
                                float _rcp_426 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_20, d2_20), dw_20).x);
                                float _rcp_427 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_20, d2_20), dw_20).y);
                                float2 r_42 = make_float2(_rcp_426, _rcp_427);
                                float2 r_41 = r_42;
                                float2 _f32x2_mul_a_428 = mul_f32x2_noftz(ng_20, nu_20);
                                float2 _f32x2_mul_b_429 = mul_f32x2_noftz(r_41, make_float2(bl, bl));
                                float2 m_20;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_428), "l"(*(const unsigned long long*)&_f32x2_mul_b_429));
                                float2 ns_20 = make_float2(((x_21.x < 0.0f) ? p_20.x : 1.0f), ((x_21.y < 0.0f) ? p_20.y : 1.0f));
                                gate_words[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_20, ns_20))) | sign_21;
                                float2 x_22 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[22]));
                                float2 v_22 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[22]));
                                unsigned int sign_22 = (gate_words[22] ^ up_words[22]) & 2147516416u;
                                float _abs_430;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_430) : "f"(x_22.x));
                                float _abs_431;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_431) : "f"(x_22.y));
                                float2 ax_21 = make_float2(_abs_430, _abs_431);
                                float _abs_432;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_432) : "f"(v_22.x));
                                float _abs_433;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_433) : "f"(v_22.y));
                                float2 av_21 = make_float2(_abs_432, _abs_433);
                                float2 _f32x2_mul_b_434 = make_float2(k1, k1);
                                float2 te_21;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_21) : "l"(*(const unsigned long long*)&ax_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_434));
                                float2 _f32x2_mul_b_435 = make_float2(k2, k2);
                                float2 tw_21;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_21) : "l"(*(const unsigned long long*)&av_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_435));
                                float _exp2_436 = approx_exp2(te_21.x);
                                float _exp2_437 = approx_exp2(te_21.y);
                                float2 e_21 = make_float2(_exp2_436, _exp2_437);
                                float _exp2_438 = approx_exp2(tw_21.x);
                                float _exp2_439 = approx_exp2(tw_21.y);
                                float2 w_21 = make_float2(_exp2_438, _exp2_439);
                                float2 _f32x2_mul_b_440 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_21;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_21) : "l"(*(const unsigned long long*)&ax_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_440));
                                float _exp2_441 = approx_exp2_noftz(tp_21.x);
                                float _exp2_442 = approx_exp2_noftz(tp_21.y);
                                float2 p_21 = make_float2(_exp2_441, _exp2_442);
                                float2 d1_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_21);
                                float2 d2_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_21);
                                float2 dw_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_21);
                                float2 ng_21 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_21);
                                float2 nu_21 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_21);
                                float _rcp_443 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_21, d2_21), dw_21).x);
                                float _rcp_444 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_21, d2_21), dw_21).y);
                                float2 r_44 = make_float2(_rcp_443, _rcp_444);
                                float2 r_43 = r_44;
                                float2 _f32x2_mul_a_445 = mul_f32x2_noftz(ng_21, nu_21);
                                float2 _f32x2_mul_b_446 = mul_f32x2_noftz(r_43, make_float2(bl, bl));
                                float2 m_21;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_445), "l"(*(const unsigned long long*)&_f32x2_mul_b_446));
                                float2 ns_21 = make_float2(((x_22.x < 0.0f) ? p_21.x : 1.0f), ((x_22.y < 0.0f) ? p_21.y : 1.0f));
                                gate_words[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_21, ns_21))) | sign_22;
                                float2 x_23 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[23]));
                                float2 v_23 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[23]));
                                unsigned int sign_23 = (gate_words[23] ^ up_words[23]) & 2147516416u;
                                float _abs_447;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_447) : "f"(x_23.x));
                                float _abs_448;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_448) : "f"(x_23.y));
                                float2 ax_22 = make_float2(_abs_447, _abs_448);
                                float _abs_449;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_449) : "f"(v_23.x));
                                float _abs_450;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_450) : "f"(v_23.y));
                                float2 av_22 = make_float2(_abs_449, _abs_450);
                                float2 _f32x2_mul_b_451 = make_float2(k1, k1);
                                float2 te_22;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_22) : "l"(*(const unsigned long long*)&ax_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_451));
                                float2 _f32x2_mul_b_452 = make_float2(k2, k2);
                                float2 tw_22;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_22) : "l"(*(const unsigned long long*)&av_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_452));
                                float _exp2_453 = approx_exp2(te_22.x);
                                float _exp2_454 = approx_exp2(te_22.y);
                                float2 e_22 = make_float2(_exp2_453, _exp2_454);
                                float _exp2_455 = approx_exp2(tw_22.x);
                                float _exp2_456 = approx_exp2(tw_22.y);
                                float2 w_22 = make_float2(_exp2_455, _exp2_456);
                                float2 _f32x2_mul_b_457 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_22;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_22) : "l"(*(const unsigned long long*)&ax_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_457));
                                float _exp2_458 = approx_exp2_noftz(tp_22.x);
                                float _exp2_459 = approx_exp2_noftz(tp_22.y);
                                float2 p_22 = make_float2(_exp2_458, _exp2_459);
                                float2 d1_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_22);
                                float2 d2_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_22);
                                float2 dw_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_22);
                                float2 ng_22 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_22);
                                float2 nu_22 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_22);
                                float _rcp_460 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_22, d2_22), dw_22).x);
                                float _rcp_461 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_22, d2_22), dw_22).y);
                                float2 r_46 = make_float2(_rcp_460, _rcp_461);
                                float2 r_45 = r_46;
                                float2 _f32x2_mul_a_462 = mul_f32x2_noftz(ng_22, nu_22);
                                float2 _f32x2_mul_b_463 = mul_f32x2_noftz(r_45, make_float2(bl, bl));
                                float2 m_22;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_462), "l"(*(const unsigned long long*)&_f32x2_mul_b_463));
                                float2 ns_22 = make_float2(((x_23.x < 0.0f) ? p_22.x : 1.0f), ((x_23.y < 0.0f) ? p_22.y : 1.0f));
                                gate_words[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_22, ns_22))) | sign_23;
                                float2 x_24 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[24]));
                                float2 v_24 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[24]));
                                unsigned int sign_24 = (gate_words[24] ^ up_words[24]) & 2147516416u;
                                float _abs_464;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_464) : "f"(x_24.x));
                                float _abs_465;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_465) : "f"(x_24.y));
                                float2 ax_23 = make_float2(_abs_464, _abs_465);
                                float _abs_466;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_466) : "f"(v_24.x));
                                float _abs_467;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_467) : "f"(v_24.y));
                                float2 av_23 = make_float2(_abs_466, _abs_467);
                                float2 _f32x2_mul_b_468 = make_float2(k1, k1);
                                float2 te_23;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_23) : "l"(*(const unsigned long long*)&ax_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_468));
                                float2 _f32x2_mul_b_469 = make_float2(k2, k2);
                                float2 tw_23;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_23) : "l"(*(const unsigned long long*)&av_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_469));
                                float _exp2_470 = approx_exp2(te_23.x);
                                float _exp2_471 = approx_exp2(te_23.y);
                                float2 e_23 = make_float2(_exp2_470, _exp2_471);
                                float _exp2_472 = approx_exp2(tw_23.x);
                                float _exp2_473 = approx_exp2(tw_23.y);
                                float2 w_23 = make_float2(_exp2_472, _exp2_473);
                                float2 _f32x2_mul_b_474 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_23;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_23) : "l"(*(const unsigned long long*)&ax_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_474));
                                float _exp2_475 = approx_exp2_noftz(tp_23.x);
                                float _exp2_476 = approx_exp2_noftz(tp_23.y);
                                float2 p_23 = make_float2(_exp2_475, _exp2_476);
                                float2 d1_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_23);
                                float2 d2_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_23);
                                float2 dw_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_23);
                                float2 ng_23 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_23);
                                float2 nu_23 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_23);
                                float _rcp_477 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_23, d2_23), dw_23).x);
                                float _rcp_478 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_23, d2_23), dw_23).y);
                                float2 r_48 = make_float2(_rcp_477, _rcp_478);
                                float2 r_47 = r_48;
                                float2 _f32x2_mul_a_479 = mul_f32x2_noftz(ng_23, nu_23);
                                float2 _f32x2_mul_b_480 = mul_f32x2_noftz(r_47, make_float2(bl, bl));
                                float2 m_23;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_479), "l"(*(const unsigned long long*)&_f32x2_mul_b_480));
                                float2 ns_23 = make_float2(((x_24.x < 0.0f) ? p_23.x : 1.0f), ((x_24.y < 0.0f) ? p_23.y : 1.0f));
                                gate_words[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_23, ns_23))) | sign_24;
                                float2 x_25 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[25]));
                                float2 v_25 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[25]));
                                unsigned int sign_25 = (gate_words[25] ^ up_words[25]) & 2147516416u;
                                float _abs_481;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_481) : "f"(x_25.x));
                                float _abs_482;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_482) : "f"(x_25.y));
                                float2 ax_24 = make_float2(_abs_481, _abs_482);
                                float _abs_483;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_483) : "f"(v_25.x));
                                float _abs_484;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_484) : "f"(v_25.y));
                                float2 av_24 = make_float2(_abs_483, _abs_484);
                                float2 _f32x2_mul_b_485 = make_float2(k1, k1);
                                float2 te_24;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_24) : "l"(*(const unsigned long long*)&ax_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_485));
                                float2 _f32x2_mul_b_486 = make_float2(k2, k2);
                                float2 tw_24;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_24) : "l"(*(const unsigned long long*)&av_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_486));
                                float _exp2_487 = approx_exp2(te_24.x);
                                float _exp2_488 = approx_exp2(te_24.y);
                                float2 e_24 = make_float2(_exp2_487, _exp2_488);
                                float _exp2_489 = approx_exp2(tw_24.x);
                                float _exp2_490 = approx_exp2(tw_24.y);
                                float2 w_24 = make_float2(_exp2_489, _exp2_490);
                                float2 _f32x2_mul_b_491 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_24;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_24) : "l"(*(const unsigned long long*)&ax_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_491));
                                float _exp2_492 = approx_exp2_noftz(tp_24.x);
                                float _exp2_493 = approx_exp2_noftz(tp_24.y);
                                float2 p_24 = make_float2(_exp2_492, _exp2_493);
                                float2 d1_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_24);
                                float2 d2_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_24);
                                float2 dw_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_24);
                                float2 ng_24 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_24);
                                float2 nu_24 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_24);
                                float _rcp_494 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_24, d2_24), dw_24).x);
                                float _rcp_495 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_24, d2_24), dw_24).y);
                                float2 r_50 = make_float2(_rcp_494, _rcp_495);
                                float2 r_49 = r_50;
                                float2 _f32x2_mul_a_496 = mul_f32x2_noftz(ng_24, nu_24);
                                float2 _f32x2_mul_b_497 = mul_f32x2_noftz(r_49, make_float2(bl, bl));
                                float2 m_24;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_496), "l"(*(const unsigned long long*)&_f32x2_mul_b_497));
                                float2 ns_24 = make_float2(((x_25.x < 0.0f) ? p_24.x : 1.0f), ((x_25.y < 0.0f) ? p_24.y : 1.0f));
                                gate_words[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_24, ns_24))) | sign_25;
                                float2 x_26 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[26]));
                                float2 v_26 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[26]));
                                unsigned int sign_26 = (gate_words[26] ^ up_words[26]) & 2147516416u;
                                float _abs_498;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_498) : "f"(x_26.x));
                                float _abs_499;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_499) : "f"(x_26.y));
                                float2 ax_25 = make_float2(_abs_498, _abs_499);
                                float _abs_500;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_500) : "f"(v_26.x));
                                float _abs_501;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_501) : "f"(v_26.y));
                                float2 av_25 = make_float2(_abs_500, _abs_501);
                                float2 _f32x2_mul_b_502 = make_float2(k1, k1);
                                float2 te_25;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_25) : "l"(*(const unsigned long long*)&ax_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_502));
                                float2 _f32x2_mul_b_503 = make_float2(k2, k2);
                                float2 tw_25;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_25) : "l"(*(const unsigned long long*)&av_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_503));
                                float _exp2_504 = approx_exp2(te_25.x);
                                float _exp2_505 = approx_exp2(te_25.y);
                                float2 e_25 = make_float2(_exp2_504, _exp2_505);
                                float _exp2_506 = approx_exp2(tw_25.x);
                                float _exp2_507 = approx_exp2(tw_25.y);
                                float2 w_25 = make_float2(_exp2_506, _exp2_507);
                                float2 _f32x2_mul_b_508 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_25;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_25) : "l"(*(const unsigned long long*)&ax_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_508));
                                float _exp2_509 = approx_exp2_noftz(tp_25.x);
                                float _exp2_510 = approx_exp2_noftz(tp_25.y);
                                float2 p_25 = make_float2(_exp2_509, _exp2_510);
                                float2 d1_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_25);
                                float2 d2_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_25);
                                float2 dw_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_25);
                                float2 ng_25 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_25);
                                float2 nu_25 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_25);
                                float _rcp_511 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_25, d2_25), dw_25).x);
                                float _rcp_512 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_25, d2_25), dw_25).y);
                                float2 r_52 = make_float2(_rcp_511, _rcp_512);
                                float2 r_51 = r_52;
                                float2 _f32x2_mul_a_513 = mul_f32x2_noftz(ng_25, nu_25);
                                float2 _f32x2_mul_b_514 = mul_f32x2_noftz(r_51, make_float2(bl, bl));
                                float2 m_25;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_513), "l"(*(const unsigned long long*)&_f32x2_mul_b_514));
                                float2 ns_25 = make_float2(((x_26.x < 0.0f) ? p_25.x : 1.0f), ((x_26.y < 0.0f) ? p_25.y : 1.0f));
                                gate_words[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_25, ns_25))) | sign_26;
                                float2 x_27 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[27]));
                                float2 v_27 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[27]));
                                unsigned int sign_27 = (gate_words[27] ^ up_words[27]) & 2147516416u;
                                float _abs_515;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_515) : "f"(x_27.x));
                                float _abs_516;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_516) : "f"(x_27.y));
                                float2 ax_26 = make_float2(_abs_515, _abs_516);
                                float _abs_517;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_517) : "f"(v_27.x));
                                float _abs_518;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_518) : "f"(v_27.y));
                                float2 av_26 = make_float2(_abs_517, _abs_518);
                                float2 _f32x2_mul_b_519 = make_float2(k1, k1);
                                float2 te_26;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_26) : "l"(*(const unsigned long long*)&ax_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_519));
                                float2 _f32x2_mul_b_520 = make_float2(k2, k2);
                                float2 tw_26;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_26) : "l"(*(const unsigned long long*)&av_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_520));
                                float _exp2_521 = approx_exp2(te_26.x);
                                float _exp2_522 = approx_exp2(te_26.y);
                                float2 e_26 = make_float2(_exp2_521, _exp2_522);
                                float _exp2_523 = approx_exp2(tw_26.x);
                                float _exp2_524 = approx_exp2(tw_26.y);
                                float2 w_26 = make_float2(_exp2_523, _exp2_524);
                                float2 _f32x2_mul_b_525 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_26;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_26) : "l"(*(const unsigned long long*)&ax_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_525));
                                float _exp2_526 = approx_exp2_noftz(tp_26.x);
                                float _exp2_527 = approx_exp2_noftz(tp_26.y);
                                float2 p_26 = make_float2(_exp2_526, _exp2_527);
                                float2 d1_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_26);
                                float2 d2_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_26);
                                float2 dw_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_26);
                                float2 ng_26 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_26);
                                float2 nu_26 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_26);
                                float _rcp_528 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_26, d2_26), dw_26).x);
                                float _rcp_529 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_26, d2_26), dw_26).y);
                                float2 r_54 = make_float2(_rcp_528, _rcp_529);
                                float2 r_53 = r_54;
                                float2 _f32x2_mul_a_530 = mul_f32x2_noftz(ng_26, nu_26);
                                float2 _f32x2_mul_b_531 = mul_f32x2_noftz(r_53, make_float2(bl, bl));
                                float2 m_26;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_530), "l"(*(const unsigned long long*)&_f32x2_mul_b_531));
                                float2 ns_26 = make_float2(((x_27.x < 0.0f) ? p_26.x : 1.0f), ((x_27.y < 0.0f) ? p_26.y : 1.0f));
                                gate_words[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_26, ns_26))) | sign_27;
                                float2 x_28 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[28]));
                                float2 v_28 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[28]));
                                unsigned int sign_28 = (gate_words[28] ^ up_words[28]) & 2147516416u;
                                float _abs_532;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_532) : "f"(x_28.x));
                                float _abs_533;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_533) : "f"(x_28.y));
                                float2 ax_27 = make_float2(_abs_532, _abs_533);
                                float _abs_534;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_534) : "f"(v_28.x));
                                float _abs_535;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_535) : "f"(v_28.y));
                                float2 av_27 = make_float2(_abs_534, _abs_535);
                                float2 _f32x2_mul_b_536 = make_float2(k1, k1);
                                float2 te_27;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_27) : "l"(*(const unsigned long long*)&ax_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_536));
                                float2 _f32x2_mul_b_537 = make_float2(k2, k2);
                                float2 tw_27;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_27) : "l"(*(const unsigned long long*)&av_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_537));
                                float _exp2_538 = approx_exp2(te_27.x);
                                float _exp2_539 = approx_exp2(te_27.y);
                                float2 e_27 = make_float2(_exp2_538, _exp2_539);
                                float _exp2_540 = approx_exp2(tw_27.x);
                                float _exp2_541 = approx_exp2(tw_27.y);
                                float2 w_27 = make_float2(_exp2_540, _exp2_541);
                                float2 _f32x2_mul_b_542 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_27;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_27) : "l"(*(const unsigned long long*)&ax_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_542));
                                float _exp2_543 = approx_exp2_noftz(tp_27.x);
                                float _exp2_544 = approx_exp2_noftz(tp_27.y);
                                float2 p_27 = make_float2(_exp2_543, _exp2_544);
                                float2 d1_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_27);
                                float2 d2_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_27);
                                float2 dw_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_27);
                                float2 ng_27 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_27);
                                float2 nu_27 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_27);
                                float _rcp_545 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_27, d2_27), dw_27).x);
                                float _rcp_546 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_27, d2_27), dw_27).y);
                                float2 r_56 = make_float2(_rcp_545, _rcp_546);
                                float2 r_55 = r_56;
                                float2 _f32x2_mul_a_547 = mul_f32x2_noftz(ng_27, nu_27);
                                float2 _f32x2_mul_b_548 = mul_f32x2_noftz(r_55, make_float2(bl, bl));
                                float2 m_27;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_547), "l"(*(const unsigned long long*)&_f32x2_mul_b_548));
                                float2 ns_27 = make_float2(((x_28.x < 0.0f) ? p_27.x : 1.0f), ((x_28.y < 0.0f) ? p_27.y : 1.0f));
                                gate_words[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_27, ns_27))) | sign_28;
                                float2 x_29 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[29]));
                                float2 v_29 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[29]));
                                unsigned int sign_29 = (gate_words[29] ^ up_words[29]) & 2147516416u;
                                float _abs_549;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_549) : "f"(x_29.x));
                                float _abs_550;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_550) : "f"(x_29.y));
                                float2 ax_28 = make_float2(_abs_549, _abs_550);
                                float _abs_551;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_551) : "f"(v_29.x));
                                float _abs_552;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_552) : "f"(v_29.y));
                                float2 av_28 = make_float2(_abs_551, _abs_552);
                                float2 _f32x2_mul_b_553 = make_float2(k1, k1);
                                float2 te_28;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_28) : "l"(*(const unsigned long long*)&ax_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_553));
                                float2 _f32x2_mul_b_554 = make_float2(k2, k2);
                                float2 tw_28;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_28) : "l"(*(const unsigned long long*)&av_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_554));
                                float _exp2_555 = approx_exp2(te_28.x);
                                float _exp2_556 = approx_exp2(te_28.y);
                                float2 e_28 = make_float2(_exp2_555, _exp2_556);
                                float _exp2_557 = approx_exp2(tw_28.x);
                                float _exp2_558 = approx_exp2(tw_28.y);
                                float2 w_28 = make_float2(_exp2_557, _exp2_558);
                                float2 _f32x2_mul_b_559 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_28;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_28) : "l"(*(const unsigned long long*)&ax_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_559));
                                float _exp2_560 = approx_exp2_noftz(tp_28.x);
                                float _exp2_561 = approx_exp2_noftz(tp_28.y);
                                float2 p_28 = make_float2(_exp2_560, _exp2_561);
                                float2 d1_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_28);
                                float2 d2_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_28);
                                float2 dw_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_28);
                                float2 ng_28 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_28);
                                float2 nu_28 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_28);
                                float _rcp_562 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_28, d2_28), dw_28).x);
                                float _rcp_563 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_28, d2_28), dw_28).y);
                                float2 r_58 = make_float2(_rcp_562, _rcp_563);
                                float2 r_57 = r_58;
                                float2 _f32x2_mul_a_564 = mul_f32x2_noftz(ng_28, nu_28);
                                float2 _f32x2_mul_b_565 = mul_f32x2_noftz(r_57, make_float2(bl, bl));
                                float2 m_28;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_564), "l"(*(const unsigned long long*)&_f32x2_mul_b_565));
                                float2 ns_28 = make_float2(((x_29.x < 0.0f) ? p_28.x : 1.0f), ((x_29.y < 0.0f) ? p_28.y : 1.0f));
                                gate_words[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_28, ns_28))) | sign_29;
                                float2 x_30 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[30]));
                                float2 v_30 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[30]));
                                unsigned int sign_30 = (gate_words[30] ^ up_words[30]) & 2147516416u;
                                float _abs_566;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_566) : "f"(x_30.x));
                                float _abs_567;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_567) : "f"(x_30.y));
                                float2 ax_29 = make_float2(_abs_566, _abs_567);
                                float _abs_568;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_568) : "f"(v_30.x));
                                float _abs_569;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_569) : "f"(v_30.y));
                                float2 av_29 = make_float2(_abs_568, _abs_569);
                                float2 _f32x2_mul_b_570 = make_float2(k1, k1);
                                float2 te_29;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_29) : "l"(*(const unsigned long long*)&ax_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_570));
                                float2 _f32x2_mul_b_571 = make_float2(k2, k2);
                                float2 tw_29;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_29) : "l"(*(const unsigned long long*)&av_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_571));
                                float _exp2_572 = approx_exp2(te_29.x);
                                float _exp2_573 = approx_exp2(te_29.y);
                                float2 e_29 = make_float2(_exp2_572, _exp2_573);
                                float _exp2_574 = approx_exp2(tw_29.x);
                                float _exp2_575 = approx_exp2(tw_29.y);
                                float2 w_29 = make_float2(_exp2_574, _exp2_575);
                                float2 _f32x2_mul_b_576 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_29;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_29) : "l"(*(const unsigned long long*)&ax_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_576));
                                float _exp2_577 = approx_exp2_noftz(tp_29.x);
                                float _exp2_578 = approx_exp2_noftz(tp_29.y);
                                float2 p_29 = make_float2(_exp2_577, _exp2_578);
                                float2 d1_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_29);
                                float2 d2_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_29);
                                float2 dw_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_29);
                                float2 ng_29 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_29);
                                float2 nu_29 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_29);
                                float _rcp_579 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_29, d2_29), dw_29).x);
                                float _rcp_580 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_29, d2_29), dw_29).y);
                                float2 r_60 = make_float2(_rcp_579, _rcp_580);
                                float2 r_59 = r_60;
                                float2 _f32x2_mul_a_581 = mul_f32x2_noftz(ng_29, nu_29);
                                float2 _f32x2_mul_b_582 = mul_f32x2_noftz(r_59, make_float2(bl, bl));
                                float2 m_29;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_581), "l"(*(const unsigned long long*)&_f32x2_mul_b_582));
                                float2 ns_29 = make_float2(((x_30.x < 0.0f) ? p_29.x : 1.0f), ((x_30.y < 0.0f) ? p_29.y : 1.0f));
                                gate_words[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_29, ns_29))) | sign_30;
                                float2 x_31 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_words[31]));
                                float2 v_31 = cake_bf16x2_to_f32x2(__as_bf16x2(up_words[31]));
                                unsigned int sign_31 = (gate_words[31] ^ up_words[31]) & 2147516416u;
                                float _abs_583;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_583) : "f"(x_31.x));
                                float _abs_584;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_584) : "f"(x_31.y));
                                float2 ax_30 = make_float2(_abs_583, _abs_584);
                                float _abs_585;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_585) : "f"(v_31.x));
                                float _abs_586;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_586) : "f"(v_31.y));
                                float2 av_30 = make_float2(_abs_585, _abs_586);
                                float2 _f32x2_mul_b_587 = make_float2(k1, k1);
                                float2 te_30;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_30) : "l"(*(const unsigned long long*)&ax_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_587));
                                float2 _f32x2_mul_b_588 = make_float2(k2, k2);
                                float2 tw_30;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_30) : "l"(*(const unsigned long long*)&av_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_588));
                                float _exp2_589 = approx_exp2(te_30.x);
                                float _exp2_590 = approx_exp2(te_30.y);
                                float2 e_30 = make_float2(_exp2_589, _exp2_590);
                                float _exp2_591 = approx_exp2(tw_30.x);
                                float _exp2_592 = approx_exp2(tw_30.y);
                                float2 w_30 = make_float2(_exp2_591, _exp2_592);
                                float2 _f32x2_mul_b_593 = make_float2(-1.442695f, -1.442695f);
                                float2 tp_30;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_30) : "l"(*(const unsigned long long*)&ax_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_593));
                                float _exp2_594 = approx_exp2_noftz(tp_30.x);
                                float _exp2_595 = approx_exp2_noftz(tp_30.y);
                                float2 p_30 = make_float2(_exp2_594, _exp2_595);
                                float2 d1_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_30);
                                float2 d2_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_30);
                                float2 dw_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_30);
                                float2 ng_30 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_30);
                                float2 nu_30 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_30);
                                float _rcp_596 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_30, d2_30), dw_30).x);
                                float _rcp_597 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_30, d2_30), dw_30).y);
                                float2 r_62 = make_float2(_rcp_596, _rcp_597);
                                float2 r_61 = r_62;
                                float2 _f32x2_mul_a_598 = mul_f32x2_noftz(ng_30, nu_30);
                                float2 _f32x2_mul_b_599 = mul_f32x2_noftz(r_61, make_float2(bl, bl));
                                float2 m_30;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_598), "l"(*(const unsigned long long*)&_f32x2_mul_b_599));
                                float2 ns_30 = make_float2(((x_31.x < 0.0f) ? p_30.x : 1.0f), ((x_31.y < 0.0f) ? p_30.y : 1.0f));
                                gate_words[31] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_30, ns_30))) | sign_31;
                                int warp_4 = tid / 32;
                                int local_warp_4 = warp_4 / 4 + warp_4 % 4 * 2;
                                int lane_4_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_5 = 4; tile_col_5 < 8; tile_col_5++) {
                                    unsigned int packed_7[4];
                                    #pragma unroll
                                    for (int pair_12 = 0; pair_12 < 4; pair_12++) {
                                        packed_7[pair_12] = gate_words[tile_col_5 * 4 + pair_12];
                                    }
                                    unsigned int address_13 = replay_hidden_addr + (unsigned int)(((tile_col_5 * 16 + lane_4_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_4 * 16 + lane_4_1 % 16) * 64 + (tile_col_5 * 16 + lane_4_1 / 16 * 8) % 64) * 2);
                                    uint32_t _stmatrix_addr_600 = static_cast<uint32_t>(address_13 ^ (address_13 & 1023) >> 7 << 4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_600), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[3]))
                                        : "memory");
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                __syncthreads();
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&h_sw_r), 0, (row_12 - macro_row_offset) * 128, col_8 * 2, 0, 0, replay_hidden_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll
                            for (int stage_2 = 0; stage_2 < 3; stage_2++) {
                                if (tile_end > first_tile + stage_2) {
                                    int row_13 = first_row_7;
                                    if (col_blocks_3 <= first_col + stage_2) {
                                        row_13 = row_13 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_13 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                        int _min_601 = ((3) < (tile_end - first_tile) ? (3) : (tile_end - first_tile));
                        replay_seq += _min_601;
                    }
                } else {
                    if (tile_word_0 == 0) {
                        int col_blocks_4 = (hidden + 512 - 1) / 512;
                        {
                            col_blocks_4 = (intermediate + 512 - 1) / 512;
                        }
                        int x_32 = -1;
                        int y_2 = -1;
                        int expert_2 = -1;
                        int k_start_2 = 0;
                        int k_end_2 = 0;
                        int first_2 = 0;
                        int first_block_2 = (tile_word_2 * (macro_size / mini_size) + tile_word_3) * (mini_size / 256);
                        int end_block_2 = ((first_block_2 + mini_size / 256) < (tokens / 256) ? (first_block_2 + mini_size / 256) : (tokens / 256));
                        int block_7 = first_block_2;
                        uint32_t fast_div_q_8 = (dh_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), dh_col_blocks_div.multiplier) >> dh_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                        uint32_t fast_div_r_8 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_8 * (uint32_t)(dh_col_blocks_div.divisor);
                        block_7 = first_block_2 + (int)fast_div_q_8;
                        if (block_7 < end_block_2) {
                            int index_3 = counts[3 * experts + block_7];
                            int offset_8 = counts[experts + index_3] / 256;
                            int first_row_8 = ((first_block_2) > (offset_8) ? (first_block_2) : (offset_8));
                            int _min_602 = ((end_block_2) < (offset_8 + counts[index_3] / 256) ? (end_block_2) : (offset_8 + counts[index_3] / 256));
                            int rows_9 = _min_602 - first_row_8;
                            int supergroup_2 = (tile_word_1 - (first_row_8 - first_block_2) * col_blocks_4) / (rows_9 * 8);
                            int full_cols_2 = col_blocks_4 / 8 * 8;
                            int row_14 = 0;
                            int col_9 = 0;
                            if (tile_word_1 - (first_row_8 - first_block_2) * col_blocks_4 < rows_9 * full_cols_2) {
                                row_14 = (tile_word_1 - (first_row_8 - first_block_2) * col_blocks_4) % (rows_9 * 8) / 8;
                                col_9 = supergroup_2 * 8 + (tile_word_1 - (first_row_8 - first_block_2) * col_blocks_4) % 8;
                            } else {
                                row_14 = (tile_word_1 - (first_row_8 - first_block_2) * col_blocks_4 - rows_9 * full_cols_2) / (col_blocks_4 - full_cols_2);
                                col_9 = full_cols_2 + (tile_word_1 - (first_row_8 - first_block_2) * col_blocks_4 - rows_9 * full_cols_2) % (col_blocks_4 - full_cols_2);
                            }
                            if ((supergroup_2 & 1) != 0) {
                                row_14 = rows_9 - row_14 - 1;
                            }
                            x_32 = first_row_8 + row_14 - tile_word_2 * (macro_size / 256);
                            y_2 = col_9;
                            expert_2 = index_3;
                        }
                        int has_hi_2 = 0;
                        has_hi_2 = (int)((y_2 * 2 + 1) * 256 < intermediate);
                        int global_mini_3 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                        int macro_rows_2 = tile_word_2 * (macro_size / 256);
                        int iterations_2 = hidden / 64;
                        if (expert_2 < 0) {
                            if (tid == 0) {
                                bool enabled_value_6 = macros > 1;
                                if (enabled_value_6 != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_603 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                        int mini_rows_6 = ((0) > (_min_603) ? (0) : (_min_603));
                                        int required_3 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_7 = 1;
                                        if (enabled_value_7 != 0) {
                                            int poll_8_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_8_first) : "l"(dy_ready + global_mini_3) : "memory");
                                            int poll_8 = poll_8_first;
                                            while (poll_8 < required_3) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_8_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_8_next) : "l"(dy_ready + global_mini_3) : "memory");
                                                poll_8 = poll_8_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    #pragma unroll 1
                                    for (int idx_12 = 0; idx_12 < iterations_2; idx_12++) {
                                        int _stage_gemm_finished_2 = (int)((unsigned int)(gemm_wide + idx_12) % 4);
                                        int _parity_gemm_finished_2 = (int)((unsigned int)(gemm_wide + idx_12) / 4 % 2) ^ 1;
                                        mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_2) * 8, _parity_gemm_finished_2);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a_nt_addr + (unsigned int)((gemm_wide + idx_12) % 4 * 16384)), "l"((&dy_r)), "r"(0), "r"(x_32 * 256 + cta_rank_0 * 128), "r"(idx_12), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_12) % 4 * 16384)), "l"((&wd_r)), "r"(0), "r"(idx_12 * 64), "r"(y_2 * 2 * 4 + cta_rank_0 * 2), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_12) % 4 * 16384)), "l"((&wd_r)), "r"(0), "r"(idx_12 * 64), "r"((y_2 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int head_2 = ((3) < (iterations_2) ? (3) : (iterations_2));
                                        int tail_3 = 0;
                                        int _min_604 = ((3) < (iterations_2 - head_2) ? (3) : (iterations_2 - head_2));
                                        tail_3 = _min_604;
                                        int _stage_output_finished_4 = (int)((unsigned int)gemm_acc_tiles % 2);
                                        int _parity_output_finished_4 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                        mbarrier_wait(output_finished_addr + (_stage_output_finished_4) * 8, _parity_output_finished_4);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_13 = 0; idx_13 < head_2; idx_13++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_13) % 4)) * 8, 98304);
                                            int _stage_gemm_arrived_6 = (int)((unsigned int)(gemm_wide + idx_13) % 4);
                                            int _parity_gemm_arrived_6 = (int)((unsigned int)(gemm_wide + idx_13) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_6) * 8, _parity_gemm_arrived_6);
                                            int _smem_desc_lo_605 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_13) % 4) * 1024;
                                            int _smem_desc_lo_606 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_13) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_605), "r"(_smem_desc_lo_606), "r"(tmem_accumulator), "r"(((idx_13 == 0) ? 0 : 1)) : "memory");
                                        }
                                        int _stage_output_finished_5 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                        int _parity_output_finished_5 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                        mbarrier_wait(output_finished_addr + (_stage_output_finished_5) * 8, _parity_output_finished_5);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_14 = 0; idx_14 < head_2; idx_14++) {
                                            int _smem_desc_lo_607 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_14) % 4) * 1024;
                                            int _smem_desc_lo_608 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_14) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_607), "r"(_smem_desc_lo_608), "r"((tmem_accumulator + (256))), "r"(((idx_14 == 0) ? 0 : 1)) : "memory");
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_14) % 4)) * 8, (uint16_t)(3));
                                        }
                                        #pragma unroll 1
                                        for (int idx_15 = head_2; idx_15 < iterations_2 - tail_3; idx_15++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_15) % 4)) * 8, 98304);
                                            int _stage_gemm_arrived_7 = (int)((unsigned int)(gemm_wide + idx_15) % 4);
                                            int _parity_gemm_arrived_7 = (int)((unsigned int)(gemm_wide + idx_15) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_7) * 8, _parity_gemm_arrived_7);
                                            int _smem_desc_lo_609 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_15) % 4) * 1024;
                                            int _smem_desc_lo_610 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_15) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_609), "r"(_smem_desc_lo_610), "r"(tmem_accumulator), "r"(((idx_15 == 0) ? 0 : 1)) : "memory");
                                            int _smem_desc_lo_611 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_15) % 4) * 1024;
                                            int _smem_desc_lo_612 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_15) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_611), "r"(_smem_desc_lo_612), "r"((tmem_accumulator + (256))), "r"(((idx_15 == 0) ? 0 : 1)) : "memory");
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_15) % 4)) * 8, (uint16_t)(3));
                                        }
                                        #pragma unroll 1
                                        for (int idx_16 = iterations_2 - tail_3; idx_16 < iterations_2; idx_16++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_16) % 4)) * 8, 98304);
                                            int _stage_gemm_arrived_8 = (int)((unsigned int)(gemm_wide + idx_16) % 4);
                                            int _parity_gemm_arrived_8 = (int)((unsigned int)(gemm_wide + idx_16) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_8) * 8, _parity_gemm_arrived_8);
                                            int _smem_desc_lo_613 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_16) % 4) * 1024;
                                            int _smem_desc_lo_614 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_16) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_613), "r"(_smem_desc_lo_614), "r"(tmem_accumulator), "r"(((idx_16 == 0) ? 0 : 1)) : "memory");
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        #pragma unroll 1
                                        for (int idx_17 = iterations_2 - tail_3; idx_17 < iterations_2; idx_17++) {
                                            int _smem_desc_lo_615 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_17) % 4) * 1024;
                                            int _smem_desc_lo_616 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_17) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_615), "r"(_smem_desc_lo_616), "r"((tmem_accumulator + (256))), "r"(((idx_17 == 0) ? 0 : 1)) : "memory");
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_17) % 4)) * 8, (uint16_t)(3));
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                int _parity_output_arrived_4 = (int)((unsigned int)gemm_out0 % 2);
                                mbarrier_wait(output_arrived_addr, _parity_output_arrived_4);
                                unsigned int packed_8[128];
                                #pragma unroll
                                for (int chunk_8 = 0; chunk_8 < 8; chunk_8++) {
                                    #pragma unroll
                                    for (int sub_4 = 0; sub_4 < 2; sub_4++) {
                                        unsigned int address_14 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_4 * 16 << 16) + (unsigned int)(chunk_8 * 32);
                                        float _tmem_load_14[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                                            : "r"(address_14));
                                        #pragma unroll
                                        for (int pair_13 = 0; pair_13 < 8; pair_13++) {
                                            packed_8[chunk_8 * 16 + sub_4 * 8 + pair_13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[pair_13 * 2], _tmem_load_14[pair_13 * 2 + 1])));
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_3 = 1;
                                if (last_3 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                if (tid == 0) {
                                    int previous_offset_4 = (tile_word_2 + 1) * macro_size;
                                    int output_row_2 = x_32 * 256 + cta_rank_0 * 128;
                                    int _min_617 = ((macro_size) < (tokens - previous_offset_4) ? (macro_size) : (tokens - previous_offset_4));
                                    if (output_row_2 < _min_617) {
                                    }
                                }
                                #pragma unroll
                                for (int chunk_9 = 0; chunk_9 < 8; chunk_9++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_5 = tid / 32;
                                    int lane_6 = tid % 32;
                                    #pragma unroll
                                    for (int half_4 = 0; half_4 < 2; half_4++) {
                                        #pragma unroll
                                        for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                            int row_15 = warp_0_5 * 32 + half_4 * 16 + lane_6 % 16;
                                            int col_10 = col_tile_4 * 16 + lane_6 / 16 * 8;
                                            unsigned int address_15 = d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192) + (unsigned int)((row_15 * 32 + col_10) * 2);
                                            address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                            int offset_9 = chunk_9 * 16 + half_4 * 8 + col_tile_4 * 4;
                                            uint32_t _stmatrix_addr_618 = static_cast<uint32_t>(address_15);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_618), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_9 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&dh_r), 0, x_32 * 256 + cta_rank_0 * 128, y_2 * 2 * 8 + chunk_9, 0, 0, d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                int _parity_output_arrived_5 = (int)((unsigned int)gemm_out1 % 2);
                                mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_5);
                                if (has_hi_2 != 0) {
                                    unsigned int packed_0_2[128];
                                    #pragma unroll
                                    for (int chunk_10 = 0; chunk_10 < 8; chunk_10++) {
                                        #pragma unroll
                                        for (int sub_5 = 0; sub_5 < 2; sub_5++) {
                                            unsigned int address_16 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_5 * 16 << 16) + 256 + (unsigned int)(chunk_10 * 32);
                                            float _tmem_load_15[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                                                : "r"(address_16));
                                            #pragma unroll
                                            for (int pair_14 = 0; pair_14 < 8; pair_14++) {
                                                packed_0_2[chunk_10 * 16 + sub_5 * 8 + pair_14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[pair_14 * 2], _tmem_load_15[pair_14 * 2 + 1])));
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    int last_1_2 = 1;
                                    if (last_1_2 != 0) {
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_11 = 0; chunk_11 < 8; chunk_11++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_6 = tid / 32;
                                        int lane_7 = tid % 32;
                                        #pragma unroll
                                        for (int half_5 = 0; half_5 < 2; half_5++) {
                                            #pragma unroll
                                            for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                                int row_16 = warp_0_6 * 32 + half_5 * 16 + lane_7 % 16;
                                                int col_11 = col_tile_5 * 16 + lane_7 / 16 * 8;
                                                unsigned int address_17 = d_smem_addr + (unsigned int)((8 + chunk_11) % 3 * 8192) + (unsigned int)((row_16 * 32 + col_11) * 2);
                                                address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                                int offset_10 = chunk_11 * 16 + half_5 * 8 + col_tile_5 * 4;
                                                uint32_t _stmatrix_addr_619 = static_cast<uint32_t>(address_17);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_619), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            tma_store_5d((&dh_r), 0, x_32 * 256 + cta_rank_0 * 128, (y_2 * 2 + 1) * 8 + chunk_11, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_11) % 3 * 8192));
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                }
                                if (has_hi_2 == 0) {
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
                                            bool enabled_value_8 = 1;
                                            if (enabled_value_8 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + (shared_down_0 + (macro_rows_2 + x_32) * (intermediate / 256) + y_2 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                            bool enabled_value_2_1 = macros > 1;
                                            if (enabled_value_2_1 != 0) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_3_1 = 1;
                                                if (enabled_value_3_1 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                            if (has_hi_2 != 0) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_4_1 = 1;
                                                if (enabled_value_4_1 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + (shared_down_0 + (macro_rows_2 + x_32) * (intermediate / 256) + y_2 * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        int live_2 = (int)(expert_2 >= 0);
                        gemm_wide += iterations_2 * live_2;
                        gemm_out0 += live_2;
                        gemm_out1 += live_2;
                        gemm_acc_tiles += 2 * live_2;
                    } else if (tile_word_0 == 1) {
                        int col_blocks_5 = intermediate / 128;
                        int num_tiles_1 = tokens / 128 * col_blocks_5;
                        int macro_row_offset_1 = tile_word_2 * (macro_size / 128);
                        int first_tile_1 = tile_word_1 * 16 + cta_rank_0 * 8;
                        int tile_end_1 = num_tiles_1;
                        {
                            int global_mini_4 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                            int mini_tiles_1 = mini_size / 128 * col_blocks_5;
                            first_tile_1 = first_tile_1 + global_mini_4 * mini_tiles_1;
                            int _min_620 = ((num_tiles_1) < ((global_mini_4 + 1) * mini_tiles_1) ? (num_tiles_1) : ((global_mini_4 + 1) * mini_tiles_1));
                            tile_end_1 = _min_620;
                        }
                        int _min_621 = ((8) < (tile_end_1 - first_tile_1) ? (8) : (tile_end_1 - first_tile_1));
                        int tiles = ((0) > (_min_621) ? (0) : (_min_621));
                        if (tiles > 0) {
                            if (tid == 0) {
                                int _min_4 = ((tiles) < (2) ? (tiles) : (2));
                                #pragma unroll 1
                                for (int stage_3 = 0; stage_3 < _min_4; stage_3++) {
                                    int col_blocks_0_2 = intermediate / 128;
                                    int row_17 = (first_tile_1 + stage_3) / col_blocks_0_2;
                                    int col_12 = (first_tile_1 + stage_3) % col_blocks_0_2;
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage_3) % 2)) * 8, 98304);
                                    int parent_1 = row_17 / 2 * (intermediate / 256) + col_12 / 2;
                                    int poll_9_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_9_first) : "l"(dh_ready + parent_1) : "memory");
                                    int poll_9 = poll_9_first;
                                    while (poll_9 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_9_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_9_next) : "l"(dh_ready + parent_1) : "memory");
                                        poll_9 = poll_9_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    bool enabled_value_9 = tile_word_2 > 0;
                                    if (enabled_value_9 != 0) {
                                        int poll_0_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_0_first) : "l"(replay_gu + parent_1) : "memory");
                                        int poll_0 = poll_0_first;
                                        while (poll_0 < 4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_0_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_0_next) : "l"(replay_gu + parent_1) : "memory");
                                            poll_0 = poll_0_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_dh_addr + (unsigned int)((swiglu_seq + stage_3) % 2 * 32768)), "l"((&dh_sw_r)), "r"(0), "r"((row_17 - macro_row_offset_1) * 128), "r"(col_12 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage_3) % 2)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_gate_addr + (unsigned int)((swiglu_seq + stage_3) % 2 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_17 - macro_row_offset_1) * 128), "r"(col_12 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage_3) % 2)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_up_addr + (unsigned int)((swiglu_seq + stage_3) % 2 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_17 - macro_row_offset_1) * 128), "r"(col_12 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage_3) % 2)) * 8) : "memory");
                                }
                            }
                            #pragma unroll 1
                            for (int index_4 = 0; index_4 < tiles; index_4++) {
                                int stage_4 = (swiglu_seq + index_4) % 2;
                                int _parity_swiglu_arrived = (int)((unsigned int)(swiglu_seq + index_4) / 2 % 2);
                                mbarrier_wait(swiglu_arrived_addr + (stage_4) * 8, _parity_swiglu_arrived);
                                int row_18 = (first_tile_1 + index_4) / col_blocks_5;
                                int col_13 = (first_tile_1 + index_4) % col_blocks_5;
                                int tile_row = tid % 128;
                                int half_6 = tid / 128;
                                int local_row = (row_18 - macro_row_offset_1) * 128 + tile_row;
                                int peer_7 = schedule_rank[row_18 * 128 + tile_row];
                                float weight_3 = weights[local_row];
                                float router_gradient = 0.0f;
                                unsigned int gate_tile = sw_gate_addr + (unsigned int)(stage_4 * 32768);
                                unsigned int up_tile = sw_up_addr + (unsigned int)(stage_4 * 32768);
                                unsigned int dh_tile = sw_dh_addr + (unsigned int)(stage_4 * 32768);
                                unsigned int row_offset = (unsigned int)(half_6 * 16384 + tile_row * 128);
                                unsigned int row_swizzle = gate_tile + row_offset >> 7 << 4 & 112;
                                #pragma unroll 1
                                for (int group = 0; group < 16; group++) {
                                    unsigned int column = row_offset | (unsigned int)(group * 8) ^ row_swizzle;
                                    unsigned int address_18 = gate_tile + column;
                                    unsigned int address_0 = up_tile + column;
                                    unsigned int words[2];
                                    uint64_t _smem_raw_622;
                                    asm volatile("ld.shared.b64 %0, [%1];" : "=l"(_smem_raw_622) : "r"(address_18) : "memory");
                                    words[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_622)[0];
                                    words[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_622)[1];
                                    unsigned int words_0[2];
                                    uint64_t _smem_raw_623;
                                    asm volatile("ld.shared.b64 %0, [%1];" : "=l"(_smem_raw_623) : "r"(address_0) : "memory");
                                    words_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_623)[0];
                                    words_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_623)[1];
                                    unsigned int words_1[2];
                                    uint64_t _smem_raw_624;
                                    asm volatile("ld.shared.b64 %0, [%1];" : "=l"(_smem_raw_624) : "r"(dh_tile + column) : "memory");
                                    words_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_624)[0];
                                    words_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_624)[1];
                                    unsigned int dgate_packed[2];
                                    unsigned int dup_packed[2];
                                    #pragma unroll
                                    for (int pair_15 = 0; pair_15 < 2; pair_15++) {
                                        float2 _cvt_f32_5 = make_float2(__uint_as_float(words[pair_15] << 16), __uint_as_float(words[pair_15] & 4294901760u));
                                        float2 _cvt_f32_6 = make_float2(__uint_as_float(words_0[pair_15] << 16), __uint_as_float(words_0[pair_15] & 4294901760u));
                                        float2 _cvt_f32_7 = make_float2(__uint_as_float(words_1[pair_15] << 16), __uint_as_float(words_1[pair_15] & 4294901760u));
                                        float _abs_625;
                                        asm("abs.f32 %0, %1;" : "=f"(_abs_625) : "f"(_cvt_f32_5.x));
                                        float _abs_626;
                                        asm("abs.f32 %0, %1;" : "=f"(_abs_626) : "f"(_cvt_f32_5.y));
                                        float2 ax_31 = make_float2(_abs_625, _abs_626);
                                        float _abs_627;
                                        asm("abs.f32 %0, %1;" : "=f"(_abs_627) : "f"(_cvt_f32_6.x));
                                        float _abs_628;
                                        asm("abs.f32 %0, %1;" : "=f"(_abs_628) : "f"(_cvt_f32_6.y));
                                        float2 av_31 = make_float2(_abs_627, _abs_628);
                                        float2 _f32x2_mul_b_629 = make_float2(k1, k1);
                                        float2 te_31;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_31) : "l"(*(const unsigned long long*)&ax_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_629));
                                        float2 _f32x2_mul_b_630 = make_float2(k2, k2);
                                        float2 tw_31;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_31) : "l"(*(const unsigned long long*)&av_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_630));
                                        float _exp2_631 = approx_exp2_noftz(te_31.x);
                                        float _exp2_632 = approx_exp2_noftz(te_31.y);
                                        float2 e_31 = make_float2(_exp2_631, _exp2_632);
                                        float _exp2_633 = approx_exp2_noftz(tw_31.x);
                                        float _exp2_634 = approx_exp2_noftz(tw_31.y);
                                        float2 w_31 = make_float2(_exp2_633, _exp2_634);
                                        float2 _f32x2_mul_b_635 = make_float2(-1.442695f, -1.442695f);
                                        float2 tp_31;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_31) : "l"(*(const unsigned long long*)&ax_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_635));
                                        float _exp2_636 = approx_exp2_noftz(tp_31.x);
                                        float _exp2_637 = approx_exp2_noftz(tp_31.y);
                                        float2 p_31 = make_float2(_exp2_636, _exp2_637);
                                        float2 d1_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_31);
                                        float2 d2_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_31);
                                        float2 dw_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_31);
                                        float2 ng_31 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_31);
                                        float2 nu_31 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_31);
                                        float2 d12;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12) : "l"(*(const unsigned long long*)&d1_31), "l"(*(const unsigned long long*)&d2_31));
                                        float _rcp_638 = approx_rcp(mul_f32x2_noftz(d12, dw_31).x);
                                        float _rcp_639 = approx_rcp(mul_f32x2_noftz(d12, dw_31).y);
                                        float2 r_63 = make_float2(_rcp_638, _rcp_639);
                                        float2 r_64 = r_63;
                                        float2 _f32x2_mul_a_640 = mul_f32x2_noftz(make_float2(weight_3, weight_3), _cvt_f32_7);
                                        float2 _f32x2_mul_b_641 = mul_f32x2_noftz(r_64, r_64);
                                        float2 m1;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_640), "l"(*(const unsigned long long*)&_f32x2_mul_b_641));
                                        unsigned int lo = __as_u32(mul_f32x2_noftz(ng_31, make_float2(situ_beta, situ_beta)).x) | __as_u32(_cvt_f32_5.x) & 2147483648u;
                                        unsigned int hi = __as_u32(mul_f32x2_noftz(ng_31, make_float2(situ_beta, situ_beta)).y) | __as_u32(_cvt_f32_5.y) & 2147483648u;
                                        float2 bng = make_float2(__uint_as_float(lo), __uint_as_float(hi));
                                        unsigned int lo_0 = __as_u32(mul_f32x2_noftz(nu_31, make_float2(situ_lb, situ_lb)).x) | __as_u32(_cvt_f32_6.x) & 2147483648u;
                                        unsigned int hi_0 = __as_u32(mul_f32x2_noftz(nu_31, make_float2(situ_lb, situ_lb)).y) | __as_u32(_cvt_f32_6.y) & 2147483648u;
                                        float2 lnu = make_float2(__uint_as_float(lo_0), __uint_as_float(hi_0));
                                        float2 q = make_float2(((_cvt_f32_5.x < 0.0f) ? 1.0f : p_31.x), ((_cvt_f32_5.y < 0.0f) ? 1.0f : p_31.y));
                                        float2 post = make_float2(((_cvt_f32_5.x < 0.0f) ? p_31.x : 1.0f), ((_cvt_f32_5.y < 0.0f) ? p_31.y : 1.0f));
                                        float2 inner = fma_f32x2_rn_noftz(bng, mul_f32x2_noftz(d1_31, q), mul_f32x2_noftz(mul_f32x2_noftz(e_31, make_float2(4.0f, 4.0f)), d2_31));
                                        float2 _f32x2_mul_a_642 = mul_f32x2_noftz(mul_f32x2_noftz(m1, lnu), mul_f32x2_noftz(dw_31, inner));
                                        float2 dg;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg) : "l"(*(const unsigned long long*)&_f32x2_mul_a_642), "l"(*(const unsigned long long*)&post));
                                        float2 _f32x2_mul_a_643 = mul_f32x2_noftz(mul_f32x2_noftz(m1, bng), mul_f32x2_noftz(mul_f32x2_noftz(w_31, make_float2(4.0f, 4.0f)), d12));
                                        float2 du;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du) : "l"(*(const unsigned long long*)&_f32x2_mul_a_643), "l"(*(const unsigned long long*)&post));
                                        float2 _f32x2_mul_a_644 = mul_f32x2_noftz(mul_f32x2_noftz(ng_31, nu_31), mul_f32x2_noftz(r_64, make_float2(bl, bl)));
                                        float2 habs;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs) : "l"(*(const unsigned long long*)&_f32x2_mul_a_644), "l"(*(const unsigned long long*)&post));
                                        float2 s = make_float2(__uint_as_float(__as_u32(_cvt_f32_5.x) ^ __as_u32(_cvt_f32_6.x)), __uint_as_float(__as_u32(_cvt_f32_5.y) ^ __as_u32(_cvt_f32_6.y)));
                                        unsigned int lo_1 = __as_u32(habs.x) | __as_u32(s.x) & 2147483648u;
                                        unsigned int hi_1 = __as_u32(habs.y) | __as_u32(s.y) & 2147483648u;
                                        dgate_packed[pair_15] = __as_u32(__float22bfloat162_rn(dg));
                                        dup_packed[pair_15] = __as_u32(__float22bfloat162_rn(du));
                                        float2 situ_stored = cake_bf16x2_to_f32x2(__float22bfloat162_rn(make_float2(__uint_as_float(lo_1), __uint_as_float(hi_1))));
                                        float hidden_x = ((tile_word_2 > 0) ? make_float2(__uint_as_float(lo_1), __uint_as_float(hi_1)).x : situ_stored.x);
                                        float hidden_y = ((tile_word_2 > 0) ? make_float2(__uint_as_float(lo_1), __uint_as_float(hi_1)).y : situ_stored.y);
                                        router_gradient = router_gradient + (_cvt_f32_7.x * hidden_x + _cvt_f32_7.y * hidden_y);
                                    }
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(address_18), "r"(dgate_packed[0]), "r"(dgate_packed[1]) : "memory");
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(address_0), "r"(dup_packed[0]), "r"(dup_packed[1]) : "memory");
                                }
                                if (half_6 != 0) {
                                    flat_view_0[stage_4 * 128 + tile_row] = router_gradient;
                                }
                                __syncthreads();
                                if (half_6 == 0 && peer_7 >= 0) {
                                    router_gradient = router_gradient + flat_view_0[stage_4 * 128 + tile_row];
                                    partials[local_row * col_blocks_5 + col_13] = router_gradient;
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                __syncthreads();
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&dg_sw_r), 0, (row_18 - macro_row_offset_1) * 128, col_13 * 2, 0, 0, sw_gate_addr + (unsigned int)(stage_4 * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&du_sw_r), 0, (row_18 - macro_row_offset_1) * 128, col_13 * 2, 0, 0, sw_up_addr + (unsigned int)(stage_4 * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                    if (tiles > index_4 + 2) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                        int col_blocks_0_3 = intermediate / 128;
                                        int row_0 = (first_tile_1 + index_4 + 2) / col_blocks_0_3;
                                        int col_0 = (first_tile_1 + index_4 + 2) % col_blocks_0_3;
                                        mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + index_4 + 2) % 2)) * 8, 98304);
                                        int parent_2 = row_0 / 2 * (intermediate / 256) + col_0 / 2;
                                        int poll_10_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_10_first) : "l"(dh_ready + parent_2) : "memory");
                                        int poll_10 = poll_10_first;
                                        while (poll_10 < 2) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_10_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_10_next) : "l"(dh_ready + parent_2) : "memory");
                                            poll_10 = poll_10_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                        bool enabled_value_10 = tile_word_2 > 0;
                                        if (enabled_value_10 != 0) {
                                            int poll_1_1_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_1_first) : "l"(replay_gu + parent_2) : "memory");
                                            int poll_1_1 = poll_1_1_first;
                                            while (poll_1_1 < 4) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_1_1_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_1_next) : "l"(replay_gu + parent_2) : "memory");
                                                poll_1_1 = poll_1_1_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                            :: "r"(sw_dh_addr + (unsigned int)((swiglu_seq + index_4 + 2) % 2 * 32768)), "l"((&dh_sw_r)), "r"(0), "r"((row_0 - macro_row_offset_1) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + index_4 + 2) % 2)) * 8) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                            :: "r"(sw_gate_addr + (unsigned int)((swiglu_seq + index_4 + 2) % 2 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_0 - macro_row_offset_1) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + index_4 + 2) % 2)) * 8) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                            :: "r"(sw_up_addr + (unsigned int)((swiglu_seq + index_4 + 2) % 2 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_0 - macro_row_offset_1) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + index_4 + 2) % 2)) * 8) : "memory");
                                    }
                                }
                                __syncthreads();
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                #pragma unroll 1
                                for (int index_5 = 0; index_5 < tiles; index_5++) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + ((first_tile_1 + index_5) / col_blocks_5 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                        if (tid == 0) {
                            bool enabled_value_11 = macros > 1;
                            if (enabled_value_11 != 0) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                        swiglu_seq += tiles;
                    } else {
                        if (tile_word_0 == 2) {
                            int col_blocks_6 = (intermediate + 512 - 1) / 512;
                            {
                                col_blocks_6 = (hidden + 512 - 1) / 512;
                            }
                            int x_33 = -1;
                            int y_3 = -1;
                            int expert_3 = -1;
                            int k_start_3 = 0;
                            int k_end_3 = 0;
                            int first_3 = 0;
                            int first_block_3 = (tile_word_2 * (macro_size / mini_size) + tile_word_3) * (mini_size / 256);
                            int end_block_3 = ((first_block_3 + mini_size / 256) < (tokens / 256) ? (first_block_3 + mini_size / 256) : (tokens / 256));
                            int block_8 = first_block_3;
                            uint32_t fast_div_q_9 = (dx_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), dx_col_blocks_div.multiplier) >> dx_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                            uint32_t fast_div_r_9 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_9 * (uint32_t)(dx_col_blocks_div.divisor);
                            block_8 = first_block_3 + (int)fast_div_q_9;
                            if (block_8 < end_block_3) {
                                int index_6 = counts[3 * experts + block_8];
                                int offset_11 = counts[experts + index_6] / 256;
                                int first_row_9 = ((first_block_3) > (offset_11) ? (first_block_3) : (offset_11));
                                int _min_645 = ((end_block_3) < (offset_11 + counts[index_6] / 256) ? (end_block_3) : (offset_11 + counts[index_6] / 256));
                                int rows_10 = _min_645 - first_row_9;
                                int supergroup_3 = (tile_word_1 - (first_row_9 - first_block_3) * col_blocks_6) / (rows_10 * 8);
                                int full_cols_3 = col_blocks_6 / 8 * 8;
                                int row_19 = 0;
                                int col_14 = 0;
                                if (tile_word_1 - (first_row_9 - first_block_3) * col_blocks_6 < rows_10 * full_cols_3) {
                                    row_19 = (tile_word_1 - (first_row_9 - first_block_3) * col_blocks_6) % (rows_10 * 8) / 8;
                                    col_14 = supergroup_3 * 8 + (tile_word_1 - (first_row_9 - first_block_3) * col_blocks_6) % 8;
                                } else {
                                    row_19 = (tile_word_1 - (first_row_9 - first_block_3) * col_blocks_6 - rows_10 * full_cols_3) / (col_blocks_6 - full_cols_3);
                                    col_14 = full_cols_3 + (tile_word_1 - (first_row_9 - first_block_3) * col_blocks_6 - rows_10 * full_cols_3) % (col_blocks_6 - full_cols_3);
                                }
                                if ((supergroup_3 & 1) != 0) {
                                    row_19 = rows_10 - row_19 - 1;
                                }
                                x_33 = first_row_9 + row_19 - tile_word_2 * (macro_size / 256);
                                y_3 = col_14;
                                expert_3 = index_6;
                            }
                            int has_hi_3 = 0;
                            has_hi_3 = (int)((y_3 * 2 + 1) * 256 < hidden);
                            int global_mini_5 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                            int macro_rows_3 = tile_word_2 * (macro_size / 256);
                            int iterations_3 = intermediate / 64 + intermediate / 64;
                            if (expert_3 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_12 = macros > 1;
                                    if (enabled_value_12 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            bool enabled_value_13 = 1;
                                            if (enabled_value_13 != 0) {
                                                int poll_11_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_11_first) : "l"(dg_ready + (shared_rows + macro_rows_3 + x_33)) : "memory");
                                                int poll_11 = poll_11_first;
                                                while (poll_11 < row_count) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int poll_11_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_11_next) : "l"(dg_ready + (shared_rows + macro_rows_3 + x_33)) : "memory");
                                                    poll_11 = poll_11_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                            int _min_646 = ((mini_size) < (tokens - global_mini_5 * mini_size) ? (mini_size) : (tokens - global_mini_5 * mini_size));
                                            int mini_rows_7 = ((0) > (_min_646) ? (0) : (_min_646));
                                            int required_4 = (mini_rows_7 + 127) / 128 * ((intermediate + 511) / 512);
                                        }
                                        #pragma unroll 1
                                        for (int idx_18 = 0; idx_18 < iterations_3; idx_18++) {
                                            int _stage_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_18) % 4);
                                            int _parity_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_18) / 4 % 2) ^ 1;
                                            mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_3) * 8, _parity_gemm_finished_3);
                                            if (idx_18 < intermediate / 64) {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_nt_addr + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&dg_r)), "r"(0), "r"(x_33 * 256 + cta_rank_0 * 128), "r"(idx_18), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&wg_r)), "r"(0), "r"(idx_18 * 64), "r"(y_3 * 2 * 4 + cta_rank_0 * 2), "r"(expert_3), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&wg_r)), "r"(0), "r"(idx_18 * 64), "r"((y_3 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_3), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            } else {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_nt_addr + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&du_r)), "r"(0), "r"(x_33 * 256 + cta_rank_0 * 128), "r"(idx_18 - intermediate / 64), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&wu_r)), "r"(0), "r"((idx_18 - intermediate / 64) * 64), "r"(y_3 * 2 * 4 + cta_rank_0 * 2), "r"(expert_3), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&wu_r)), "r"(0), "r"((idx_18 - intermediate / 64) * 64), "r"((y_3 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_3), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int head_3 = ((3) < (iterations_3) ? (3) : (iterations_3));
                                            int tail_4 = 0;
                                            int _min_647 = ((3) < (iterations_3 - head_3) ? (3) : (iterations_3 - head_3));
                                            tail_4 = _min_647;
                                            int _stage_output_finished_6 = (int)((unsigned int)gemm_acc_tiles % 2);
                                            int _parity_output_finished_6 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                            mbarrier_wait(output_finished_addr + (_stage_output_finished_6) * 8, _parity_output_finished_6);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_19 = 0; idx_19 < head_3; idx_19++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_19) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_9 = (int)((unsigned int)(gemm_wide + idx_19) % 4);
                                                int _parity_gemm_arrived_9 = (int)((unsigned int)(gemm_wide + idx_19) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_9) * 8, _parity_gemm_arrived_9);
                                                int _smem_desc_lo_648 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_19) % 4) * 1024;
                                                int _smem_desc_lo_649 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_19) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_648), "r"(_smem_desc_lo_649), "r"(tmem_accumulator), "r"(((idx_19 == 0) ? 0 : 1)) : "memory");
                                            }
                                            int _stage_output_finished_7 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                            int _parity_output_finished_7 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                            mbarrier_wait(output_finished_addr + (_stage_output_finished_7) * 8, _parity_output_finished_7);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_20 = 0; idx_20 < head_3; idx_20++) {
                                                int _smem_desc_lo_650 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_20) % 4) * 1024;
                                                int _smem_desc_lo_651 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_20) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_650), "r"(_smem_desc_lo_651), "r"((tmem_accumulator + (256))), "r"(((idx_20 == 0) ? 0 : 1)) : "memory");
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_20) % 4)) * 8, (uint16_t)(3));
                                            }
                                            #pragma unroll 1
                                            for (int idx_21 = head_3; idx_21 < iterations_3 - tail_4; idx_21++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_21) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_10 = (int)((unsigned int)(gemm_wide + idx_21) % 4);
                                                int _parity_gemm_arrived_10 = (int)((unsigned int)(gemm_wide + idx_21) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_10) * 8, _parity_gemm_arrived_10);
                                                int _smem_desc_lo_652 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                                int _smem_desc_lo_653 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_21) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_652), "r"(_smem_desc_lo_653), "r"(tmem_accumulator), "r"(((idx_21 == 0) ? 0 : 1)) : "memory");
                                                int _smem_desc_lo_654 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                                int _smem_desc_lo_655 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_21) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_654), "r"(_smem_desc_lo_655), "r"((tmem_accumulator + (256))), "r"(((idx_21 == 0) ? 0 : 1)) : "memory");
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_21) % 4)) * 8, (uint16_t)(3));
                                            }
                                            #pragma unroll 1
                                            for (int idx_22 = iterations_3 - tail_4; idx_22 < iterations_3; idx_22++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_22) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_11 = (int)((unsigned int)(gemm_wide + idx_22) % 4);
                                                int _parity_gemm_arrived_11 = (int)((unsigned int)(gemm_wide + idx_22) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_11) * 8, _parity_gemm_arrived_11);
                                                int _smem_desc_lo_656 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_22) % 4) * 1024;
                                                int _smem_desc_lo_657 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_22) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_656), "r"(_smem_desc_lo_657), "r"(tmem_accumulator), "r"(((idx_22 == 0) ? 0 : 1)) : "memory");
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            #pragma unroll 1
                                            for (int idx_23 = iterations_3 - tail_4; idx_23 < iterations_3; idx_23++) {
                                                int _smem_desc_lo_658 = (((a_nt_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_23) % 4) * 1024;
                                                int _smem_desc_lo_659 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_23) % 4) * 1024;
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
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_658), "r"(_smem_desc_lo_659), "r"((tmem_accumulator + (256))), "r"(((idx_23 == 0) ? 0 : 1)) : "memory");
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_23) % 4)) * 8, (uint16_t)(3));
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    int _parity_output_arrived_6 = (int)((unsigned int)gemm_out0 % 2);
                                    mbarrier_wait(output_arrived_addr, _parity_output_arrived_6);
                                    unsigned int packed_9[128];
                                    #pragma unroll
                                    for (int chunk_12 = 0; chunk_12 < 8; chunk_12++) {
                                        #pragma unroll
                                        for (int sub_6 = 0; sub_6 < 2; sub_6++) {
                                            unsigned int address_19 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_6 * 16 << 16) + (unsigned int)(chunk_12 * 32);
                                            float _tmem_load_16[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15]))
                                                : "r"(address_19));
                                            #pragma unroll
                                            for (int pair_16 = 0; pair_16 < 8; pair_16++) {
                                                packed_9[chunk_12 * 16 + sub_6 * 8 + pair_16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[pair_16 * 2], _tmem_load_16[pair_16 * 2 + 1])));
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    int last_4 = 1;
                                    if (last_4 != 0) {
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                    if (tid == 0) {
                                        int previous_offset_5 = (tile_word_2 + 1) * macro_size;
                                        int output_row_3 = x_33 * 256 + cta_rank_0 * 128;
                                        int _min_660 = ((macro_size) < (tokens - previous_offset_5) ? (macro_size) : (tokens - previous_offset_5));
                                        if (output_row_3 < _min_660) {
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_13 = 0; chunk_13 < 8; chunk_13++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_7 = tid / 32;
                                        int lane_8 = tid % 32;
                                        #pragma unroll
                                        for (int half_7 = 0; half_7 < 2; half_7++) {
                                            #pragma unroll
                                            for (int col_tile_6 = 0; col_tile_6 < 2; col_tile_6++) {
                                                int row_20 = warp_0_7 * 32 + half_7 * 16 + lane_8 % 16;
                                                int col_15 = col_tile_6 * 16 + lane_8 / 16 * 8;
                                                unsigned int address_20 = d_smem_addr + (unsigned int)(chunk_13 % 3 * 8192) + (unsigned int)((row_20 * 32 + col_15) * 2);
                                                address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                                int offset_12 = chunk_13 * 16 + half_7 * 8 + col_tile_6 * 4;
                                                uint32_t _stmatrix_addr_661 = static_cast<uint32_t>(address_20);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_661), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[offset_12 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            tma_store_5d((&dx_r), 0, x_33 * 256 + cta_rank_0 * 128, y_3 * 2 * 8 + chunk_13, 0, 0, d_smem_addr + (unsigned int)(chunk_13 % 3 * 8192));
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    int _parity_output_arrived_7 = (int)((unsigned int)gemm_out1 % 2);
                                    mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_7);
                                    if (has_hi_3 != 0) {
                                        unsigned int packed_0_3[128];
                                        #pragma unroll
                                        for (int chunk_14 = 0; chunk_14 < 8; chunk_14++) {
                                            #pragma unroll
                                            for (int sub_7 = 0; sub_7 < 2; sub_7++) {
                                                unsigned int address_21 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_7 * 16 << 16) + 256 + (unsigned int)(chunk_14 * 32);
                                                float _tmem_load_17[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[15]))
                                                    : "r"(address_21));
                                                #pragma unroll
                                                for (int pair_17 = 0; pair_17 < 8; pair_17++) {
                                                    packed_0_3[chunk_14 * 16 + sub_7 * 8 + pair_17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[pair_17 * 2], _tmem_load_17[pair_17 * 2 + 1])));
                                                }
                                            }
                                        }
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        int last_1_3 = 1;
                                        if (last_1_3 != 0) {
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                            }
                                        }
                                        #pragma unroll
                                        for (int chunk_15 = 0; chunk_15 < 8; chunk_15++) {
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            int warp_0_8 = tid / 32;
                                            int lane_9 = tid % 32;
                                            #pragma unroll
                                            for (int half_8 = 0; half_8 < 2; half_8++) {
                                                #pragma unroll
                                                for (int col_tile_7 = 0; col_tile_7 < 2; col_tile_7++) {
                                                    int row_21 = warp_0_8 * 32 + half_8 * 16 + lane_9 % 16;
                                                    int col_16 = col_tile_7 * 16 + lane_9 / 16 * 8;
                                                    unsigned int address_22 = d_smem_addr + (unsigned int)((8 + chunk_15) % 3 * 8192) + (unsigned int)((row_21 * 32 + col_16) * 2);
                                                    address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                                    int offset_13 = chunk_15 * 16 + half_8 * 8 + col_tile_7 * 4;
                                                    uint32_t _stmatrix_addr_662 = static_cast<uint32_t>(address_22);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_662), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                tma_store_5d((&dx_r), 0, x_33 * 256 + cta_rank_0 * 128, (y_3 * 2 + 1) * 8 + chunk_15, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_15) % 3 * 8192));
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                    }
                                    if (has_hi_3 == 0) {
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
                                                bool enabled_value_14 = 1;
                                                if (enabled_value_14 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dx_ready)) + (global_mini_5))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                                bool enabled_value_5_1 = macros > 1;
                                                if (enabled_value_5_1 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_6_1 = 1;
                                                    if (enabled_value_6_1 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                                if (has_hi_3 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_7_1 = 1;
                                                    if (enabled_value_7_1 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dx_ready)) + (global_mini_5))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            int live_3 = (int)(expert_3 >= 0);
                            gemm_wide += iterations_3 * live_3;
                            gemm_out0 += live_3;
                            gemm_out1 += live_3;
                            gemm_acc_tiles += 2 * live_3;
                        } else if (tile_word_0 == 3) {
                            int col_blocks_7 = (macro_size + 512 - 1) / 512;
                            {
                                col_blocks_7 = (intermediate + 512 - 1) / 512;
                            }
                            int x_34 = -1;
                            int y_4 = -1;
                            int expert_4 = -1;
                            int k_start_4 = 0;
                            int k_end_4 = 0;
                            int first_4 = 0;
                            int row_blocks = hidden / 256;
                            int expert_idx = 0;
                            int local_task = tile_word_1;
                            uint32_t fast_div_q_10 = (shared_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), shared_wgrad_div.multiplier) >> shared_wgrad_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                            uint32_t fast_div_r_10 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_10 * (uint32_t)(shared_wgrad_div.divisor);
                            expert_idx = (int)fast_div_q_10;
                            local_task = (int)fast_div_r_10;
                            int offset_14 = counts[experts + expert_idx];
                            int real = counts[2 * experts + expert_idx];
                            real = (real + 64 - 1) / 64 * 64;
                            int _max_663 = ((offset_14) > (tile_word_2 * macro_size) ? (offset_14) : (tile_word_2 * macro_size));
                            k_start_4 = _max_663;
                            int _min_665 = (((tile_word_2 + 1) * macro_size) < (tokens) ? ((tile_word_2 + 1) * macro_size) : (tokens));
                            int _min_664 = ((offset_14 + real) < (_min_665) ? (offset_14 + real) : (_min_665));
                            k_end_4 = _min_664;
                            first_4 = (int)(k_start_4 == offset_14);
                            if (k_start_4 < k_end_4 && local_task < row_blocks * col_blocks_7) {
                                int supergroup_4 = local_task / (row_blocks * 8);
                                int full_cols_4 = col_blocks_7 / 8 * 8;
                                int row_22 = 0;
                                int col_17 = 0;
                                if (local_task < row_blocks * full_cols_4) {
                                    row_22 = local_task % (row_blocks * 8) / 8;
                                    col_17 = supergroup_4 * 8 + local_task % 8;
                                } else {
                                    row_22 = (local_task - row_blocks * full_cols_4) / (col_blocks_7 - full_cols_4);
                                    col_17 = full_cols_4 + (local_task - row_blocks * full_cols_4) % (col_blocks_7 - full_cols_4);
                                }
                                if ((supergroup_4 & 1) != 0) {
                                    row_22 = row_blocks - row_22 - 1;
                                }
                                x_34 = row_22;
                                y_4 = col_17;
                                expert_4 = expert_idx;
                            }
                            int has_hi_4 = 0;
                            has_hi_4 = (int)((y_4 * 2 + 1) * 256 < intermediate);
                            int global_mini_6 = tile_word_2 * (macro_size / mini_size);
                            int macro_rows_4 = tile_word_2 * (macro_size / 256);
                            int iterations_4 = hidden / 64;
                            iterations_4 = (k_end_4 - k_start_4) / 64;
                            if (expert_4 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_15 = macros > 1;
                                    if (enabled_value_15 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        #pragma unroll 1
                                        for (int idx_24 = 0; idx_24 < iterations_4; idx_24++) {
                                            int token_row = k_start_4 + idx_24 * 64;
                                            if (idx_24 == 0 || token_row % 256 == 0) {
                                                bool enabled_value_16 = tile_word_2 > 0;
                                                if (enabled_value_16 != 0) {
                                                    int poll_12_first;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_12_first) : "l"(replay_h + (token_row / 256)) : "memory");
                                                    int poll_12 = poll_12_first;
                                                    while (poll_12 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int poll_12_next;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_12_next) : "l"(replay_h + (token_row / 256)) : "memory");
                                                        poll_12 = poll_12_next;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_24 == 0 || token_row % mini_size == 0) {
                                                int input_mini = token_row / mini_size;
                                                int input_rows = ((mini_size) < (tokens - input_mini * mini_size) ? (mini_size) : (tokens - input_mini * mini_size));
                                                int input_count = (input_rows + 127) / 128 * ((hidden + 511) / 512);
                                                bool enabled_value_17 = 1;
                                                if (enabled_value_17 != 0) {
                                                    int poll_13_first;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_13_first) : "l"(dy_ready + input_mini) : "memory");
                                                    int poll_13 = poll_13_first;
                                                    while (poll_13 < input_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int poll_13_next;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_13_next) : "l"(dy_ready + input_mini) : "memory");
                                                        poll_13 = poll_13_next;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            int _stage_gemm_finished_4 = (int)((unsigned int)(gemm_wide + idx_24) % 4);
                                            int _parity_gemm_finished_4 = (int)((unsigned int)(gemm_wide + idx_24) / 4 % 2) ^ 1;
                                            mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_4) * 8, _parity_gemm_finished_4);
                                            int local_row_1 = k_start_4 + idx_24 * 64 - tile_word_2 * macro_size;
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(mn_view_addr + (unsigned int)((gemm_wide + idx_24) % 4 * 16384)), "l"((&dy_atb_r)), "r"(0), "r"(local_row_1), "r"(x_34 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_24) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_24) % 4 * 16384)), "l"((&h_atb_r)), "r"(0), "r"(local_row_1), "r"(y_4 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_24) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_24) % 4 * 16384)), "l"((&h_atb_r)), "r"(0), "r"(local_row_1), "r"((y_4 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_24) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int head_4 = ((3) < (iterations_4) ? (3) : (iterations_4));
                                            int tail_5 = 0;
                                            int _min_666 = ((3) < (iterations_4 - head_4) ? (3) : (iterations_4 - head_4));
                                            tail_5 = _min_666;
                                            int _stage_output_finished_8 = (int)((unsigned int)gemm_acc_tiles % 2);
                                            int _parity_output_finished_8 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                            mbarrier_wait(output_finished_addr + (_stage_output_finished_8) * 8, _parity_output_finished_8);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_25 = 0; idx_25 < head_4; idx_25++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_25) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_12 = (int)((unsigned int)(gemm_wide + idx_25) % 4);
                                                int _parity_gemm_arrived_12 = (int)((unsigned int)(gemm_wide + idx_25) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_12) * 8, _parity_gemm_arrived_12);
                                                int _smem_desc_lo_667 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_25) % 4) * 1024;
                                                int _smem_desc_lo_668 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_25) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_667), "r"(_smem_desc_lo_668), "r"(tmem_accumulator), "r"(((idx_25 == 0) ? 0 : 1)) : "memory");
                                            }
                                            int _stage_output_finished_9 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                            int _parity_output_finished_9 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                            mbarrier_wait(output_finished_addr + (_stage_output_finished_9) * 8, _parity_output_finished_9);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_26 = 0; idx_26 < head_4; idx_26++) {
                                                int _smem_desc_lo_669 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_26) % 4) * 1024;
                                                int _smem_desc_lo_670 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_26) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_669), "r"(_smem_desc_lo_670), "r"((tmem_accumulator + (256))), "r"(((idx_26 == 0) ? 0 : 1)) : "memory");
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_26) % 4)) * 8, (uint16_t)(3));
                                            }
                                            #pragma unroll 1
                                            for (int idx_27 = head_4; idx_27 < iterations_4 - tail_5; idx_27++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_27) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_13 = (int)((unsigned int)(gemm_wide + idx_27) % 4);
                                                int _parity_gemm_arrived_13 = (int)((unsigned int)(gemm_wide + idx_27) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_13) * 8, _parity_gemm_arrived_13);
                                                int _smem_desc_lo_671 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_27) % 4) * 1024;
                                                int _smem_desc_lo_672 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_27) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_671), "r"(_smem_desc_lo_672), "r"(tmem_accumulator), "r"(((idx_27 == 0) ? 0 : 1)) : "memory");
                                                int _smem_desc_lo_673 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_27) % 4) * 1024;
                                                int _smem_desc_lo_674 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_27) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_673), "r"(_smem_desc_lo_674), "r"((tmem_accumulator + (256))), "r"(((idx_27 == 0) ? 0 : 1)) : "memory");
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_27) % 4)) * 8, (uint16_t)(3));
                                            }
                                            #pragma unroll 1
                                            for (int idx_28 = iterations_4 - tail_5; idx_28 < iterations_4; idx_28++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_28) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_14 = (int)((unsigned int)(gemm_wide + idx_28) % 4);
                                                int _parity_gemm_arrived_14 = (int)((unsigned int)(gemm_wide + idx_28) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_14) * 8, _parity_gemm_arrived_14);
                                                int _smem_desc_lo_675 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_28) % 4) * 1024;
                                                int _smem_desc_lo_676 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_28) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_675), "r"(_smem_desc_lo_676), "r"(tmem_accumulator), "r"(((idx_28 == 0) ? 0 : 1)) : "memory");
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            #pragma unroll 1
                                            for (int idx_29 = iterations_4 - tail_5; idx_29 < iterations_4; idx_29++) {
                                                int _smem_desc_lo_677 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_29) % 4) * 1024;
                                                int _smem_desc_lo_678 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_29) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_677), "r"(_smem_desc_lo_678), "r"((tmem_accumulator + (256))), "r"(((idx_29 == 0) ? 0 : 1)) : "memory");
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_29) % 4)) * 8, (uint16_t)(3));
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    int _parity_output_arrived_8 = (int)((unsigned int)gemm_out0 % 2);
                                    mbarrier_wait(output_arrived_addr, _parity_output_arrived_8);
                                    int warp_row = tid / 32 * 32;
                                    #pragma unroll 1
                                    for (int chunk_16 = 0; chunk_16 < 16; chunk_16++) {
                                        float _tmem_load_18[16];
                                        tmem_ld_x16(&_tmem_load_18[0], taddr_1 + (unsigned int)(warp_row << 16) + (unsigned int)(chunk_16 * 16));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        #pragma unroll
                                        for (int vec_4 = 0; vec_4 < 4; vec_4++) {
                                            unsigned int address_23 = d_smem_addr + (unsigned int)(chunk_16 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_4 * 16);
                                            address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_23 - flat_view_addr)), "f"(_tmem_load_18[vec_4 * 4]), "f"(_tmem_load_18[vec_4 * 4 + 1]), "f"(_tmem_load_18[vec_4 * 4 + 2]), "f"(_tmem_load_18[vec_4 * 4 + 3]) : "memory");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                            #error "TmaReduceAdd5d requires SM90 or newer"
                                            #endif
                                            asm volatile(
                                                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dwd_r)), "r"(0), "r"(x_34 * 256 + cta_rank_0 * 128), "r"(y_4 * 2 * 16 + chunk_16), "r"(expert_4), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_16 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                    }
                                    int _parity_output_arrived_9 = (int)((unsigned int)gemm_out1 % 2);
                                    mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_9);
                                    if (has_hi_4 != 0) {
                                        #pragma unroll 1
                                        for (int chunk_17 = 0; chunk_17 < 16; chunk_17++) {
                                            float _tmem_load_19[16];
                                            tmem_ld_x16(&_tmem_load_19[0], taddr_1 + (unsigned int)(warp_row << 16) + 256 + (unsigned int)(chunk_17 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_5 = 0; vec_5 < 4; vec_5++) {
                                                unsigned int address_24 = d_smem_addr + (unsigned int)((16 + chunk_17) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_5 * 16);
                                                address_24 = address_24 ^ (address_24 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_24 - flat_view_addr)), "f"(_tmem_load_19[vec_5 * 4]), "f"(_tmem_load_19[vec_5 * 4 + 1]), "f"(_tmem_load_19[vec_5 * 4 + 2]), "f"(_tmem_load_19[vec_5 * 4 + 3]) : "memory");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                #error "TmaReduceAdd5d requires SM90 or newer"
                                                #endif
                                                asm volatile(
                                                    "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwd_r)), "r"(0), "r"(x_34 * 256 + cta_rank_0 * 128), "r"((y_4 * 2 + 1) * 16 + chunk_17), "r"(expert_4), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_17) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                    }
                                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                                    if (tid / 32 == 0) {
                                        if (warp == 0) {
                                            if (elect_sync()) {
                                                bool enabled_value_18 = macros > 1;
                                                if (enabled_value_18 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_8_1 = 1;
                                                    if (enabled_value_8_1 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                                if (has_hi_4 != 0) {
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            int live_4 = (int)(expert_4 >= 0);
                            gemm_wide += iterations_4 * live_4;
                            gemm_out0 += live_4;
                            gemm_out1 += live_4;
                            gemm_acc_tiles += 2 * live_4;
                        } else {
                            if (tile_word_0 == 4) {
                                int col_blocks_8 = (macro_size + 512 - 1) / 512;
                                {
                                    col_blocks_8 = (hidden + 512 - 1) / 512;
                                }
                                int x_35 = -1;
                                int y_5 = -1;
                                int expert_5 = -1;
                                int k_start_5 = 0;
                                int k_end_5 = 0;
                                int first_5 = 0;
                                int row_blocks_1 = intermediate / 256;
                                int expert_idx_1 = 0;
                                int local_task_1 = tile_word_1;
                                uint32_t fast_div_q_11 = (shared_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), shared_wgrad_div.multiplier) >> shared_wgrad_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                                uint32_t fast_div_r_11 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_11 * (uint32_t)(shared_wgrad_div.divisor);
                                expert_idx_1 = (int)fast_div_q_11;
                                local_task_1 = (int)fast_div_r_11;
                                int offset_15 = counts[experts + expert_idx_1];
                                int real_1 = counts[2 * experts + expert_idx_1];
                                real_1 = (real_1 + 64 - 1) / 64 * 64;
                                int _max_679 = ((offset_15) > (tile_word_2 * macro_size) ? (offset_15) : (tile_word_2 * macro_size));
                                k_start_5 = _max_679;
                                int _min_681 = (((tile_word_2 + 1) * macro_size) < (tokens) ? ((tile_word_2 + 1) * macro_size) : (tokens));
                                int _min_680 = ((offset_15 + real_1) < (_min_681) ? (offset_15 + real_1) : (_min_681));
                                k_end_5 = _min_680;
                                first_5 = (int)(k_start_5 == offset_15);
                                if (k_start_5 < k_end_5 && local_task_1 < row_blocks_1 * col_blocks_8) {
                                    int supergroup_5 = local_task_1 / (row_blocks_1 * 8);
                                    int full_cols_5 = col_blocks_8 / 8 * 8;
                                    int row_23 = 0;
                                    int col_18 = 0;
                                    if (local_task_1 < row_blocks_1 * full_cols_5) {
                                        row_23 = local_task_1 % (row_blocks_1 * 8) / 8;
                                        col_18 = supergroup_5 * 8 + local_task_1 % 8;
                                    } else {
                                        row_23 = (local_task_1 - row_blocks_1 * full_cols_5) / (col_blocks_8 - full_cols_5);
                                        col_18 = full_cols_5 + (local_task_1 - row_blocks_1 * full_cols_5) % (col_blocks_8 - full_cols_5);
                                    }
                                    if ((supergroup_5 & 1) != 0) {
                                        row_23 = row_blocks_1 - row_23 - 1;
                                    }
                                    x_35 = row_23;
                                    y_5 = col_18;
                                    expert_5 = expert_idx_1;
                                }
                                int has_hi_5 = 0;
                                has_hi_5 = (int)((y_5 * 2 + 1) * 256 < hidden);
                                int global_mini_7 = tile_word_2 * (macro_size / mini_size);
                                int macro_rows_5 = tile_word_2 * (macro_size / 256);
                                int iterations_5 = intermediate / 64;
                                iterations_5 = (k_end_5 - k_start_5) / 64;
                                if (expert_5 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_19 = macros > 1;
                                        if (enabled_value_19 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            #pragma unroll 1
                                            for (int idx_30 = 0; idx_30 < iterations_5; idx_30++) {
                                                int token_row_1 = k_start_5 + idx_30 * 64;
                                                if (idx_30 == 0 || token_row_1 % 256 == 0) {
                                                    bool enabled_value_20 = 1;
                                                    if (enabled_value_20 != 0) {
                                                        int poll_14_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_14_first) : "l"(dg_ready + (shared_rows + token_row_1 / 256)) : "memory");
                                                        int poll_14 = poll_14_first;
                                                        while (poll_14 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_14_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_14_next) : "l"(dg_ready + (shared_rows + token_row_1 / 256)) : "memory");
                                                            poll_14 = poll_14_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_30 == 0 || token_row_1 % mini_size == 0) {
                                                    int input_mini_1 = token_row_1 / mini_size;
                                                    int input_rows_1 = ((mini_size) < (tokens - input_mini_1 * mini_size) ? (mini_size) : (tokens - input_mini_1 * mini_size));
                                                    int input_count_1 = (input_rows_1 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_21 = tile_word_2 > 0;
                                                    if (enabled_value_21 != 0) {
                                                        int poll_15_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_15_first) : "l"(replay_x + input_mini_1) : "memory");
                                                        int poll_15 = poll_15_first;
                                                        while (poll_15 < input_count_1) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_15_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_15_next) : "l"(replay_x + input_mini_1) : "memory");
                                                            poll_15 = poll_15_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                int _stage_gemm_finished_5 = (int)((unsigned int)(gemm_wide + idx_30) % 4);
                                                int _parity_gemm_finished_5 = (int)((unsigned int)(gemm_wide + idx_30) / 4 % 2) ^ 1;
                                                mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_5) * 8, _parity_gemm_finished_5);
                                                int local_row_2 = k_start_5 + idx_30 * 64 - tile_word_2 * macro_size;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr + (unsigned int)((gemm_wide + idx_30) % 4 * 16384)), "l"((&dg_atb_r)), "r"(0), "r"(local_row_2), "r"(x_35 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_30) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_30) % 4 * 16384)), "l"((&x_atb_r)), "r"(0), "r"(local_row_2), "r"(y_5 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_30) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_30) % 4 * 16384)), "l"((&x_atb_r)), "r"(0), "r"(local_row_2), "r"((y_5 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_30) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                                        if (warp == 4) {
                                            if (elect_sync()) {
                                                int head_5 = ((3) < (iterations_5) ? (3) : (iterations_5));
                                                int tail_6 = 0;
                                                int _min_682 = ((3) < (iterations_5 - head_5) ? (3) : (iterations_5 - head_5));
                                                tail_6 = _min_682;
                                                int _stage_output_finished_10 = (int)((unsigned int)gemm_acc_tiles % 2);
                                                int _parity_output_finished_10 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                                mbarrier_wait(output_finished_addr + (_stage_output_finished_10) * 8, _parity_output_finished_10);
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_31 = 0; idx_31 < head_5; idx_31++) {
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_31) % 4)) * 8, 98304);
                                                    int _stage_gemm_arrived_15 = (int)((unsigned int)(gemm_wide + idx_31) % 4);
                                                    int _parity_gemm_arrived_15 = (int)((unsigned int)(gemm_wide + idx_31) / 4 % 2);
                                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_15) * 8, _parity_gemm_arrived_15);
                                                    int _smem_desc_lo_683 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_31) % 4) * 1024;
                                                    int _smem_desc_lo_684 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_31) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_683), "r"(_smem_desc_lo_684), "r"(tmem_accumulator), "r"(((idx_31 == 0) ? 0 : 1)) : "memory");
                                                }
                                                int _stage_output_finished_11 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                                int _parity_output_finished_11 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                                mbarrier_wait(output_finished_addr + (_stage_output_finished_11) * 8, _parity_output_finished_11);
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_32 = 0; idx_32 < head_5; idx_32++) {
                                                    int _smem_desc_lo_685 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_32) % 4) * 1024;
                                                    int _smem_desc_lo_686 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_32) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_685), "r"(_smem_desc_lo_686), "r"((tmem_accumulator + (256))), "r"(((idx_32 == 0) ? 0 : 1)) : "memory");
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_32) % 4)) * 8, (uint16_t)(3));
                                                }
                                                #pragma unroll 1
                                                for (int idx_33 = head_5; idx_33 < iterations_5 - tail_6; idx_33++) {
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_33) % 4)) * 8, 98304);
                                                    int _stage_gemm_arrived_16 = (int)((unsigned int)(gemm_wide + idx_33) % 4);
                                                    int _parity_gemm_arrived_16 = (int)((unsigned int)(gemm_wide + idx_33) / 4 % 2);
                                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_16) * 8, _parity_gemm_arrived_16);
                                                    int _smem_desc_lo_687 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_33) % 4) * 1024;
                                                    int _smem_desc_lo_688 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_33) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_687), "r"(_smem_desc_lo_688), "r"(tmem_accumulator), "r"(((idx_33 == 0) ? 0 : 1)) : "memory");
                                                    int _smem_desc_lo_689 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_33) % 4) * 1024;
                                                    int _smem_desc_lo_690 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_33) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_689), "r"(_smem_desc_lo_690), "r"((tmem_accumulator + (256))), "r"(((idx_33 == 0) ? 0 : 1)) : "memory");
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_33) % 4)) * 8, (uint16_t)(3));
                                                }
                                                #pragma unroll 1
                                                for (int idx_34 = iterations_5 - tail_6; idx_34 < iterations_5; idx_34++) {
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_34) % 4)) * 8, 98304);
                                                    int _stage_gemm_arrived_17 = (int)((unsigned int)(gemm_wide + idx_34) % 4);
                                                    int _parity_gemm_arrived_17 = (int)((unsigned int)(gemm_wide + idx_34) / 4 % 2);
                                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_17) * 8, _parity_gemm_arrived_17);
                                                    int _smem_desc_lo_691 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_34) % 4) * 1024;
                                                    int _smem_desc_lo_692 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_34) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_691), "r"(_smem_desc_lo_692), "r"(tmem_accumulator), "r"(((idx_34 == 0) ? 0 : 1)) : "memory");
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                                #pragma unroll 1
                                                for (int idx_35 = iterations_5 - tail_6; idx_35 < iterations_5; idx_35++) {
                                                    int _smem_desc_lo_693 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_35) % 4) * 1024;
                                                    int _smem_desc_lo_694 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_35) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_693), "r"(_smem_desc_lo_694), "r"((tmem_accumulator + (256))), "r"(((idx_35 == 0) ? 0 : 1)) : "memory");
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_35) % 4)) * 8, (uint16_t)(3));
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                            }
                                        }
                                    } else if (tid < 128) {
                                        int _parity_output_arrived_10 = (int)((unsigned int)gemm_out0 % 2);
                                        mbarrier_wait(output_arrived_addr, _parity_output_arrived_10);
                                        int warp_row_1 = tid / 32 * 32;
                                        #pragma unroll 1
                                        for (int chunk_18 = 0; chunk_18 < 16; chunk_18++) {
                                            float tmem_load_3[16];
                                            tmem_ld_x16(&tmem_load_3[0], taddr_1 + (unsigned int)(warp_row_1 << 16) + (unsigned int)(chunk_18 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_6 = 0; vec_6 < 4; vec_6++) {
                                                unsigned int address_25 = d_smem_addr + (unsigned int)(chunk_18 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_6 * 16);
                                                address_25 = address_25 ^ (address_25 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_25 - flat_view_addr)), "f"(tmem_load_3[vec_6 * 4]), "f"(tmem_load_3[vec_6 * 4 + 1]), "f"(tmem_load_3[vec_6 * 4 + 2]), "f"(tmem_load_3[vec_6 * 4 + 3]) : "memory");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                #error "TmaReduceAdd5d requires SM90 or newer"
                                                #endif
                                                asm volatile(
                                                    "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwg_r)), "r"(0), "r"(x_35 * 256 + cta_rank_0 * 128), "r"(y_5 * 2 * 16 + chunk_18), "r"(expert_5), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_18 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                        }
                                        int _parity_output_arrived_11 = (int)((unsigned int)gemm_out1 % 2);
                                        mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_11);
                                        if (has_hi_5 != 0) {
                                            #pragma unroll 1
                                            for (int chunk_19 = 0; chunk_19 < 16; chunk_19++) {
                                                float tmem_load_1_2[16];
                                                tmem_ld_x16(&tmem_load_1_2[0], taddr_1 + (unsigned int)(warp_row_1 << 16) + 256 + (unsigned int)(chunk_19 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_7 = 0; vec_7 < 4; vec_7++) {
                                                    unsigned int address_26 = d_smem_addr + (unsigned int)((16 + chunk_19) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_7 * 16);
                                                    address_26 = address_26 ^ (address_26 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_26 - flat_view_addr)), "f"(tmem_load_1_2[vec_7 * 4]), "f"(tmem_load_1_2[vec_7 * 4 + 1]), "f"(tmem_load_1_2[vec_7 * 4 + 2]), "f"(tmem_load_1_2[vec_7 * 4 + 3]) : "memory");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                    #error "TmaReduceAdd5d requires SM90 or newer"
                                                    #endif
                                                    asm volatile(
                                                        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                        :: "l"((&dwg_r)), "r"(0), "r"(x_35 * 256 + cta_rank_0 * 128), "r"((y_5 * 2 + 1) * 16 + chunk_19), "r"(expert_5), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_19) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    asm volatile("cp.async.bulk.commit_group;");
                                                }
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                                        if (tid / 32 == 0) {
                                            if (warp == 0) {
                                                if (elect_sync()) {
                                                    bool enabled_value_22 = macros > 1;
                                                    if (enabled_value_22 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_9_1 = 1;
                                                        if (enabled_value_9_1 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                    if (has_hi_5 != 0) {
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                int live_5 = (int)(expert_5 >= 0);
                                gemm_wide += iterations_5 * live_5;
                                gemm_out0 += live_5;
                                gemm_out1 += live_5;
                                gemm_acc_tiles += 2 * live_5;
                            } else if (tile_word_0 == 5) {
                                int col_blocks_9 = (macro_size + 512 - 1) / 512;
                                {
                                    col_blocks_9 = (hidden + 512 - 1) / 512;
                                }
                                int x_36 = -1;
                                int y_6 = -1;
                                int expert_6 = -1;
                                int k_start_6 = 0;
                                int k_end_6 = 0;
                                int first_6 = 0;
                                int row_blocks_2 = intermediate / 256;
                                int expert_idx_2 = 0;
                                int local_task_2 = tile_word_1;
                                uint32_t fast_div_q_12 = (shared_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), shared_wgrad_div.multiplier) >> shared_wgrad_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                                uint32_t fast_div_r_12 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_12 * (uint32_t)(shared_wgrad_div.divisor);
                                expert_idx_2 = (int)fast_div_q_12;
                                local_task_2 = (int)fast_div_r_12;
                                int offset_16 = counts[experts + expert_idx_2];
                                int real_2 = counts[2 * experts + expert_idx_2];
                                real_2 = (real_2 + 64 - 1) / 64 * 64;
                                int _max_695 = ((offset_16) > (tile_word_2 * macro_size) ? (offset_16) : (tile_word_2 * macro_size));
                                k_start_6 = _max_695;
                                int _min_697 = (((tile_word_2 + 1) * macro_size) < (tokens) ? ((tile_word_2 + 1) * macro_size) : (tokens));
                                int _min_696 = ((offset_16 + real_2) < (_min_697) ? (offset_16 + real_2) : (_min_697));
                                k_end_6 = _min_696;
                                first_6 = (int)(k_start_6 == offset_16);
                                if (k_start_6 < k_end_6 && local_task_2 < row_blocks_2 * col_blocks_9) {
                                    int supergroup_6 = local_task_2 / (row_blocks_2 * 8);
                                    int full_cols_6 = col_blocks_9 / 8 * 8;
                                    int row_24 = 0;
                                    int col_19 = 0;
                                    if (local_task_2 < row_blocks_2 * full_cols_6) {
                                        row_24 = local_task_2 % (row_blocks_2 * 8) / 8;
                                        col_19 = supergroup_6 * 8 + local_task_2 % 8;
                                    } else {
                                        row_24 = (local_task_2 - row_blocks_2 * full_cols_6) / (col_blocks_9 - full_cols_6);
                                        col_19 = full_cols_6 + (local_task_2 - row_blocks_2 * full_cols_6) % (col_blocks_9 - full_cols_6);
                                    }
                                    if ((supergroup_6 & 1) != 0) {
                                        row_24 = row_blocks_2 - row_24 - 1;
                                    }
                                    x_36 = row_24;
                                    y_6 = col_19;
                                    expert_6 = expert_idx_2;
                                }
                                int has_hi_6 = 0;
                                has_hi_6 = (int)((y_6 * 2 + 1) * 256 < hidden);
                                int global_mini_8 = tile_word_2 * (macro_size / mini_size);
                                int macro_rows_6 = tile_word_2 * (macro_size / 256);
                                int iterations_6 = intermediate / 64;
                                iterations_6 = (k_end_6 - k_start_6) / 64;
                                if (expert_6 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_23 = macros > 1;
                                        if (enabled_value_23 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            #pragma unroll 1
                                            for (int idx_36 = 0; idx_36 < iterations_6; idx_36++) {
                                                int token_row_2 = k_start_6 + idx_36 * 64;
                                                if (idx_36 == 0 || token_row_2 % 256 == 0) {
                                                    bool enabled_value_24 = 1;
                                                    if (enabled_value_24 != 0) {
                                                        int poll_16_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_16_first) : "l"(dg_ready + (shared_rows + token_row_2 / 256)) : "memory");
                                                        int poll_16 = poll_16_first;
                                                        while (poll_16 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_16_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_16_next) : "l"(dg_ready + (shared_rows + token_row_2 / 256)) : "memory");
                                                            poll_16 = poll_16_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_36 == 0 || token_row_2 % mini_size == 0) {
                                                    int input_mini_2 = token_row_2 / mini_size;
                                                    int input_rows_2 = ((mini_size) < (tokens - input_mini_2 * mini_size) ? (mini_size) : (tokens - input_mini_2 * mini_size));
                                                    int input_count_2 = (input_rows_2 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_25 = tile_word_2 > 0;
                                                    if (enabled_value_25 != 0) {
                                                        int poll_17_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_17_first) : "l"(replay_x + input_mini_2) : "memory");
                                                        int poll_17 = poll_17_first;
                                                        while (poll_17 < input_count_2) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_17_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_17_next) : "l"(replay_x + input_mini_2) : "memory");
                                                            poll_17 = poll_17_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                int _stage_gemm_finished_6 = (int)((unsigned int)(gemm_wide + idx_36) % 4);
                                                int _parity_gemm_finished_6 = (int)((unsigned int)(gemm_wide + idx_36) / 4 % 2) ^ 1;
                                                mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_6) * 8, _parity_gemm_finished_6);
                                                int local_row_3 = k_start_6 + idx_36 * 64 - tile_word_2 * macro_size;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr + (unsigned int)((gemm_wide + idx_36) % 4 * 16384)), "l"((&du_atb_r)), "r"(0), "r"(local_row_3), "r"(x_36 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_36) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_36) % 4 * 16384)), "l"((&x_atb_r)), "r"(0), "r"(local_row_3), "r"(y_6 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_36) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_36) % 4 * 16384)), "l"((&x_atb_r)), "r"(0), "r"(local_row_3), "r"((y_6 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_36) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                                        if (warp == 4) {
                                            if (elect_sync()) {
                                                int head_6 = ((3) < (iterations_6) ? (3) : (iterations_6));
                                                int tail_7 = 0;
                                                int _min_698 = ((3) < (iterations_6 - head_6) ? (3) : (iterations_6 - head_6));
                                                tail_7 = _min_698;
                                                int _stage_output_finished_12 = (int)((unsigned int)gemm_acc_tiles % 2);
                                                int _parity_output_finished_12 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                                mbarrier_wait(output_finished_addr + (_stage_output_finished_12) * 8, _parity_output_finished_12);
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_37 = 0; idx_37 < head_6; idx_37++) {
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_37) % 4)) * 8, 98304);
                                                    int _stage_gemm_arrived_18 = (int)((unsigned int)(gemm_wide + idx_37) % 4);
                                                    int _parity_gemm_arrived_18 = (int)((unsigned int)(gemm_wide + idx_37) / 4 % 2);
                                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_18) * 8, _parity_gemm_arrived_18);
                                                    int _smem_desc_lo_699 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_37) % 4) * 1024;
                                                    int _smem_desc_lo_700 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_37) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_699), "r"(_smem_desc_lo_700), "r"(tmem_accumulator), "r"(((idx_37 == 0) ? 0 : 1)) : "memory");
                                                }
                                                int _stage_output_finished_13 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                                int _parity_output_finished_13 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                                mbarrier_wait(output_finished_addr + (_stage_output_finished_13) * 8, _parity_output_finished_13);
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_38 = 0; idx_38 < head_6; idx_38++) {
                                                    int _smem_desc_lo_701 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_38) % 4) * 1024;
                                                    int _smem_desc_lo_702 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_38) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_701), "r"(_smem_desc_lo_702), "r"((tmem_accumulator + (256))), "r"(((idx_38 == 0) ? 0 : 1)) : "memory");
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_38) % 4)) * 8, (uint16_t)(3));
                                                }
                                                #pragma unroll 1
                                                for (int idx_39 = head_6; idx_39 < iterations_6 - tail_7; idx_39++) {
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_39) % 4)) * 8, 98304);
                                                    int _stage_gemm_arrived_19 = (int)((unsigned int)(gemm_wide + idx_39) % 4);
                                                    int _parity_gemm_arrived_19 = (int)((unsigned int)(gemm_wide + idx_39) / 4 % 2);
                                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_19) * 8, _parity_gemm_arrived_19);
                                                    int _smem_desc_lo_703 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_39) % 4) * 1024;
                                                    int _smem_desc_lo_704 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_39) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_703), "r"(_smem_desc_lo_704), "r"(tmem_accumulator), "r"(((idx_39 == 0) ? 0 : 1)) : "memory");
                                                    int _smem_desc_lo_705 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_39) % 4) * 1024;
                                                    int _smem_desc_lo_706 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_39) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_705), "r"(_smem_desc_lo_706), "r"((tmem_accumulator + (256))), "r"(((idx_39 == 0) ? 0 : 1)) : "memory");
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_39) % 4)) * 8, (uint16_t)(3));
                                                }
                                                #pragma unroll 1
                                                for (int idx_40 = iterations_6 - tail_7; idx_40 < iterations_6; idx_40++) {
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_40) % 4)) * 8, 98304);
                                                    int _stage_gemm_arrived_20 = (int)((unsigned int)(gemm_wide + idx_40) % 4);
                                                    int _parity_gemm_arrived_20 = (int)((unsigned int)(gemm_wide + idx_40) / 4 % 2);
                                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_20) * 8, _parity_gemm_arrived_20);
                                                    int _smem_desc_lo_707 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_40) % 4) * 1024;
                                                    int _smem_desc_lo_708 = ((((mn_view_addr_0) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_40) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_707), "r"(_smem_desc_lo_708), "r"(tmem_accumulator), "r"(((idx_40 == 0) ? 0 : 1)) : "memory");
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                                #pragma unroll 1
                                                for (int idx_41 = iterations_6 - tail_7; idx_41 < iterations_6; idx_41++) {
                                                    int _smem_desc_lo_709 = ((((mn_view_addr) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_41) % 4) * 1024;
                                                    int _smem_desc_lo_710 = ((((mn_view_addr_1) >> 4) & 0x3FFF) | 0x2000000) + ((gemm_wide + idx_41) % 4) * 1024;
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
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_709), "r"(_smem_desc_lo_710), "r"((tmem_accumulator + (256))), "r"(((idx_41 == 0) ? 0 : 1)) : "memory");
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_41) % 4)) * 8, (uint16_t)(3));
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                            }
                                        }
                                    } else if (tid < 128) {
                                        int _parity_output_arrived_12 = (int)((unsigned int)gemm_out0 % 2);
                                        mbarrier_wait(output_arrived_addr, _parity_output_arrived_12);
                                        int warp_row_2 = tid / 32 * 32;
                                        #pragma unroll 1
                                        for (int chunk_20 = 0; chunk_20 < 16; chunk_20++) {
                                            float tmem_load_4[16];
                                            tmem_ld_x16(&tmem_load_4[0], taddr_1 + (unsigned int)(warp_row_2 << 16) + (unsigned int)(chunk_20 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_8 = 0; vec_8 < 4; vec_8++) {
                                                unsigned int address_27 = d_smem_addr + (unsigned int)(chunk_20 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_8 * 16);
                                                address_27 = address_27 ^ (address_27 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_27 - flat_view_addr)), "f"(tmem_load_4[vec_8 * 4]), "f"(tmem_load_4[vec_8 * 4 + 1]), "f"(tmem_load_4[vec_8 * 4 + 2]), "f"(tmem_load_4[vec_8 * 4 + 3]) : "memory");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                #error "TmaReduceAdd5d requires SM90 or newer"
                                                #endif
                                                asm volatile(
                                                    "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwu_r)), "r"(0), "r"(x_36 * 256 + cta_rank_0 * 128), "r"(y_6 * 2 * 16 + chunk_20), "r"(expert_6), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_20 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                        }
                                        int _parity_output_arrived_13 = (int)((unsigned int)gemm_out1 % 2);
                                        mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_13);
                                        if (has_hi_6 != 0) {
                                            #pragma unroll 1
                                            for (int chunk_21 = 0; chunk_21 < 16; chunk_21++) {
                                                float tmem_load_1_3[16];
                                                tmem_ld_x16(&tmem_load_1_3[0], taddr_1 + (unsigned int)(warp_row_2 << 16) + 256 + (unsigned int)(chunk_21 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_9 = 0; vec_9 < 4; vec_9++) {
                                                    unsigned int address_28 = d_smem_addr + (unsigned int)((16 + chunk_21) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_9 * 16);
                                                    address_28 = address_28 ^ (address_28 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_28 - flat_view_addr)), "f"(tmem_load_1_3[vec_9 * 4]), "f"(tmem_load_1_3[vec_9 * 4 + 1]), "f"(tmem_load_1_3[vec_9 * 4 + 2]), "f"(tmem_load_1_3[vec_9 * 4 + 3]) : "memory");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                    #error "TmaReduceAdd5d requires SM90 or newer"
                                                    #endif
                                                    asm volatile(
                                                        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                        :: "l"((&dwu_r)), "r"(0), "r"(x_36 * 256 + cta_rank_0 * 128), "r"((y_6 * 2 + 1) * 16 + chunk_21), "r"(expert_6), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_21) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    asm volatile("cp.async.bulk.commit_group;");
                                                }
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                                        if (tid / 32 == 0) {
                                            if (warp == 0) {
                                                if (elect_sync()) {
                                                    bool enabled_value_26 = macros > 1;
                                                    if (enabled_value_26 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_10_1 = 1;
                                                        if (enabled_value_10_1 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                    if (has_hi_6 != 0) {
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                int live_6 = (int)(expert_6 >= 0);
                                gemm_wide += iterations_6 * live_6;
                                gemm_out0 += live_6;
                                gemm_out1 += live_6;
                                gemm_acc_tiles += 2 * live_6;
                            }
                        }
                    }
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
            int kind_2 = -1;
            int task_0 = 0;
            int macro_0 = 0;
            int mini_0 = 0;
            int shared_0 = 0;
            if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
                if (shared_tasks > cluster - comm_clusters) {
                    shared_0 = 1;
                    if (shared_down > cluster - comm_clusters) {
                        kind_2 = 0;
                        task_0 = cluster - comm_clusters;
                    } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                        kind_2 = 1;
                        task_0 = cluster - comm_clusters - shared_down;
                    } else {
                        if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                            kind_2 = 2;
                            task_0 = cluster - comm_clusters - shared_down - shared_swiglu;
                        } else {
                            int weight_task_2 = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                            uint32_t fast_div_q_13 = (shared_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)weight_task_2), shared_wgrad_div.multiplier) >> shared_wgrad_div.shift_right) : (uint32_t)((unsigned int)weight_task_2);
                            uint32_t fast_div_r_13 = (uint32_t)((unsigned int)weight_task_2) - fast_div_q_13 * (uint32_t)(shared_wgrad_div.divisor);
                            kind_2 = (int)fast_div_q_13 + 3;
                            task_0 = (int)fast_div_r_13;
                        }
                    }
                } else {
                    int routed_1 = cluster - comm_clusters - shared_tasks;
                    int macro_task_1 = routed_1;
                    int replay_tasks_1 = 0;
                    if (routed_1 >= saved_tasks) {
                        uint32_t fast_div_q_1_1 = (replay_macro_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(routed_1 - saved_tasks)), replay_macro_tasks_div.multiplier) >> replay_macro_tasks_div.shift_right) : (uint32_t)((unsigned int)(routed_1 - saved_tasks));
                        uint32_t fast_div_r_1_1 = (uint32_t)((unsigned int)(routed_1 - saved_tasks)) - fast_div_q_1_1 * (uint32_t)(replay_macro_tasks_div.divisor);
                        macro_0 = (int)fast_div_q_1_1 + 1;
                        macro_task_1 = (int)fast_div_r_1_1;
                    }
                    int _min_711 = ((tokens - macro_0 * macro_size) < (macro_size) ? (tokens - macro_0 * macro_size) : (macro_size));
                    uint32_t fast_div_q_2_1 = (mini_size_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(_min_711 + mini_size - 1)), mini_size_div.multiplier) >> mini_size_div.shift_right) : (uint32_t)((unsigned int)(_min_711 + mini_size - 1));
                    uint32_t fast_div_r_2_1 = (uint32_t)((unsigned int)(_min_711 + mini_size - 1)) - fast_div_q_2_1 * (uint32_t)(mini_size_div.divisor);
                    int macro_minis_1 = (int)fast_div_q_2_1;
                    if (routed_1 >= saved_tasks) {
                        replay_tasks_1 = macro_minis_1 * mini_replay;
                    }
                    if (macro_task_1 < replay_tasks_1) {
                        uint32_t fast_div_q_3_1 = (mini_replay_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)macro_task_1), mini_replay_div.multiplier) >> mini_replay_div.shift_right) : (uint32_t)((unsigned int)macro_task_1);
                        uint32_t fast_div_r_3_1 = (uint32_t)((unsigned int)macro_task_1) - fast_div_q_3_1 * (uint32_t)(mini_replay_div.divisor);
                        mini_0 = (int)fast_div_q_3_1;
                        int mini_task_1 = (int)fast_div_r_3_1;
                        if (mini_task_1 < mini_down) {
                            kind_2 = 6;
                            task_0 = mini_task_1;
                        } else if (mini_task_1 < 2 * mini_down) {
                            kind_2 = 7;
                            task_0 = mini_task_1 - mini_down;
                        } else {
                            kind_2 = 8;
                            task_0 = mini_task_1 - 2 * mini_down;
                        }
                    } else {
                        int bwd_task_1 = macro_task_1 - replay_tasks_1;
                        if (bwd_task_1 < macro_minis_1 * mini_bwd) {
                            int fill_1 = ((macro_minis_1 - 1) < (1) ? (macro_minis_1 - 1) : (1));
                            int slot_down_1 = fill_1 * mini_down;
                            int end0_1 = mini_down;
                            int end1_1 = end0_1 + slot_down_1 + mini_swiglu;
                            int _max_712 = ((macro_minis_1 - 2) > (0) ? (macro_minis_1 - 2) : (0));
                            int end2_1 = end1_1 + _max_712 * mini_bwd;
                            int end3_1 = end2_1 + fill_1 * (mini_bwd - mini_down);
                            if (bwd_task_1 < end0_1) {
                                kind_2 = 0;
                                mini_0 = 0;
                                task_0 = bwd_task_1;
                            } else if (bwd_task_1 < end1_1) {
                                int fill_task_1 = bwd_task_1 - end0_1;
                                if (fill_task_1 < slot_down_1) {
                                    kind_2 = 0;
                                    mini_0 = 1;
                                    task_0 = fill_task_1;
                                } else {
                                    kind_2 = 1;
                                    mini_0 = 0;
                                    task_0 = fill_task_1 - slot_down_1;
                                }
                            } else {
                                if (bwd_task_1 < end2_1) {
                                    uint32_t fast_div_q_4_1 = (mini_bwd_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(bwd_task_1 - end1_1)), mini_bwd_div.multiplier) >> mini_bwd_div.shift_right) : (uint32_t)((unsigned int)(bwd_task_1 - end1_1));
                                    uint32_t fast_div_r_4_1 = (uint32_t)((unsigned int)(bwd_task_1 - end1_1)) - fast_div_q_4_1 * (uint32_t)(mini_bwd_div.divisor);
                                    int slot_5 = (int)fast_div_q_4_1 + 2;
                                    int slot_task_1 = (int)fast_div_r_4_1;
                                    if (slot_task_1 < mini_down) {
                                        kind_2 = 0;
                                        mini_0 = slot_5;
                                        task_0 = slot_task_1;
                                    } else if (slot_task_1 < mini_down + mini_swiglu) {
                                        kind_2 = 1;
                                        mini_0 = slot_5 - 1;
                                        task_0 = slot_task_1 - mini_down;
                                    } else {
                                        kind_2 = 2;
                                        mini_0 = slot_5 - 2;
                                        task_0 = slot_task_1 - mini_down - mini_swiglu;
                                    }
                                } else if (bwd_task_1 < end3_1) {
                                    int drain_task_1 = bwd_task_1 - end2_1;
                                    if (drain_task_1 < mini_swiglu) {
                                        kind_2 = 1;
                                        mini_0 = macro_minis_1 - 1;
                                        task_0 = drain_task_1;
                                    } else {
                                        kind_2 = 2;
                                        mini_0 = macro_minis_1 - 2;
                                        task_0 = drain_task_1 - mini_swiglu;
                                    }
                                } else {
                                    kind_2 = 2;
                                    mini_0 = macro_minis_1 - 1;
                                    task_0 = bwd_task_1 - end3_1;
                                }
                            }
                        } else {
                            int weight_task_3 = bwd_task_1 - macro_minis_1 * mini_bwd;
                            uint32_t fast_div_q_5_1 = (weight_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)weight_task_3), weight_tasks_div.multiplier) >> weight_tasks_div.shift_right) : (uint32_t)((unsigned int)weight_task_3);
                            uint32_t fast_div_r_5_1 = (uint32_t)((unsigned int)weight_task_3) - fast_div_q_5_1 * (uint32_t)(weight_tasks_div.divisor);
                            kind_2 = (int)fast_div_q_5_1 + 3;
                            task_0 = (int)fast_div_r_5_1;
                        }
                    }
                }
            }
            if ((tile_word_0 == 1 || tile_word_0 == 8) && cluster >= 0 && kind_2 != 1 && kind_2 != 8) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            }
            tile_word_0 = kind_2;
            tile_word_1 = task_0;
            tile_word_2 = macro_0;
            tile_word_3 = mini_0;
            tile_word_4 = shared_0;
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
