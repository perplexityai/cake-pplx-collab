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

#define IS_CAUSAL 1
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 64
#define MMA_K 16
#define TS_TMEM_STEP 8
#define DQ_GROUP_SIZE 16384
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



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}



__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}



__device__ __forceinline__ void tcgen05_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
}






__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_train_1f0c2f01d6f6b42a6b60(float* __restrict__ dQ_accum, float* __restrict__ LSE_log2_gmem, float* __restrict__ dPsum_gmem, __nv_bfloat16* __restrict__ dK_gmem, __nv_bfloat16* __restrict__ dV_gmem, const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap dO, const __grid_constant__ CUtensorMap dK, const __grid_constant__ CUtensorMap dV, int seqlen_q, int seqlen_kv, float softmax_scale, float softmax_scale_log2, int num_m_blocks, int num_heads)
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
    #define q_full_addr (mbar_base + 0)
    #define lse_full_addr (mbar_base + 16)
    #define do_full_addr (mbar_base + 32)
    #define dpsum_full_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 48)
    #define dp_full_addr (mbar_base + 56)
    #define s_empty_addr (mbar_base + 64)
    #define p_full_addr (mbar_base + 72)
    #define p_empty_addr (mbar_base + 80)
    #define ds_ready_addr (mbar_base + 88)
    #define dq_full_addr (mbar_base + 96)
    #define dq_reduce_done_addr (mbar_base + 104)
    #define dkv_full_addr (mbar_base + 112)
    #define dkv_empty_addr (mbar_base + 128)
    #define dq_tail_done_addr (mbar_base + 144)
    #define q_empty_addr (mbar_base + 152)
    #define lse_empty_addr (mbar_base + 168)
    #define do_empty_addr (mbar_base + 184)
    #define dpsum_empty_addr (mbar_base + 192)
    #define ds_empty_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* q_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_buf_addr = smem + 1024;
    __nv_bfloat16* q_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_mn_addr = smem + 1024;
    __nv_bfloat16* dk_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int dk_staging_addr = smem + 1024;
    __nv_bfloat16* k_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_buf_addr = smem + 33792;
    __nv_bfloat16* k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_mn_addr = smem + 33792;
    __nv_bfloat16* v_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int v_buf_addr = smem + 50176;
    __nv_bfloat16* do_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int do_buf_addr = smem + 66560;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int do_mn_addr = smem + 66560;
    __nv_bfloat16* dv_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dv_staging_addr = smem + 66560;
    __nv_bfloat16* ds_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int ds_buf_addr = smem + 82944;
    float* dq_stage = reinterpret_cast<float*>(smem_raw + 115712);
    const int dq_stage_addr = smem + 115712;
    float* lse_buf = reinterpret_cast<float*>(smem_raw + 148480);
    const int lse_buf_addr = smem + 148480;
    float* dpsum_buf = reinterpret_cast<float*>(smem_raw + 149504);
    const int dpsum_buf_addr = smem + 149504;

    // Kernel setup ops
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dO))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dV))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dK))) : "memory"); }

    // Mbarrier init (26 barriers at smem_raw[0..208))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // lse_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // do_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // dpsum_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // dp_full: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // s_empty: 1 barriers, init_count=8
            mbarrier_init(smem + 64, 8);
            // p_full: 1 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            // p_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // ds_ready: 1 barriers, init_count=8
            mbarrier_init(smem + 88, 8);
            // dq_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // dq_reduce_done: 1 barriers, init_count=4
            mbarrier_init(smem + 104, 4);
            // --- pipeline 'dkv_pipe' ---
            // dkv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // dkv_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // dq_tail_done: 1 barriers, init_count=4
            mbarrier_init(smem + 144, 4);
            // --- pipeline 'q_pipe' ---
            // q_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 152, 2);
            mbarrier_init(smem + 160, 2);
            // lse_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 168, 8);
            mbarrier_init(smem + 176, 8);
            // do_empty: 1 barriers, init_count=2
            mbarrier_init(smem + 184, 2);
            // dpsum_empty: 1 barriers, init_count=8
            mbarrier_init(smem + 192, 8);
            // ds_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 12) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_scores = taddr;
    const unsigned int tmem_softmax_p = taddr + 384;
    const unsigned int tmem_dv_accum = taddr + 128;
    const unsigned int tmem_dp_dq = taddr + 192;
    const unsigned int tmem_ds_tmem = taddr + 448;
    const unsigned int tmem_dq_alias = taddr + 192;
    const unsigned int tmem_dk_accum = taddr + 320;

    // Kernel post-init ops
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    }

    // ---- Role: empty ----
    if (warp == 15) {
        // idle — no tasks assigned
    }
    // ---- Role: epi_idle ----
    if (warp == 14) {
        // idle — no tasks assigned
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            int n_block = blockIdx.x;
            int bh = blockIdx.z * num_heads + blockIdx.y;
            int m_start;
            {
                m_start = n_block;
            }
            int head_idx = blockIdx.y;
            int batch_idx = blockIdx.z;
            int off_kv = n_block * BLOCK_N;
            int stats_q_base = bh * seqlen_q;
            unsigned int q_load_stage = 0;
            int off_q = m_start * BLOCK_M;
            int stats_off_q = stats_q_base + off_q;
            unsigned int _phase_q_empty = 1;
            mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 32768);
            }
            if (elect_sync()) {
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_buf_addr), "l"((&K)), "r"(0), "r"(off_kv), "r"(head_idx), "r"(batch_idx),
                       "r"(q_full_addr + (q_load_stage) * 8), "l"(0ULL) : "memory");
            }
            if (elect_sync()) {
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(q_buf_addr + q_load_stage * 16384), "l"((&Q)), "r"(0), "r"(off_q), "r"(head_idx), "r"(batch_idx),
                       "r"(q_full_addr + (q_load_stage) * 8), "l"(0ULL) : "memory");
            }
            mbarrier_wait(lse_empty_addr + (q_load_stage) * 8, _phase_q_empty);
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(lse_full_addr + (q_load_stage) * 8, 512);
            }
            if (elect_sync()) {
                cp_async_bulk_gmem2smem(lse_buf_addr + q_load_stage * (unsigned int)BLOCK_M * 4, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(LSE_log2_gmem) + (unsigned long long)stats_off_q * (unsigned long long)4), BLOCK_M * 4, lse_full_addr + (q_load_stage) * 8);
            }
            q_load_stage += 1;
            if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
            unsigned int _phase_do_empty_0 = 1;
            mbarrier_wait(do_empty_addr, _phase_do_empty_0);
            _phase_do_empty_0 ^= 1;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(do_full_addr, 32768);
            }
            if (elect_sync()) {
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_buf_addr), "l"((&V)), "r"(0), "r"(off_kv), "r"(head_idx), "r"(batch_idx),
                       "r"(do_full_addr), "l"(0ULL) : "memory");
            }
            if (elect_sync()) {
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(do_buf_addr), "l"((&dO)), "r"(0), "r"(off_q), "r"(head_idx), "r"(batch_idx),
                       "r"(do_full_addr), "l"(0ULL) : "memory");
            }
            unsigned int _phase_dpsum_empty_0 = 1;
            mbarrier_wait(dpsum_empty_addr, _phase_dpsum_empty_0);
            _phase_dpsum_empty_0 ^= 1;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(dpsum_full_addr, 512);
            }
            if (elect_sync()) {
                cp_async_bulk_gmem2smem(dpsum_buf_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dPsum_gmem) + (unsigned long long)stats_off_q * (unsigned long long)4), BLOCK_M * 4, dpsum_full_addr);
            }
            #pragma unroll 1
            for (int m_iter = m_start + 1; m_iter < num_m_blocks; m_iter++) {
                int off_q_0 = m_iter * BLOCK_M;
                int stats_off_q_1 = stats_q_base + off_q_0;
                mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 16384);
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(q_buf_addr + q_load_stage * 16384), "l"((&Q)), "r"(0), "r"(off_q_0), "r"(head_idx), "r"(batch_idx),
                           "r"(q_full_addr + (q_load_stage) * 8), "l"(0ULL) : "memory");
                }
                mbarrier_wait(lse_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(lse_full_addr + (q_load_stage) * 8, 512);
                }
                if (elect_sync()) {
                    cp_async_bulk_gmem2smem(lse_buf_addr + q_load_stage * (unsigned int)BLOCK_M * 4, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(LSE_log2_gmem) + (unsigned long long)stats_off_q_1 * (unsigned long long)4), BLOCK_M * 4, lse_full_addr + (q_load_stage) * 8);
                }
                q_load_stage += 1;
                if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                mbarrier_wait(do_empty_addr, _phase_do_empty_0);
                _phase_do_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(do_full_addr, 16384);
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(do_buf_addr), "l"((&dO)), "r"(0), "r"(off_q_0), "r"(head_idx), "r"(batch_idx),
                           "r"(do_full_addr), "l"(0ULL) : "memory");
                }
                mbarrier_wait(dpsum_empty_addr, _phase_dpsum_empty_0);
                _phase_dpsum_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(dpsum_full_addr, 512);
                }
                if (elect_sync()) {
                    cp_async_bulk_gmem2smem(dpsum_buf_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dPsum_gmem) + (unsigned long long)stats_off_q_1 * (unsigned long long)4), BLOCK_M * 4, dpsum_full_addr);
                }
            }
            q_load_stage += 1;
            if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
            mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 16384);
            }
            mbarrier_wait(lse_empty_addr + (q_load_stage) * 8, _phase_q_empty);
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(lse_full_addr + (q_load_stage) * 8, 512);
            }
            mbarrier_wait(do_empty_addr, _phase_do_empty_0);
            _phase_do_empty_0 ^= 1;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(do_full_addr, 16384);
            }
            mbarrier_wait(dpsum_empty_addr, _phase_dpsum_empty_0);
            _phase_dpsum_empty_0 ^= 1;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(dpsum_full_addr, 512);
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int n_block_1 = blockIdx.x;
            int bh_1 = blockIdx.z * num_heads + blockIdx.y;
            int m_start_1;
            {
                m_start_1 = n_block_1;
            }
            unsigned int q_qk_stage = 0;
            unsigned int q_mma_stage = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_s_empty_0 = 1;
            unsigned int _phase_do_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_ds_ready_0 = 0;
            unsigned int _phase_dq_reduce_done_0 = 0;
            unsigned int _phase_dkv_empty_0 = 1;
            unsigned int _phase_dkv_empty_1 = 1;
            if (elect_sync()) {
                mbarrier_wait(q_full_addr + (q_qk_stage) * 8, _phase_q_full);
                mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                _phase_s_empty_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_0 = ((k_buf_addr) >> 4) & 0x7FFF;
                int _smem_desc_lo_1 = (((q_buf_addr) >> 4) & 0x7FFF) + (q_qk_stage) * 1024;
#else
                int _smem_desc_lo_0 = ((k_buf_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_1 = (((q_buf_addr) >> 4) & 0x3FFF) + (q_qk_stage) * 1024;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_scores), "r"(0) : "memory");
                tcgen05_commit2(s_full_addr, q_empty_addr + (q_qk_stage) * 8);
                q_qk_stage += 1;
                if (q_qk_stage == 2) { q_qk_stage = 0; _phase_q_full ^= 1; }
                mbarrier_wait(do_full_addr, _phase_do_full_0);
                _phase_do_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_2 = ((v_buf_addr) >> 4) & 0x7FFF;
                int _smem_desc_lo_3 = ((do_buf_addr) >> 4) & 0x7FFF;
#else
                int _smem_desc_lo_2 = ((v_buf_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_3 = ((do_buf_addr) >> 4) & 0x3FFF;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_dp_dq), "r"(0) : "memory");
                tcgen05_commit2(dp_full_addr, do_empty_addr);
                #pragma unroll 1
                for (int m_iter_1 = m_start_1 + 1; m_iter_1 < num_m_blocks; m_iter_1++) {
                    int first_dk = ((m_iter_1 == m_start_1 + 1) ? 1 : 0);
                    mbarrier_wait(q_full_addr + (q_qk_stage) * 8, _phase_q_full);
                    mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                    _phase_s_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = ((k_buf_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_5 = (((q_buf_addr) >> 4) & 0x7FFF) + (q_qk_stage) * 1024;
#else
                    int _smem_desc_lo_4 = ((k_buf_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_5 = (((q_buf_addr) >> 4) & 0x3FFF) + (q_qk_stage) * 1024;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_scores), "r"(0) : "memory");
                    tcgen05_commit2(s_full_addr, q_empty_addr + (q_qk_stage) * 8);
                    q_qk_stage += 1;
                    if (q_qk_stage == 2) { q_qk_stage = 0; _phase_q_full ^= 1; }
                    int first_dv = ((m_iter_1 == m_start_1 + 1) ? 1 : 0);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_6 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 135333008;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_dv_accum), "r"(_smem_desc_lo_6), "r"(tmem_softmax_p), "r"(((first_dv) ? 0 : 1)) : "memory");
                    tcgen05_commit(p_empty_addr);
                    tcgen05_commit(do_empty_addr);
                    mbarrier_wait(ds_ready_addr, _phase_ds_ready_0);
                    _phase_ds_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_7 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (q_mma_stage) * 1024;
#else
                    int _smem_desc_lo_7 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (q_mma_stage) * 1024;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 135333008;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_dk_accum), "r"(_smem_desc_lo_7), "r"(tmem_ds_tmem), "r"(((first_dk) ? 0 : 1)) : "memory");
                    tcgen05_commit(q_empty_addr + (q_mma_stage) * 8);
                    q_mma_stage += 1;
                    if (q_mma_stage == 2) { q_mma_stage = 0; }
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = (((ds_buf_addr) >> 4) & 0x7FFF) | 0x4000000;
                    int _smem_desc_lo_9 = (((k_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_8 = (((ds_buf_addr) >> 4) & 0x3FFF) | 0x4000000;
                    int _smem_desc_lo_9 = (((k_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tmem_dq_alias), "r"(0) : "memory");
                    tcgen05_commit(dq_full_addr);
                    tcgen05_commit(ds_empty_addr);
                    mbarrier_wait(do_full_addr, _phase_do_full_0);
                    _phase_do_full_0 ^= 1;
                    mbarrier_wait(dq_reduce_done_addr, _phase_dq_reduce_done_0);
                    _phase_dq_reduce_done_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = ((v_buf_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_11 = ((do_buf_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_10 = ((v_buf_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_11 = ((do_buf_addr) >> 4) & 0x3FFF;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_dp_dq), "r"(0) : "memory");
                    tcgen05_commit2(dp_full_addr, do_empty_addr);
                }
                int tail_first_dv = ((num_m_blocks == m_start_1 + 1) ? 1 : 0);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_12 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                int _smem_desc_lo_12 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 135333008;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_dv_accum), "r"(_smem_desc_lo_12), "r"(tmem_softmax_p), "r"(((tail_first_dv) ? 0 : 1)) : "memory");
                tcgen05_commit(p_empty_addr);
                tcgen05_commit(do_empty_addr);
                mbarrier_wait(dkv_empty_addr, _phase_dkv_empty_0);
                _phase_dkv_empty_0 ^= 1;
                tcgen05_commit(dkv_full_addr);
                mbarrier_wait(dkv_empty_addr + 8, _phase_dkv_empty_1);
                _phase_dkv_empty_1 ^= 1;
                mbarrier_wait(ds_ready_addr, _phase_ds_ready_0);
                _phase_ds_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int tail_first_dk = ((num_m_blocks == m_start_1 + 1) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_13 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (q_mma_stage) * 1024;
#else
                int _smem_desc_lo_13 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (q_mma_stage) * 1024;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 135333008;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_dk_accum), "r"(_smem_desc_lo_13), "r"(tmem_ds_tmem), "r"(((tail_first_dk) ? 0 : 1)) : "memory");
                tcgen05_commit(dkv_full_addr + 8);
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_14 = (((ds_buf_addr) >> 4) & 0x7FFF) | 0x4000000;
                int _smem_desc_lo_15 = (((k_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                int _smem_desc_lo_14 = (((ds_buf_addr) >> 4) & 0x3FFF) | 0x4000000;
                int _smem_desc_lo_15 = (((k_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"(tmem_dq_alias), "r"(0) : "memory");
                tcgen05_commit(dq_full_addr);
                tcgen05_commit(q_empty_addr + (q_mma_stage) * 8);
                tcgen05_commit(ds_empty_addr);
            }
        }
    }
    // ---- Role: compute ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
        { // compute_main
            int n_block_2 = blockIdx.x;
            int bh_2 = blockIdx.z * num_heads + blockIdx.y;
            int m_start_2;
            {
                m_start_2 = n_block_2;
            }
            int head_idx_1 = blockIdx.y;
            int batch_idx_1 = blockIdx.z;
            int off_q_base = bh_2 * seqlen_q;
            unsigned int q_compute_stage = 0;
            int wg_idx = 1 - (warp - 4) / 4;
            int m_blk = 0;
            unsigned int _phase_lse_full = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_dpsum_full_0 = 0;
            unsigned int _phase_dp_full_0 = 0;
            unsigned int _phase_ds_empty_0 = 1;
            #pragma unroll 1
            for (int m_iter_2 = m_start_2; m_iter_2 < num_m_blocks; m_iter_2++) {
                int off_q_1 = off_q_base + m_iter_2 * BLOCK_M;
                mbarrier_wait(lse_full_addr + (q_compute_stage) * 8, _phase_lse_full);
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)(wg_idx * 32) + (unsigned int)((warp - 4) % 4 * 32 << 16)));
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(taddr + (unsigned int)(wg_idx * 32) + 64 + (unsigned int)((warp - 4) % 4 * 32 << 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(s_empty_addr);
                }
                int c_pair = wg_idx * 2;
                if (m_iter_2 == n_block_2) {
                    int mask_count = (warp - 4) % 4 * 32 + lane - c_pair * 16;
                    uint32_t _thresh_mask_0;
                    {
                        int _lim_1 = mask_count;
                        if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                        else if (_lim_1 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_0) : "r"(_lim_1));
                        }
                    }
                    if (!(~_thresh_mask_0 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                    if (!(~_thresh_mask_0 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                }
                float lse[32];
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    lse[j] = lse_buf[q_compute_stage * (unsigned int)BLOCK_M + (unsigned int)(c_pair * 16) + (unsigned int)j];
                }
                float2 _f2_0 = make_float2(softmax_scale_log2, softmax_scale_log2);
                #pragma unroll
                for (int pair = 0; pair < 16; pair++) {
                    float2 affine_pair = fma_sub_f32x2_rn_ftz(make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1]), _f2_0, make_float2(lse[pair * 2], lse[pair * 2 + 1]));
                    _tmem_load_0[pair * 2] = affine_pair.x;
                    _tmem_load_0[pair * 2 + 1] = affine_pair.y;
                }
                unsigned int _tmem_load_0_bf16[16];
                softmax_frag_exp2_cast(&_tmem_load_0[0], _tmem_load_0_bf16, 0);
                int _parity_p_empty = (int)((unsigned int)m_blk % 2) ^ 1;
                mbarrier_wait(p_empty_addr, _parity_p_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int p_addr = taddr + 384 + (unsigned int)(c_pair * 8) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(p_addr), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]), "r"(_tmem_load_0_bf16[8]), "r"(_tmem_load_0_bf16[9]), "r"(_tmem_load_0_bf16[10]), "r"(_tmem_load_0_bf16[11]), "r"(_tmem_load_0_bf16[12]), "r"(_tmem_load_0_bf16[13]), "r"(_tmem_load_0_bf16[14]), "r"(_tmem_load_0_bf16[15]));
                int c_pair_0 = wg_idx * 2 + 4;
                if (m_iter_2 == n_block_2) {
                    int mask_count_1 = (warp - 4) % 4 * 32 + lane - c_pair_0 * 16;
                    uint32_t _thresh_mask_2;
                    {
                        int _lim_3 = mask_count_1;
                        if (_lim_3 <= 0) { _thresh_mask_2 = 0u; }
                        else if (_lim_3 >= 32) { _thresh_mask_2 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_2) : "r"(_lim_3));
                        }
                    }
                    if (!(~_thresh_mask_2 & (1u << 0))) _tmem_load_1[0] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 1))) _tmem_load_1[1] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 2))) _tmem_load_1[2] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 3))) _tmem_load_1[3] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 4))) _tmem_load_1[4] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 5))) _tmem_load_1[5] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 6))) _tmem_load_1[6] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 7))) _tmem_load_1[7] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 8))) _tmem_load_1[8] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 9))) _tmem_load_1[9] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 10))) _tmem_load_1[10] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 11))) _tmem_load_1[11] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 12))) _tmem_load_1[12] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 13))) _tmem_load_1[13] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 14))) _tmem_load_1[14] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 15))) _tmem_load_1[15] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 16))) _tmem_load_1[16] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 17))) _tmem_load_1[17] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 18))) _tmem_load_1[18] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 19))) _tmem_load_1[19] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 20))) _tmem_load_1[20] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 21))) _tmem_load_1[21] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 22))) _tmem_load_1[22] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 23))) _tmem_load_1[23] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 24))) _tmem_load_1[24] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 25))) _tmem_load_1[25] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 26))) _tmem_load_1[26] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 27))) _tmem_load_1[27] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 28))) _tmem_load_1[28] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 29))) _tmem_load_1[29] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 30))) _tmem_load_1[30] = -CUDART_INF_F;
                    if (!(~_thresh_mask_2 & (1u << 31))) _tmem_load_1[31] = -CUDART_INF_F;
                }
                float lse_1[32];
                #pragma unroll
                for (int j_1 = 0; j_1 < 32; j_1++) {
                    lse_1[j_1] = lse_buf[q_compute_stage * (unsigned int)BLOCK_M + (unsigned int)(c_pair_0 * 16) + (unsigned int)j_1];
                }
                float2 _f2_1 = make_float2(softmax_scale_log2, softmax_scale_log2);
                #pragma unroll
                for (int pair_1 = 0; pair_1 < 16; pair_1++) {
                    float2 affine_pair_1 = fma_sub_f32x2_rn_ftz(make_float2(_tmem_load_1[pair_1 * 2], _tmem_load_1[pair_1 * 2 + 1]), _f2_1, make_float2(lse_1[pair_1 * 2], lse_1[pair_1 * 2 + 1]));
                    _tmem_load_1[pair_1 * 2] = affine_pair_1.x;
                    _tmem_load_1[pair_1 * 2 + 1] = affine_pair_1.y;
                }
                unsigned int _tmem_load_1_bf16[16];
                softmax_frag_exp2_cast(&_tmem_load_1[0], _tmem_load_1_bf16, 0);
                int p_addr_2 = taddr + 384 + (unsigned int)(c_pair_0 * 8) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(p_addr_2), "r"(_tmem_load_1_bf16[0]), "r"(_tmem_load_1_bf16[1]), "r"(_tmem_load_1_bf16[2]), "r"(_tmem_load_1_bf16[3]), "r"(_tmem_load_1_bf16[4]), "r"(_tmem_load_1_bf16[5]), "r"(_tmem_load_1_bf16[6]), "r"(_tmem_load_1_bf16[7]), "r"(_tmem_load_1_bf16[8]), "r"(_tmem_load_1_bf16[9]), "r"(_tmem_load_1_bf16[10]), "r"(_tmem_load_1_bf16[11]), "r"(_tmem_load_1_bf16[12]), "r"(_tmem_load_1_bf16[13]), "r"(_tmem_load_1_bf16[14]), "r"(_tmem_load_1_bf16[15]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(lse_empty_addr + (q_compute_stage) * 8);
                }
                mbarrier_wait(dpsum_full_addr, _phase_dpsum_full_0);
                _phase_dpsum_full_0 ^= 1;
                mbarrier_wait(dp_full_addr, _phase_dp_full_0);
                _phase_dp_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int c_pair_3 = wg_idx * 2;
                int dp_addr = taddr + 192 + (unsigned int)(c_pair_3 * 16) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(dp_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float dpsum[32];
                #pragma unroll
                for (int j_2 = 0; j_2 < 32; j_2++) {
                    dpsum[j_2] = dpsum_buf[c_pair_3 * 16 + j_2];
                }
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], reinterpret_cast<const float2*>(dpsum)[_ls]);
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], reinterpret_cast<const float2*>(_tmem_load_0)[_ls]);
                unsigned int _tmem_load_2_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                mbarrier_wait(ds_empty_addr, _phase_ds_empty_0);
                _phase_ds_empty_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int c_p_1_0 = c_pair_3;
                int src_c0_1_0 = c_p_1_0 * 2;
                int src_c1_1_0 = c_p_1_0 * 2 + 1;
                int row0_1_0 = (warp - 4) % 4 * 32 + lane + src_c0_1_0 / 8 * BLOCK_N;
                int row1_1_0 = (warp - 4) % 4 * 32 + lane + src_c1_1_0 / 8 * BLOCK_N;
                int col0_1_0 = src_c0_1_0 % 8 * 16;
                int col1_1_0 = src_c1_1_0 % 8 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((ds_buf_addr + (unsigned int)(row0_1_0 * 128 + col0_1_0 ^ (row0_1_0 * 128 + col0_1_0 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((ds_buf_addr + (unsigned int)(row1_1_0 * 128 + col1_1_0 ^ (row1_1_0 * 128 + col1_1_0 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]) : "memory");
                int c_p_1_1 = c_pair_3 + 1;
                int src_c0_1_1 = c_p_1_1 * 2;
                int src_c1_1_1 = c_p_1_1 * 2 + 1;
                int row0_1_1 = (warp - 4) % 4 * 32 + lane + src_c0_1_1 / 8 * BLOCK_N;
                int row1_1_1 = (warp - 4) % 4 * 32 + lane + src_c1_1_1 / 8 * BLOCK_N;
                int col0_1_1 = src_c0_1_1 % 8 * 16;
                int col1_1_1 = src_c1_1_1 % 8 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((ds_buf_addr + (unsigned int)(row0_1_1 * 128 + col0_1_1 ^ (row0_1_1 * 128 + col0_1_1 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[8]), "r"(_tmem_load_2_bf16[9]), "r"(_tmem_load_2_bf16[10]), "r"(_tmem_load_2_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((ds_buf_addr + (unsigned int)(row1_1_1 * 128 + col1_1_1 ^ (row1_1_1 * 128 + col1_1_1 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[12]), "r"(_tmem_load_2_bf16[13]), "r"(_tmem_load_2_bf16[14]), "r"(_tmem_load_2_bf16[15]) : "memory");
                int ds_addr = taddr + 448 + (unsigned int)(c_pair_3 * 8) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(ds_addr), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]), "r"(_tmem_load_2_bf16[8]), "r"(_tmem_load_2_bf16[9]), "r"(_tmem_load_2_bf16[10]), "r"(_tmem_load_2_bf16[11]), "r"(_tmem_load_2_bf16[12]), "r"(_tmem_load_2_bf16[13]), "r"(_tmem_load_2_bf16[14]), "r"(_tmem_load_2_bf16[15]));
                int c_pair_11 = wg_idx * 2 + 4;
                int dp_addr_12 = taddr + 192 + (unsigned int)(c_pair_11 * 16) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                    : "r"(dp_addr_12));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float dpsum_13[32];
                #pragma unroll
                for (int j_3 = 0; j_3 < 32; j_3++) {
                    dpsum_13[j_3] = dpsum_buf[c_pair_11 * 16 + j_3];
                }
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], reinterpret_cast<const float2*>(dpsum_13)[_ls]);
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], reinterpret_cast<const float2*>(_tmem_load_1)[_ls]);
                unsigned int _tmem_load_3_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                int c_p_1_2 = c_pair_11;
                int src_c0_1_2 = c_p_1_2 * 2;
                int src_c1_1_2 = c_p_1_2 * 2 + 1;
                int row0_1_2 = (warp - 4) % 4 * 32 + lane + src_c0_1_2 / 8 * BLOCK_N;
                int row1_1_2 = (warp - 4) % 4 * 32 + lane + src_c1_1_2 / 8 * BLOCK_N;
                int col0_1_2 = src_c0_1_2 % 8 * 16;
                int col1_1_2 = src_c1_1_2 % 8 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((ds_buf_addr + (unsigned int)(row0_1_2 * 128 + col0_1_2 ^ (row0_1_2 * 128 + col0_1_2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[0]), "r"(_tmem_load_3_bf16[1]), "r"(_tmem_load_3_bf16[2]), "r"(_tmem_load_3_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((ds_buf_addr + (unsigned int)(row1_1_2 * 128 + col1_1_2 ^ (row1_1_2 * 128 + col1_1_2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[4]), "r"(_tmem_load_3_bf16[5]), "r"(_tmem_load_3_bf16[6]), "r"(_tmem_load_3_bf16[7]) : "memory");
                int c_p_1_3 = c_pair_11 + 1;
                int src_c0_1_3 = c_p_1_3 * 2;
                int src_c1_1_3 = c_p_1_3 * 2 + 1;
                int row0_1_3 = (warp - 4) % 4 * 32 + lane + src_c0_1_3 / 8 * BLOCK_N;
                int row1_1_3 = (warp - 4) % 4 * 32 + lane + src_c1_1_3 / 8 * BLOCK_N;
                int col0_1_3 = src_c0_1_3 % 8 * 16;
                int col1_1_3 = src_c1_1_3 % 8 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((ds_buf_addr + (unsigned int)(row0_1_3 * 128 + col0_1_3 ^ (row0_1_3 * 128 + col0_1_3 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[8]), "r"(_tmem_load_3_bf16[9]), "r"(_tmem_load_3_bf16[10]), "r"(_tmem_load_3_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((ds_buf_addr + (unsigned int)(row1_1_3 * 128 + col1_1_3 ^ (row1_1_3 * 128 + col1_1_3 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[12]), "r"(_tmem_load_3_bf16[13]), "r"(_tmem_load_3_bf16[14]), "r"(_tmem_load_3_bf16[15]) : "memory");
                int ds_addr_28 = taddr + 448 + (unsigned int)(c_pair_11 * 8) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(ds_addr_28), "r"(_tmem_load_3_bf16[0]), "r"(_tmem_load_3_bf16[1]), "r"(_tmem_load_3_bf16[2]), "r"(_tmem_load_3_bf16[3]), "r"(_tmem_load_3_bf16[4]), "r"(_tmem_load_3_bf16[5]), "r"(_tmem_load_3_bf16[6]), "r"(_tmem_load_3_bf16[7]), "r"(_tmem_load_3_bf16[8]), "r"(_tmem_load_3_bf16[9]), "r"(_tmem_load_3_bf16[10]), "r"(_tmem_load_3_bf16[11]), "r"(_tmem_load_3_bf16[12]), "r"(_tmem_load_3_bf16[13]), "r"(_tmem_load_3_bf16[14]), "r"(_tmem_load_3_bf16[15]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(dpsum_empty_addr);
                    mbarrier_arrive(ds_ready_addr);
                }
                q_compute_stage += 1;
                if (q_compute_stage == 2) { q_compute_stage = 0; _phase_lse_full ^= 1; }
                m_blk += 1;
            }
            int off_kv_1 = n_block_2 * BLOCK_N;
            unsigned int _phase_dkv_full_0 = 0;
            mbarrier_wait(dkv_full_addr, _phase_dkv_full_0);
            _phase_dkv_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int dv_addr = taddr + 128 + (unsigned int)(wg_idx * 32) + (unsigned int)((warp - 4) % 4 * 32 << 16);
            float _tmem_load_4[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                : "r"(dv_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int dv_stage_addr = dv_staging_addr + (unsigned int)(wg_idx * 8192);
            unsigned int _tmem_load_4_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int vec = 0; vec < 4; vec++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dv_stage_addr + (((warp - 4) % 4 * 32 + lane) * 64 + vec * 16 ^ (((warp - 4) % 4 * 32 + lane) * 64 + vec * 16 >> 7 & 3) << 4))), "r"(_tmem_load_4_bf16[vec * 4]), "r"(_tmem_load_4_bf16[vec * 4 + 1]), "r"(_tmem_load_4_bf16[vec * 4 + 2]), "r"(_tmem_load_4_bf16[vec * 4 + 3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (wg_idx == 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
            } else {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
            }
            if (warp % 4 == 0) {
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                        " [%0, {%1, %2, %3, %4}], [%5], %6;"
                        :: "l"((&dV)), "r"(wg_idx * 32), "r"(off_kv_1), "r"(head_idx_1), "r"(batch_idx_1), "r"(dv_stage_addr), "l"(0ULL) : "memory");
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                if (wg_idx == 0) {
                    asm volatile("barrier.arrive 1, 160;" ::: "memory");
                } else {
                    asm volatile("barrier.arrive 2, 160;" ::: "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (wg_idx == 0) {
                asm volatile("barrier.sync 1, 160;" ::: "memory");
            } else {
                asm volatile("barrier.sync 2, 160;" ::: "memory");
            }
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(dkv_empty_addr);
            }
            unsigned int _phase_dkv_full_1 = 0;
            mbarrier_wait(dkv_full_addr + 8, _phase_dkv_full_1);
            _phase_dkv_full_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int dk_addr = taddr + 320 + (unsigned int)(wg_idx * 32) + (unsigned int)((warp - 4) % 4 * 32 << 16);
            float _tmem_load_5[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                : "r"(dk_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            const float2 _scale2_4 = {softmax_scale, softmax_scale};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_4);
            int dk_stage_addr = dk_staging_addr + (unsigned int)(wg_idx * 8192);
            unsigned int _tmem_load_5_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int vec_1 = 0; vec_1 < 4; vec_1++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dk_stage_addr + (((warp - 4) % 4 * 32 + lane) * 64 + vec_1 * 16 ^ (((warp - 4) % 4 * 32 + lane) * 64 + vec_1 * 16 >> 7 & 3) << 4))), "r"(_tmem_load_5_bf16[vec_1 * 4]), "r"(_tmem_load_5_bf16[vec_1 * 4 + 1]), "r"(_tmem_load_5_bf16[vec_1 * 4 + 2]), "r"(_tmem_load_5_bf16[vec_1 * 4 + 3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (wg_idx == 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
            } else {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
            }
            if (warp % 4 == 0) {
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                        " [%0, {%1, %2, %3, %4}], [%5], %6;"
                        :: "l"((&dK)), "r"(wg_idx * 32), "r"(off_kv_1), "r"(head_idx_1), "r"(batch_idx_1), "r"(dk_stage_addr), "l"(0ULL) : "memory");
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                if (wg_idx == 0) {
                    asm volatile("barrier.arrive 1, 160;" ::: "memory");
                } else {
                    asm volatile("barrier.arrive 2, 160;" ::: "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (wg_idx == 0) {
                asm volatile("barrier.sync 1, 160;" ::: "memory");
            } else {
                asm volatile("barrier.sync 2, 160;" ::: "memory");
            }
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(dkv_empty_addr + 8);
            }
            unsigned int _phase_dq_tail_done_0 = 0;
            mbarrier_wait(dq_tail_done_addr, _phase_dq_tail_done_0);
            _phase_dq_tail_done_0 ^= 1;
            asm volatile("barrier.sync 5, 256;" ::: "memory");
            if (warp == 4) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: reduce ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // reduce_main
            int n_block_3 = blockIdx.x;
            int bh_3 = blockIdx.z * num_heads + blockIdx.y;
            int off_q_base_1 = bh_3 * seqlen_q;
            int m_start_3;
            {
                m_start_3 = n_block_3;
            }
            int m_blk_1 = 0;
            unsigned int _phase_dq_full_0 = 0;
            #pragma unroll 1
            for (int m_iter_3 = m_start_3; m_iter_3 < num_m_blocks; m_iter_3++) {
                int off_q_2 = off_q_base_1 + m_iter_3 * BLOCK_M;
                mbarrier_wait(dq_full_addr, _phase_dq_full_0);
                _phase_dq_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float dq_all[64];
                #pragma unroll
                for (int stage = 0; stage < HEAD_DIM / 32; stage++) {
                    int dq_addr = taddr + 192 + (unsigned int)(stage * 32) + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_6[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                        : "r"(dq_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 32; j_4++) {
                        dq_all[stage * 32 + j_4] = _tmem_load_6[j_4];
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(dq_reduce_done_addr);
                }
                #pragma unroll
                for (int g = 0; g < HEAD_DIM / 8; g++) {
                    int smem_stage = g & 7;
                    int stage_base = dq_stage_addr + (unsigned int)(smem_stage * 4096);
                    int thread_vec_base = stage_base + (warp % 4 * 32 + lane) * 16;
                    #pragma unroll
                    for (int v4 = 0; v4 < 2; v4++) {
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(thread_vec_base + v4 * 128 * 16), "f"(dq_all[g * 8 + v4 * 4]), "f"(dq_all[g * 8 + v4 * 4 + 1]), "f"(dq_all[g * 8 + v4 * 4 + 2]), "f"(dq_all[g * 8 + v4 * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            {
                                void* _cpred_dst_0 = reinterpret_cast<void*>(dQ_accum + (off_q_2 * HEAD_DIM + g * BLOCK_M * 8));
                                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                                    " [%0], [%1], %2;"
                                    :: "l"(_cpred_dst_0), "r"(stage_base), "r"((uint32_t)(4096))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 7;");
                        }
                    }
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                }
                m_blk_1 += 1;
            }
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            asm volatile("barrier.sync 4, 128;" ::: "memory");
            if (elect_sync()) {
                mbarrier_arrive(dq_tail_done_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
