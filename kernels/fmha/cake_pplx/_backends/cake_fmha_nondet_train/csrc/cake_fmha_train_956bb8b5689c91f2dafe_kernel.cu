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

#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
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

__device__ __forceinline__ float2 neg_f32x2(float2 a) {
    float2 r;
    asm("neg.f32 %0, %1;" : "=f"(r.x) : "f"(a.x));
    asm("neg.f32 %0, %1;" : "=f"(r.y) : "f"(a.y));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

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






__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_train_956bb8b5689c91f2dafe(__nv_bfloat16* __restrict__ dQ, float* __restrict__ LSE, __nv_bfloat16* __restrict__ O_gmem, __nv_bfloat16* __restrict__ dO_gmem, const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap dO, const __grid_constant__ CUtensorMap dK, const __grid_constant__ CUtensorMap dV, int seqlen_q, int seqlen_kv, float softmax_scale, float softmax_scale_log2, int num_m_blocks, int num_n_blocks, int num_heads)
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
    #define qo_q_full_addr (mbar_base + 0)
    #define qo_do_full_addr (mbar_base + 8)
    #define qo_kv_full_addr (mbar_base + 16)
    #define qo_kv_empty_addr (mbar_base + 32)
    #define qo_lse_full_addr (mbar_base + 48)
    #define qo_dpsum_full_addr (mbar_base + 56)
    #define qo_s_full_addr (mbar_base + 64)
    #define qo_dp_full_addr (mbar_base + 72)
    #define qo_s_empty_addr (mbar_base + 80)
    #define qo_dp_empty_addr (mbar_base + 88)
    #define qo_ds_full_addr (mbar_base + 96)
    #define qo_ds_empty_addr (mbar_base + 104)
    #define qo_dq_full_addr (mbar_base + 112)
    #define qo_dq_done_addr (mbar_base + 120)
    #define q_full_addr (mbar_base + 128)
    #define lse_full_addr (mbar_base + 144)
    #define dpsum_full_addr (mbar_base + 160)
    #define do_full_addr (mbar_base + 168)
    #define s_full_addr (mbar_base + 176)
    #define dp_full_addr (mbar_base + 184)
    #define sp_empty_addr (mbar_base + 192)
    #define ds_ready_addr (mbar_base + 200)
    #define dkv_full_addr (mbar_base + 208)
    #define dkv_empty_addr (mbar_base + 224)
    #define q_empty_addr (mbar_base + 240)
    #define lse_empty_addr (mbar_base + 256)
    #define dpsum_empty_addr (mbar_base + 272)
    #define do_empty_addr (mbar_base + 280)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* qo_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int qo_q_addr = smem + 1024;
    __nv_bfloat16* qo_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int qo_do_addr = smem + 33792;
    __nv_bfloat16* qo_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int qo_k_addr = smem + 66560;
    __nv_bfloat16* qo_k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int qo_k_mn_addr = smem + 66560;
    __nv_bfloat16* qo_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int qo_v_addr = smem + 132096;
    __nv_bfloat16* q_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_buf_addr = smem + 1024;
    __nv_bfloat16* q_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_mn_addr = smem + 1024;
    __nv_bfloat16* dk_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int dk_staging_addr = smem + 1024;
    __nv_bfloat16* k_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int k_buf_addr = smem + 66560;
    __nv_bfloat16* v_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int v_buf_addr = smem + 99328;
    __nv_bfloat16* do_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int do_buf_addr = smem + 132096;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int do_mn_addr = smem + 132096;
    __nv_bfloat16* dv_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int dv_staging_addr = smem + 132096;
    float* lse_buf = reinterpret_cast<float*>(smem_raw + 197632);
    const int lse_buf_addr = smem + 197632;
    float* dpsum_buf = reinterpret_cast<float*>(smem_raw + 198656);
    const int dpsum_buf_addr = smem + 198656;

    // Kernel setup ops
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dO))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dV))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dK))) : "memory"); }

    // Mbarrier init (36 barriers at smem_raw[0..288))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // qo_q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // qo_do_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'qo_pipe' ---
            // qo_kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // qo_kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // qo_lse_full: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // qo_dpsum_full: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            // qo_s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // qo_dp_full: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // qo_s_empty: 1 barriers, init_count=8
            mbarrier_init(smem + 80, 8);
            // qo_dp_empty: 1 barriers, init_count=8
            mbarrier_init(smem + 88, 8);
            // qo_ds_full: 1 barriers, init_count=8
            mbarrier_init(smem + 96, 8);
            // qo_ds_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // qo_dq_full: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // qo_dq_done: 1 barriers, init_count=4
            mbarrier_init(smem + 120, 4);
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // lse_full: 2 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            // dpsum_full: 1 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            // do_full: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // dp_full: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // sp_empty: 1 barriers, init_count=8
            mbarrier_init(smem + 192, 8);
            // ds_ready: 1 barriers, init_count=8
            mbarrier_init(smem + 200, 8);
            // --- pipeline 'dkv_pipe' ---
            // dkv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // dkv_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 224, 8);
            mbarrier_init(smem + 232, 8);
            // --- pipeline 'q_pipe' ---
            // q_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 240, 2);
            mbarrier_init(smem + 248, 2);
            // lse_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 256, 256);
            mbarrier_init(smem + 264, 256);
            // dpsum_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 272, 256);
            // do_empty: 1 barriers, init_count=2
            mbarrier_init(smem + 280, 2);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 288);
    if (warp == 12) {
        int _tmem_hold = smem + 288;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tq_s = taddr;
    const unsigned int tq_dp = taddr + 128;
    const unsigned int tq_dq = taddr + 256;
    const unsigned int tq_ds = taddr + 384;
    const unsigned int tmem_scores = taddr;
    const unsigned int tmem_softmax_p = taddr;
    const unsigned int tmem_dv_accum = taddr + 128;
    const unsigned int tmem_dp_dq = taddr + 256;
    const unsigned int tmem_ds_tmem = taddr + 256;
    const unsigned int tmem_dk_accum = taddr + 384;

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
            unsigned int _phase_qo_kv_empty = 1;
            if (blockIdx.x < num_m_blocks) {
                int qo_head = blockIdx.y;
                int qo_batch = blockIdx.z;
                int qo_off_q = blockIdx.x * BLOCK_M;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(qo_q_full_addr, 32768);
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qo_q_addr), "l"((&Q)), "r"(0), "r"(qo_off_q), "r"(qo_head), "r"(qo_batch),
                           "r"(qo_q_full_addr), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qo_q_addr + 16384), "l"((&Q)), "r"(64), "r"(qo_off_q), "r"(qo_head), "r"(qo_batch),
                           "r"(qo_q_full_addr), "l"(0ULL) : "memory");
                }
                unsigned int qo_kv_ld = 0;
                mbarrier_wait(qo_kv_empty_addr + (qo_kv_ld) * 8, _phase_qo_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(qo_kv_full_addr + (qo_kv_ld) * 8, 65536);
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qo_k_addr + qo_kv_ld * 32768), "l"((&K)), "r"(0), "r"(0), "r"(qo_head), "r"(qo_batch),
                           "r"(qo_kv_full_addr + (qo_kv_ld) * 8), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qo_v_addr + qo_kv_ld * 32768), "l"((&V)), "r"(0), "r"(0), "r"(qo_head), "r"(qo_batch),
                           "r"(qo_kv_full_addr + (qo_kv_ld) * 8), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qo_k_addr + qo_kv_ld * 32768 + 16384), "l"((&K)), "r"(64), "r"(0), "r"(qo_head), "r"(qo_batch),
                           "r"(qo_kv_full_addr + (qo_kv_ld) * 8), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qo_v_addr + qo_kv_ld * 32768 + 16384), "l"((&V)), "r"(64), "r"(0), "r"(qo_head), "r"(qo_batch),
                           "r"(qo_kv_full_addr + (qo_kv_ld) * 8), "l"(0ULL) : "memory");
                }
                qo_kv_ld += 1;
                if (qo_kv_ld == 2) { qo_kv_ld = 0; _phase_qo_kv_empty ^= 1; }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(qo_do_full_addr, 32768);
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qo_do_addr), "l"((&dO)), "r"(0), "r"(qo_off_q), "r"(qo_head), "r"(qo_batch),
                           "r"(qo_do_full_addr), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qo_do_addr + 16384), "l"((&dO)), "r"(64), "r"(qo_off_q), "r"(qo_head), "r"(qo_batch),
                           "r"(qo_do_full_addr), "l"(0ULL) : "memory");
                }
                #pragma unroll 1
                for (int qo_n = 1; qo_n < num_n_blocks; qo_n++) {
                    mbarrier_wait(qo_kv_empty_addr + (qo_kv_ld) * 8, _phase_qo_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(qo_kv_full_addr + (qo_kv_ld) * 8, 65536);
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(qo_k_addr + qo_kv_ld * 32768), "l"((&K)), "r"(0), "r"(qo_n * BLOCK_N), "r"(qo_head), "r"(qo_batch),
                               "r"(qo_kv_full_addr + (qo_kv_ld) * 8), "l"(0ULL) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(qo_v_addr + qo_kv_ld * 32768), "l"((&V)), "r"(0), "r"(qo_n * BLOCK_N), "r"(qo_head), "r"(qo_batch),
                               "r"(qo_kv_full_addr + (qo_kv_ld) * 8), "l"(0ULL) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(qo_k_addr + qo_kv_ld * 32768 + 16384), "l"((&K)), "r"(64), "r"(qo_n * BLOCK_N), "r"(qo_head), "r"(qo_batch),
                               "r"(qo_kv_full_addr + (qo_kv_ld) * 8), "l"(0ULL) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(qo_v_addr + qo_kv_ld * 32768 + 16384), "l"((&V)), "r"(64), "r"(qo_n * BLOCK_N), "r"(qo_head), "r"(qo_batch),
                               "r"(qo_kv_full_addr + (qo_kv_ld) * 8), "l"(0ULL) : "memory");
                    }
                    qo_kv_ld += 1;
                    if (qo_kv_ld == 2) { qo_kv_ld = 0; _phase_qo_kv_empty ^= 1; }
                }
            }
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_do_empty_0 = 1;
            if (blockIdx.x >= num_m_blocks) {
                int n_block = blockIdx.x - num_m_blocks;
                int head_idx = blockIdx.y;
                int batch_idx = blockIdx.z;
                int off_kv = n_block * BLOCK_N;
                unsigned int q_load_stage = 0;
                mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 65536);
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
                        :: "r"(k_buf_addr + 16384), "l"((&K)), "r"(64), "r"(off_kv), "r"(head_idx), "r"(batch_idx),
                           "r"(q_full_addr + (q_load_stage) * 8), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(q_buf_addr + q_load_stage * 32768), "l"((&Q)), "r"(0), "r"(0), "r"(head_idx), "r"(batch_idx),
                           "r"(q_full_addr + (q_load_stage) * 8), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(q_buf_addr + q_load_stage * 32768 + 16384), "l"((&Q)), "r"(64), "r"(0), "r"(head_idx), "r"(batch_idx),
                           "r"(q_full_addr + (q_load_stage) * 8), "l"(0ULL) : "memory");
                }
                q_load_stage += 1;
                if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                mbarrier_wait(do_empty_addr, _phase_do_empty_0);
                _phase_do_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(do_full_addr, 65536);
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
                        :: "r"(v_buf_addr + 16384), "l"((&V)), "r"(64), "r"(off_kv), "r"(head_idx), "r"(batch_idx),
                           "r"(do_full_addr), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(do_buf_addr), "l"((&dO)), "r"(0), "r"(0), "r"(head_idx), "r"(batch_idx),
                           "r"(do_full_addr), "l"(0ULL) : "memory");
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(do_buf_addr + 16384), "l"((&dO)), "r"(64), "r"(0), "r"(head_idx), "r"(batch_idx),
                           "r"(do_full_addr), "l"(0ULL) : "memory");
                }
                #pragma unroll 1
                for (int m_iter = 1; m_iter < num_m_blocks; m_iter++) {
                    int off_q_0 = m_iter * BLOCK_M;
                    mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 32768);
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(q_buf_addr + q_load_stage * 32768), "l"((&Q)), "r"(0), "r"(off_q_0), "r"(head_idx), "r"(batch_idx),
                               "r"(q_full_addr + (q_load_stage) * 8), "l"(0ULL) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(q_buf_addr + q_load_stage * 32768 + 16384), "l"((&Q)), "r"(64), "r"(off_q_0), "r"(head_idx), "r"(batch_idx),
                               "r"(q_full_addr + (q_load_stage) * 8), "l"(0ULL) : "memory");
                    }
                    q_load_stage += 1;
                    if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                    mbarrier_wait(do_empty_addr, _phase_do_empty_0);
                    _phase_do_empty_0 ^= 1;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(do_full_addr, 32768);
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(do_buf_addr), "l"((&dO)), "r"(0), "r"(off_q_0), "r"(head_idx), "r"(batch_idx),
                               "r"(do_full_addr), "l"(0ULL) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(do_buf_addr + 16384), "l"((&dO)), "r"(64), "r"(off_q_0), "r"(head_idx), "r"(batch_idx),
                               "r"(do_full_addr), "l"(0ULL) : "memory");
                    }
                }
                q_load_stage += 1;
                if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 32768);
                }
                mbarrier_wait(do_empty_addr, _phase_do_empty_0);
                _phase_do_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(do_full_addr, 32768);
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int _phase_qo_kv_full = 0;
            unsigned int _phase_qo_s_empty_0 = 0;
            unsigned int _phase_qo_dp_empty_0 = 0;
            unsigned int _phase_qo_ds_full_0 = 0;
            if (blockIdx.x < num_m_blocks) {
                if (elect_sync()) {
                    unsigned int qo_kv_use = 0;
                    unsigned int qo_kv_rel = 0;
                    mbarrier_wait(qo_q_full_addr, 0);
                    mbarrier_wait(qo_kv_full_addr + (qo_kv_use) * 8, _phase_qo_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((qo_q_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_1 = (((qo_k_addr) >> 4) & 0x7FFF) + (qo_kv_use) * 2048;
#else
                    int _smem_desc_lo_0 = ((qo_q_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_1 = (((qo_k_addr) >> 4) & 0x3FFF) + (qo_kv_use) * 2048;
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tq_s), "r"(0) : "memory");
                    tcgen05_commit(qo_s_full_addr);
                    mbarrier_wait(qo_do_full_addr, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((qo_do_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_3 = (((qo_v_addr) >> 4) & 0x7FFF) + (qo_kv_use) * 2048;
#else
                    int _smem_desc_lo_2 = ((qo_do_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_3 = (((qo_v_addr) >> 4) & 0x3FFF) + (qo_kv_use) * 2048;
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tq_dp), "r"(0) : "memory");
                    tcgen05_commit(qo_dp_full_addr);
                    qo_kv_use += 1;
                    if (qo_kv_use == 2) { qo_kv_use = 0; _phase_qo_kv_full ^= 1; }
                    #pragma unroll 1
                    for (int qo_n_1 = 1; qo_n_1 < num_n_blocks; qo_n_1++) {
                        mbarrier_wait(qo_kv_full_addr + (qo_kv_use) * 8, _phase_qo_kv_full);
                        mbarrier_wait(qo_s_empty_addr, _phase_qo_s_empty_0);
                        _phase_qo_s_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_4 = ((qo_q_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_5 = (((qo_k_addr) >> 4) & 0x7FFF) + (qo_kv_use) * 2048;
#else
                        int _smem_desc_lo_4 = ((qo_q_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_5 = (((qo_k_addr) >> 4) & 0x3FFF) + (qo_kv_use) * 2048;
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tq_s), "r"(0) : "memory");
                        tcgen05_commit(qo_s_full_addr);
                        mbarrier_wait(qo_dp_empty_addr, _phase_qo_dp_empty_0);
                        _phase_qo_dp_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = ((qo_do_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_7 = (((qo_v_addr) >> 4) & 0x7FFF) + (qo_kv_use) * 2048;
#else
                        int _smem_desc_lo_6 = ((qo_do_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_7 = (((qo_v_addr) >> 4) & 0x3FFF) + (qo_kv_use) * 2048;
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tq_dp), "r"(0) : "memory");
                        tcgen05_commit(qo_dp_full_addr);
                        qo_kv_use += 1;
                        if (qo_kv_use == 2) { qo_kv_use = 0; _phase_qo_kv_full ^= 1; }
                        int qo_first = ((qo_n_1 == 1) ? 1 : 0);
                        mbarrier_wait(qo_ds_full_addr, _phase_qo_ds_full_0);
                        _phase_qo_ds_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_8 = ((((qo_k_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (qo_kv_rel) * 2048;
#else
                        int _smem_desc_lo_8 = ((((qo_k_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (qo_kv_rel) * 2048;
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tq_dq), "r"(_smem_desc_lo_8), "r"(tq_ds), "r"(((qo_first) ? 0 : 1)) : "memory");
                        tcgen05_commit(qo_kv_empty_addr + (qo_kv_rel) * 8);
                        qo_kv_rel += 1;
                        if (qo_kv_rel == 2) { qo_kv_rel = 0; }
                        tcgen05_commit(qo_ds_empty_addr);
                    }
                    mbarrier_wait(qo_s_empty_addr, _phase_qo_s_empty_0);
                    _phase_qo_s_empty_0 ^= 1;
                    mbarrier_wait(qo_dp_empty_addr, _phase_qo_dp_empty_0);
                    _phase_qo_dp_empty_0 ^= 1;
                    int qo_last_first = ((num_n_blocks == 1) ? 1 : 0);
                    mbarrier_wait(qo_ds_full_addr, _phase_qo_ds_full_0);
                    _phase_qo_ds_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_9 = ((((qo_k_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (qo_kv_rel) * 2048;
#else
                    int _smem_desc_lo_9 = ((((qo_k_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (qo_kv_rel) * 2048;
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tq_dq), "r"(_smem_desc_lo_9), "r"(tq_ds), "r"(((qo_last_first) ? 0 : 1)) : "memory");
                    tcgen05_commit(qo_kv_empty_addr + (qo_kv_rel) * 8);
                    qo_kv_rel += 1;
                    if (qo_kv_rel == 2) { qo_kv_rel = 0; }
                    tcgen05_commit(qo_ds_empty_addr);
                    tcgen05_commit(qo_dq_full_addr);
                }
            }
            unsigned int _phase_q_full = 0;
            unsigned int _phase_sp_empty_0 = 1;
            unsigned int _phase_do_full_0 = 0;
            unsigned int _phase_ds_ready_0 = 0;
            unsigned int _phase_dkv_empty_0 = 1;
            unsigned int _phase_dkv_empty_1 = 1;
            if (blockIdx.x >= num_m_blocks) {
                unsigned int q_qk_stage = 0;
                unsigned int q_mma_stage = 0;
                if (elect_sync()) {
                    mbarrier_wait(q_full_addr + (q_qk_stage) * 8, _phase_q_full);
                    mbarrier_wait(sp_empty_addr, _phase_sp_empty_0);
                    _phase_sp_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = ((k_buf_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_11 = (((q_buf_addr) >> 4) & 0x7FFF) + (q_qk_stage) * 2048;
#else
                    int _smem_desc_lo_10 = ((k_buf_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_11 = (((q_buf_addr) >> 4) & 0x3FFF) + (q_qk_stage) * 2048;
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_scores), "r"(0) : "memory");
                    tcgen05_commit2(s_full_addr, q_empty_addr + (q_qk_stage) * 8);
                    q_qk_stage += 1;
                    if (q_qk_stage == 2) { q_qk_stage = 0; _phase_q_full ^= 1; }
                    mbarrier_wait(do_full_addr, _phase_do_full_0);
                    _phase_do_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_12 = ((v_buf_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_13 = ((do_buf_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_12 = ((v_buf_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_13 = ((do_buf_addr) >> 4) & 0x3FFF;
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_12), "r"(_smem_desc_lo_13), "r"(tmem_dp_dq), "r"(0) : "memory");
                    tcgen05_commit2(dp_full_addr, do_empty_addr);
                    mbarrier_wait(sp_empty_addr, _phase_sp_empty_0);
                    _phase_sp_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_14 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_14 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tmem_dv_accum), "r"(_smem_desc_lo_14), "r"(tmem_softmax_p), "r"(0) : "memory");
                    tcgen05_commit(do_empty_addr);
                    #pragma unroll 1
                    for (int m_iter_1 = 1; m_iter_1 < num_m_blocks; m_iter_1++) {
                        int first_dk = ((m_iter_1 == 1) ? 1 : 0);
                        mbarrier_wait(q_full_addr + (q_qk_stage) * 8, _phase_q_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_15 = ((k_buf_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_16 = (((q_buf_addr) >> 4) & 0x7FFF) + (q_qk_stage) * 2048;
#else
                        int _smem_desc_lo_15 = ((k_buf_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_16 = (((q_buf_addr) >> 4) & 0x3FFF) + (q_qk_stage) * 2048;
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_15), "r"(_smem_desc_lo_16), "r"(tmem_scores), "r"(0) : "memory");
                        tcgen05_commit2(s_full_addr, q_empty_addr + (q_qk_stage) * 8);
                        q_qk_stage += 1;
                        if (q_qk_stage == 2) { q_qk_stage = 0; _phase_q_full ^= 1; }
                        mbarrier_wait(ds_ready_addr, _phase_ds_ready_0);
                        _phase_ds_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_17 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (q_mma_stage) * 2048;
#else
                        int _smem_desc_lo_17 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (q_mma_stage) * 2048;
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tmem_dk_accum), "r"(_smem_desc_lo_17), "r"(tmem_ds_tmem), "r"(((first_dk) ? 0 : 1)) : "memory");
                        tcgen05_commit(q_empty_addr + (q_mma_stage) * 8);
                        q_mma_stage += 1;
                        if (q_mma_stage == 2) { q_mma_stage = 0; }
                        mbarrier_wait(do_full_addr, _phase_do_full_0);
                        _phase_do_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_18 = ((v_buf_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_19 = ((do_buf_addr) >> 4) & 0x7FFF;
#else
                        int _smem_desc_lo_18 = ((v_buf_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_19 = ((do_buf_addr) >> 4) & 0x3FFF;
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_18), "r"(_smem_desc_lo_19), "r"(tmem_dp_dq), "r"(0) : "memory");
                        tcgen05_commit2(dp_full_addr, do_empty_addr);
                        mbarrier_wait(sp_empty_addr, _phase_sp_empty_0);
                        _phase_sp_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_20 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                        int _smem_desc_lo_20 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tmem_dv_accum), "r"(_smem_desc_lo_20), "r"(tmem_softmax_p), "r"(1) : "memory");
                        tcgen05_commit(do_empty_addr);
                    }
                    tcgen05_commit(s_full_addr);
                    mbarrier_wait(dkv_empty_addr, _phase_dkv_empty_0);
                    _phase_dkv_empty_0 ^= 1;
                    tcgen05_commit(dkv_full_addr);
                    mbarrier_wait(dkv_empty_addr + 8, _phase_dkv_empty_1);
                    _phase_dkv_empty_1 ^= 1;
                    mbarrier_wait(ds_ready_addr, _phase_ds_ready_0);
                    _phase_ds_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int tail_first_dk = ((num_m_blocks == 1) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_21 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (q_mma_stage) * 2048;
#else
                    int _smem_desc_lo_21 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (q_mma_stage) * 2048;
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tmem_dk_accum), "r"(_smem_desc_lo_21), "r"(tmem_ds_tmem), "r"(((tail_first_dk) ? 0 : 1)) : "memory");
                    tcgen05_commit(dkv_full_addr + 8);
                    tcgen05_commit(q_empty_addr + (q_mma_stage) * 8);
                }
            }
        }
    }
    // ---- Role: compute ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
        { // compute_main
            unsigned int _phase_qo_s_full_0 = 0;
            unsigned int _phase_qo_dp_full_0 = 0;
            unsigned int _phase_qo_ds_empty_0 = 1;
            if (blockIdx.x < num_m_blocks) {
                int qo_wg = 1 - (warp - 4) / 4;
                int qo_row = (warp - 4) % 4 * 32 + lane;
                unsigned int qo_lane = (unsigned int)((warp - 4) % 4 * 32 << 16);
                float2 qo_sl2 = make_float2(softmax_scale_log2, softmax_scale_log2);
                mbarrier_wait(qo_lse_full_addr, 0);
                float qo_lse = lse_buf[qo_row];
                float2 qo_neg_lse = neg_f32x2(make_float2(qo_lse, qo_lse));
                float qo_dpsum = 0.0f;
                mbarrier_wait(qo_s_full_addr, _phase_qo_s_full_0);
                _phase_qo_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float qo_s0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(qo_s0[0]), "=f"(qo_s0[1]), "=f"(qo_s0[2]), "=f"(qo_s0[3]), "=f"(qo_s0[4]), "=f"(qo_s0[5]), "=f"(qo_s0[6]), "=f"(qo_s0[7]), "=f"(qo_s0[8]), "=f"(qo_s0[9]), "=f"(qo_s0[10]), "=f"(qo_s0[11]), "=f"(qo_s0[12]), "=f"(qo_s0[13]), "=f"(qo_s0[14]), "=f"(qo_s0[15]), "=f"(qo_s0[16]), "=f"(qo_s0[17]), "=f"(qo_s0[18]), "=f"(qo_s0[19]), "=f"(qo_s0[20]), "=f"(qo_s0[21]), "=f"(qo_s0[22]), "=f"(qo_s0[23]), "=f"(qo_s0[24]), "=f"(qo_s0[25]), "=f"(qo_s0[26]), "=f"(qo_s0[27]), "=f"(qo_s0[28]), "=f"(qo_s0[29]), "=f"(qo_s0[30]), "=f"(qo_s0[31])
                    : "r"(taddr + (unsigned int)(qo_wg * 32) + qo_lane));
                float qo_s1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(qo_s1[0]), "=f"(qo_s1[1]), "=f"(qo_s1[2]), "=f"(qo_s1[3]), "=f"(qo_s1[4]), "=f"(qo_s1[5]), "=f"(qo_s1[6]), "=f"(qo_s1[7]), "=f"(qo_s1[8]), "=f"(qo_s1[9]), "=f"(qo_s1[10]), "=f"(qo_s1[11]), "=f"(qo_s1[12]), "=f"(qo_s1[13]), "=f"(qo_s1[14]), "=f"(qo_s1[15]), "=f"(qo_s1[16]), "=f"(qo_s1[17]), "=f"(qo_s1[18]), "=f"(qo_s1[19]), "=f"(qo_s1[20]), "=f"(qo_s1[21]), "=f"(qo_s1[22]), "=f"(qo_s1[23]), "=f"(qo_s1[24]), "=f"(qo_s1[25]), "=f"(qo_s1[26]), "=f"(qo_s1[27]), "=f"(qo_s1[28]), "=f"(qo_s1[29]), "=f"(qo_s1[30]), "=f"(qo_s1[31])
                    : "r"(taddr + (unsigned int)(qo_wg * 32) + 64 + qo_lane));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(qo_s_empty_addr);
                }
                #pragma unroll
                for (int pair = 0; pair < 16; pair++) {
                    float2 qo_aff = fma_f32x2_rn_ftz(make_float2(qo_s0[pair * 2], qo_s0[pair * 2 + 1]), qo_sl2, qo_neg_lse);
                    qo_s0[pair * 2] = qo_aff.x;
                    qo_s0[pair * 2 + 1] = qo_aff.y;
                }
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    qo_s0[_le] = approx_exp2(qo_s0[_le]);
                }
                #pragma unroll
                for (int pair_1 = 0; pair_1 < 16; pair_1++) {
                    float2 qo_aff_1 = fma_f32x2_rn_ftz(make_float2(qo_s1[pair_1 * 2], qo_s1[pair_1 * 2 + 1]), qo_sl2, qo_neg_lse);
                    qo_s1[pair_1 * 2] = qo_aff_1.x;
                    qo_s1[pair_1 * 2 + 1] = qo_aff_1.y;
                }
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    qo_s1[_le] = approx_exp2(qo_s1[_le]);
                }
                mbarrier_wait(qo_dpsum_full_addr, 0);
                qo_dpsum = dpsum_buf[qo_row];
                mbarrier_wait(qo_dp_full_addr, _phase_qo_dp_full_0);
                _phase_qo_dp_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float qo_d0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(qo_d0[0]), "=f"(qo_d0[1]), "=f"(qo_d0[2]), "=f"(qo_d0[3]), "=f"(qo_d0[4]), "=f"(qo_d0[5]), "=f"(qo_d0[6]), "=f"(qo_d0[7]), "=f"(qo_d0[8]), "=f"(qo_d0[9]), "=f"(qo_d0[10]), "=f"(qo_d0[11]), "=f"(qo_d0[12]), "=f"(qo_d0[13]), "=f"(qo_d0[14]), "=f"(qo_d0[15]), "=f"(qo_d0[16]), "=f"(qo_d0[17]), "=f"(qo_d0[18]), "=f"(qo_d0[19]), "=f"(qo_d0[20]), "=f"(qo_d0[21]), "=f"(qo_d0[22]), "=f"(qo_d0[23]), "=f"(qo_d0[24]), "=f"(qo_d0[25]), "=f"(qo_d0[26]), "=f"(qo_d0[27]), "=f"(qo_d0[28]), "=f"(qo_d0[29]), "=f"(qo_d0[30]), "=f"(qo_d0[31])
                    : "r"(taddr + 128 + (unsigned int)(qo_wg * 32) + qo_lane));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int qo_ds0[16];
                const float2 _sub2_0 = {qo_dpsum, qo_dpsum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    sub_f32x2_inplace(&reinterpret_cast<float2*>(qo_d0)[_ls], _sub2_0);
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qo_d0)[_ls], reinterpret_cast<const float2*>(qo_s0)[_ls]);
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qo_d0[_lp*2 + 0], qo_d0[_lp*2+1 + 0]));
                    qo_ds0[_lp] = *(uint32_t*)&_bf2;
                }
                mbarrier_wait(qo_ds_empty_addr, _phase_qo_ds_empty_0);
                _phase_qo_ds_empty_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + 384 + (unsigned int)(qo_wg * 16) + qo_lane), "r"(qo_ds0[0]), "r"(qo_ds0[1]), "r"(qo_ds0[2]), "r"(qo_ds0[3]), "r"(qo_ds0[4]), "r"(qo_ds0[5]), "r"(qo_ds0[6]), "r"(qo_ds0[7]), "r"(qo_ds0[8]), "r"(qo_ds0[9]), "r"(qo_ds0[10]), "r"(qo_ds0[11]), "r"(qo_ds0[12]), "r"(qo_ds0[13]), "r"(qo_ds0[14]), "r"(qo_ds0[15]));
                float qo_d1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(qo_d1[0]), "=f"(qo_d1[1]), "=f"(qo_d1[2]), "=f"(qo_d1[3]), "=f"(qo_d1[4]), "=f"(qo_d1[5]), "=f"(qo_d1[6]), "=f"(qo_d1[7]), "=f"(qo_d1[8]), "=f"(qo_d1[9]), "=f"(qo_d1[10]), "=f"(qo_d1[11]), "=f"(qo_d1[12]), "=f"(qo_d1[13]), "=f"(qo_d1[14]), "=f"(qo_d1[15]), "=f"(qo_d1[16]), "=f"(qo_d1[17]), "=f"(qo_d1[18]), "=f"(qo_d1[19]), "=f"(qo_d1[20]), "=f"(qo_d1[21]), "=f"(qo_d1[22]), "=f"(qo_d1[23]), "=f"(qo_d1[24]), "=f"(qo_d1[25]), "=f"(qo_d1[26]), "=f"(qo_d1[27]), "=f"(qo_d1[28]), "=f"(qo_d1[29]), "=f"(qo_d1[30]), "=f"(qo_d1[31])
                    : "r"(taddr + 128 + (unsigned int)(qo_wg * 32) + 64 + qo_lane));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(qo_dp_empty_addr);
                }
                unsigned int qo_ds1[16];
                const float2 _sub2_1 = {qo_dpsum, qo_dpsum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    sub_f32x2_inplace(&reinterpret_cast<float2*>(qo_d1)[_ls], _sub2_1);
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qo_d1)[_ls], reinterpret_cast<const float2*>(qo_s1)[_ls]);
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qo_d1[_lp*2 + 0], qo_d1[_lp*2+1 + 0]));
                    qo_ds1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + 384 + 32 + (unsigned int)(qo_wg * 16) + qo_lane), "r"(qo_ds1[0]), "r"(qo_ds1[1]), "r"(qo_ds1[2]), "r"(qo_ds1[3]), "r"(qo_ds1[4]), "r"(qo_ds1[5]), "r"(qo_ds1[6]), "r"(qo_ds1[7]), "r"(qo_ds1[8]), "r"(qo_ds1[9]), "r"(qo_ds1[10]), "r"(qo_ds1[11]), "r"(qo_ds1[12]), "r"(qo_ds1[13]), "r"(qo_ds1[14]), "r"(qo_ds1[15]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(qo_ds_full_addr);
                }
                #pragma unroll 1
                for (int qo_n_2 = 1; qo_n_2 < num_n_blocks; qo_n_2++) {
                    mbarrier_wait(qo_s_full_addr, _phase_qo_s_full_0);
                    _phase_qo_s_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float qo_s0_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(qo_s0_0[0]), "=f"(qo_s0_0[1]), "=f"(qo_s0_0[2]), "=f"(qo_s0_0[3]), "=f"(qo_s0_0[4]), "=f"(qo_s0_0[5]), "=f"(qo_s0_0[6]), "=f"(qo_s0_0[7]), "=f"(qo_s0_0[8]), "=f"(qo_s0_0[9]), "=f"(qo_s0_0[10]), "=f"(qo_s0_0[11]), "=f"(qo_s0_0[12]), "=f"(qo_s0_0[13]), "=f"(qo_s0_0[14]), "=f"(qo_s0_0[15]), "=f"(qo_s0_0[16]), "=f"(qo_s0_0[17]), "=f"(qo_s0_0[18]), "=f"(qo_s0_0[19]), "=f"(qo_s0_0[20]), "=f"(qo_s0_0[21]), "=f"(qo_s0_0[22]), "=f"(qo_s0_0[23]), "=f"(qo_s0_0[24]), "=f"(qo_s0_0[25]), "=f"(qo_s0_0[26]), "=f"(qo_s0_0[27]), "=f"(qo_s0_0[28]), "=f"(qo_s0_0[29]), "=f"(qo_s0_0[30]), "=f"(qo_s0_0[31])
                        : "r"(taddr + (unsigned int)(qo_wg * 32) + qo_lane));
                    float qo_s1_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(qo_s1_0[0]), "=f"(qo_s1_0[1]), "=f"(qo_s1_0[2]), "=f"(qo_s1_0[3]), "=f"(qo_s1_0[4]), "=f"(qo_s1_0[5]), "=f"(qo_s1_0[6]), "=f"(qo_s1_0[7]), "=f"(qo_s1_0[8]), "=f"(qo_s1_0[9]), "=f"(qo_s1_0[10]), "=f"(qo_s1_0[11]), "=f"(qo_s1_0[12]), "=f"(qo_s1_0[13]), "=f"(qo_s1_0[14]), "=f"(qo_s1_0[15]), "=f"(qo_s1_0[16]), "=f"(qo_s1_0[17]), "=f"(qo_s1_0[18]), "=f"(qo_s1_0[19]), "=f"(qo_s1_0[20]), "=f"(qo_s1_0[21]), "=f"(qo_s1_0[22]), "=f"(qo_s1_0[23]), "=f"(qo_s1_0[24]), "=f"(qo_s1_0[25]), "=f"(qo_s1_0[26]), "=f"(qo_s1_0[27]), "=f"(qo_s1_0[28]), "=f"(qo_s1_0[29]), "=f"(qo_s1_0[30]), "=f"(qo_s1_0[31])
                        : "r"(taddr + (unsigned int)(qo_wg * 32) + 64 + qo_lane));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(qo_s_empty_addr);
                    }
                    #pragma unroll
                    for (int pair_2 = 0; pair_2 < 16; pair_2++) {
                        float2 qo_aff_2 = fma_f32x2_rn_ftz(make_float2(qo_s0_0[pair_2 * 2], qo_s0_0[pair_2 * 2 + 1]), qo_sl2, qo_neg_lse);
                        qo_s0_0[pair_2 * 2] = qo_aff_2.x;
                        qo_s0_0[pair_2 * 2 + 1] = qo_aff_2.y;
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        qo_s0_0[_le] = approx_exp2(qo_s0_0[_le]);
                    }
                    #pragma unroll
                    for (int pair_3 = 0; pair_3 < 16; pair_3++) {
                        float2 qo_aff_3 = fma_f32x2_rn_ftz(make_float2(qo_s1_0[pair_3 * 2], qo_s1_0[pair_3 * 2 + 1]), qo_sl2, qo_neg_lse);
                        qo_s1_0[pair_3 * 2] = qo_aff_3.x;
                        qo_s1_0[pair_3 * 2 + 1] = qo_aff_3.y;
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        qo_s1_0[_le] = approx_exp2(qo_s1_0[_le]);
                    }
                    mbarrier_wait(qo_dp_full_addr, _phase_qo_dp_full_0);
                    _phase_qo_dp_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float qo_d0_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(qo_d0_0[0]), "=f"(qo_d0_0[1]), "=f"(qo_d0_0[2]), "=f"(qo_d0_0[3]), "=f"(qo_d0_0[4]), "=f"(qo_d0_0[5]), "=f"(qo_d0_0[6]), "=f"(qo_d0_0[7]), "=f"(qo_d0_0[8]), "=f"(qo_d0_0[9]), "=f"(qo_d0_0[10]), "=f"(qo_d0_0[11]), "=f"(qo_d0_0[12]), "=f"(qo_d0_0[13]), "=f"(qo_d0_0[14]), "=f"(qo_d0_0[15]), "=f"(qo_d0_0[16]), "=f"(qo_d0_0[17]), "=f"(qo_d0_0[18]), "=f"(qo_d0_0[19]), "=f"(qo_d0_0[20]), "=f"(qo_d0_0[21]), "=f"(qo_d0_0[22]), "=f"(qo_d0_0[23]), "=f"(qo_d0_0[24]), "=f"(qo_d0_0[25]), "=f"(qo_d0_0[26]), "=f"(qo_d0_0[27]), "=f"(qo_d0_0[28]), "=f"(qo_d0_0[29]), "=f"(qo_d0_0[30]), "=f"(qo_d0_0[31])
                        : "r"(taddr + 128 + (unsigned int)(qo_wg * 32) + qo_lane));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int qo_ds0_0[16];
                    const float2 _sub2_2 = {qo_dpsum, qo_dpsum};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        sub_f32x2_inplace(&reinterpret_cast<float2*>(qo_d0_0)[_ls], _sub2_2);
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qo_d0_0)[_ls], reinterpret_cast<const float2*>(qo_s0_0)[_ls]);
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qo_d0_0[_lp*2 + 0], qo_d0_0[_lp*2+1 + 0]));
                        qo_ds0_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    mbarrier_wait(qo_ds_empty_addr, _phase_qo_ds_empty_0);
                    _phase_qo_ds_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 384 + (unsigned int)(qo_wg * 16) + qo_lane), "r"(qo_ds0_0[0]), "r"(qo_ds0_0[1]), "r"(qo_ds0_0[2]), "r"(qo_ds0_0[3]), "r"(qo_ds0_0[4]), "r"(qo_ds0_0[5]), "r"(qo_ds0_0[6]), "r"(qo_ds0_0[7]), "r"(qo_ds0_0[8]), "r"(qo_ds0_0[9]), "r"(qo_ds0_0[10]), "r"(qo_ds0_0[11]), "r"(qo_ds0_0[12]), "r"(qo_ds0_0[13]), "r"(qo_ds0_0[14]), "r"(qo_ds0_0[15]));
                    float qo_d1_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(qo_d1_0[0]), "=f"(qo_d1_0[1]), "=f"(qo_d1_0[2]), "=f"(qo_d1_0[3]), "=f"(qo_d1_0[4]), "=f"(qo_d1_0[5]), "=f"(qo_d1_0[6]), "=f"(qo_d1_0[7]), "=f"(qo_d1_0[8]), "=f"(qo_d1_0[9]), "=f"(qo_d1_0[10]), "=f"(qo_d1_0[11]), "=f"(qo_d1_0[12]), "=f"(qo_d1_0[13]), "=f"(qo_d1_0[14]), "=f"(qo_d1_0[15]), "=f"(qo_d1_0[16]), "=f"(qo_d1_0[17]), "=f"(qo_d1_0[18]), "=f"(qo_d1_0[19]), "=f"(qo_d1_0[20]), "=f"(qo_d1_0[21]), "=f"(qo_d1_0[22]), "=f"(qo_d1_0[23]), "=f"(qo_d1_0[24]), "=f"(qo_d1_0[25]), "=f"(qo_d1_0[26]), "=f"(qo_d1_0[27]), "=f"(qo_d1_0[28]), "=f"(qo_d1_0[29]), "=f"(qo_d1_0[30]), "=f"(qo_d1_0[31])
                        : "r"(taddr + 128 + (unsigned int)(qo_wg * 32) + 64 + qo_lane));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(qo_dp_empty_addr);
                    }
                    unsigned int qo_ds1_0[16];
                    const float2 _sub2_3 = {qo_dpsum, qo_dpsum};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        sub_f32x2_inplace(&reinterpret_cast<float2*>(qo_d1_0)[_ls], _sub2_3);
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qo_d1_0)[_ls], reinterpret_cast<const float2*>(qo_s1_0)[_ls]);
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qo_d1_0[_lp*2 + 0], qo_d1_0[_lp*2+1 + 0]));
                        qo_ds1_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 384 + 32 + (unsigned int)(qo_wg * 16) + qo_lane), "r"(qo_ds1_0[0]), "r"(qo_ds1_0[1]), "r"(qo_ds1_0[2]), "r"(qo_ds1_0[3]), "r"(qo_ds1_0[4]), "r"(qo_ds1_0[5]), "r"(qo_ds1_0[6]), "r"(qo_ds1_0[7]), "r"(qo_ds1_0[8]), "r"(qo_ds1_0[9]), "r"(qo_ds1_0[10]), "r"(qo_ds1_0[11]), "r"(qo_ds1_0[12]), "r"(qo_ds1_0[13]), "r"(qo_ds1_0[14]), "r"(qo_ds1_0[15]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(qo_ds_full_addr);
                    }
                }
                mbarrier_wait(qo_dq_done_addr, 0);
            }
            unsigned int _phase_lse_full = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_dpsum_full_0 = 0;
            unsigned int _phase_dp_full_0 = 0;
            unsigned int _phase_dkv_full_0 = 0;
            unsigned int _phase_dkv_full_1 = 0;
            if (blockIdx.x >= num_m_blocks) {
                int n_block_1 = blockIdx.x - num_m_blocks;
                int head_idx_1 = blockIdx.y;
                int batch_idx_1 = blockIdx.z;
                unsigned int q_compute_stage = 0;
                int wg_idx = 1 - (warp - 4) / 4;
                int m_blk = 0;
                #pragma unroll 1
                for (int m_iter_2 = 0; m_iter_2 < num_m_blocks; m_iter_2++) {
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
                    int c_pair = wg_idx * 2;
                    float lse[32];
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        lse[j] = lse_buf[q_compute_stage * (unsigned int)BLOCK_M + (unsigned int)(c_pair * 16) + (unsigned int)j];
                    }
                    float2 _f2_0 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    #pragma unroll
                    for (int pair_4 = 0; pair_4 < 16; pair_4++) {
                        float2 affine_pair = fma_sub_f32x2_rn_ftz(make_float2(_tmem_load_0[pair_4 * 2], _tmem_load_0[pair_4 * 2 + 1]), _f2_0, make_float2(lse[pair_4 * 2], lse[pair_4 * 2 + 1]));
                        _tmem_load_0[pair_4 * 2] = affine_pair.x;
                        _tmem_load_0[pair_4 * 2 + 1] = affine_pair.y;
                    }
                    unsigned int _tmem_load_0_bf16[16];
                    softmax_frag_exp2_cast(&_tmem_load_0[0], _tmem_load_0_bf16, 0);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int p_addr = taddr + (unsigned int)(c_pair * 8) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]), "r"(_tmem_load_0_bf16[8]), "r"(_tmem_load_0_bf16[9]), "r"(_tmem_load_0_bf16[10]), "r"(_tmem_load_0_bf16[11]), "r"(_tmem_load_0_bf16[12]), "r"(_tmem_load_0_bf16[13]), "r"(_tmem_load_0_bf16[14]), "r"(_tmem_load_0_bf16[15]));
                    int c_pair_0 = wg_idx * 2 + 4;
                    float lse_1[32];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 32; j_1++) {
                        lse_1[j_1] = lse_buf[q_compute_stage * (unsigned int)BLOCK_M + (unsigned int)(c_pair_0 * 16) + (unsigned int)j_1];
                    }
                    float2 _f2_1 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    #pragma unroll
                    for (int pair_5 = 0; pair_5 < 16; pair_5++) {
                        float2 affine_pair_1 = fma_sub_f32x2_rn_ftz(make_float2(_tmem_load_1[pair_5 * 2], _tmem_load_1[pair_5 * 2 + 1]), _f2_1, make_float2(lse_1[pair_5 * 2], lse_1[pair_5 * 2 + 1]));
                        _tmem_load_1[pair_5 * 2] = affine_pair_1.x;
                        _tmem_load_1[pair_5 * 2 + 1] = affine_pair_1.y;
                    }
                    unsigned int _tmem_load_1_bf16[16];
                    softmax_frag_exp2_cast(&_tmem_load_1[0], _tmem_load_1_bf16, 0);
                    int p_addr_2 = taddr + (unsigned int)(c_pair_0 * 8) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr_2), "r"(_tmem_load_1_bf16[0]), "r"(_tmem_load_1_bf16[1]), "r"(_tmem_load_1_bf16[2]), "r"(_tmem_load_1_bf16[3]), "r"(_tmem_load_1_bf16[4]), "r"(_tmem_load_1_bf16[5]), "r"(_tmem_load_1_bf16[6]), "r"(_tmem_load_1_bf16[7]), "r"(_tmem_load_1_bf16[8]), "r"(_tmem_load_1_bf16[9]), "r"(_tmem_load_1_bf16[10]), "r"(_tmem_load_1_bf16[11]), "r"(_tmem_load_1_bf16[12]), "r"(_tmem_load_1_bf16[13]), "r"(_tmem_load_1_bf16[14]), "r"(_tmem_load_1_bf16[15]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(sp_empty_addr);
                    }
                    mbarrier_arrive(lse_empty_addr + (q_compute_stage) * 8);
                    mbarrier_wait(dpsum_full_addr, _phase_dpsum_full_0);
                    _phase_dpsum_full_0 ^= 1;
                    mbarrier_wait(dp_full_addr, _phase_dp_full_0);
                    _phase_dp_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int c_pair_3 = wg_idx * 2;
                    int dp_addr = taddr + 256 + (unsigned int)(c_pair_3 * 16) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                    float _tmem_load_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                        : "r"(dp_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int ds_addr = taddr + 256 + (unsigned int)(c_pair_3 * 8) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(ds_addr), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]), "r"(_tmem_load_2_bf16[8]), "r"(_tmem_load_2_bf16[9]), "r"(_tmem_load_2_bf16[10]), "r"(_tmem_load_2_bf16[11]), "r"(_tmem_load_2_bf16[12]), "r"(_tmem_load_2_bf16[13]), "r"(_tmem_load_2_bf16[14]), "r"(_tmem_load_2_bf16[15]));
                    int c_pair_11 = wg_idx * 2 + 4;
                    int dp_addr_12 = taddr + 256 + (unsigned int)(c_pair_11 * 16) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                    float _tmem_load_3[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                        : "r"(dp_addr_12));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
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
                    int ds_addr_28 = taddr + 256 + (unsigned int)(c_pair_11 * 8) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(ds_addr_28), "r"(_tmem_load_3_bf16[0]), "r"(_tmem_load_3_bf16[1]), "r"(_tmem_load_3_bf16[2]), "r"(_tmem_load_3_bf16[3]), "r"(_tmem_load_3_bf16[4]), "r"(_tmem_load_3_bf16[5]), "r"(_tmem_load_3_bf16[6]), "r"(_tmem_load_3_bf16[7]), "r"(_tmem_load_3_bf16[8]), "r"(_tmem_load_3_bf16[9]), "r"(_tmem_load_3_bf16[10]), "r"(_tmem_load_3_bf16[11]), "r"(_tmem_load_3_bf16[12]), "r"(_tmem_load_3_bf16[13]), "r"(_tmem_load_3_bf16[14]), "r"(_tmem_load_3_bf16[15]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    mbarrier_arrive(dpsum_empty_addr);
                    if (elect_sync()) {
                        mbarrier_arrive(ds_ready_addr);
                    }
                    q_compute_stage += 1;
                    if (q_compute_stage == 2) { q_compute_stage = 0; _phase_lse_full ^= 1; }
                    m_blk += 1;
                }
                int off_kv_1 = n_block_1 * BLOCK_N;
                mbarrier_wait(dkv_full_addr, _phase_dkv_full_0);
                _phase_dkv_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int dv_addr = taddr + 128 + (unsigned int)(wg_idx * 64) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                float _tmem_load_4[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                    : "r"(dv_addr));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_4[32]), "=f"(_tmem_load_4[33]), "=f"(_tmem_load_4[34]), "=f"(_tmem_load_4[35]), "=f"(_tmem_load_4[36]), "=f"(_tmem_load_4[37]), "=f"(_tmem_load_4[38]), "=f"(_tmem_load_4[39]), "=f"(_tmem_load_4[40]), "=f"(_tmem_load_4[41]), "=f"(_tmem_load_4[42]), "=f"(_tmem_load_4[43]), "=f"(_tmem_load_4[44]), "=f"(_tmem_load_4[45]), "=f"(_tmem_load_4[46]), "=f"(_tmem_load_4[47]), "=f"(_tmem_load_4[48]), "=f"(_tmem_load_4[49]), "=f"(_tmem_load_4[50]), "=f"(_tmem_load_4[51]), "=f"(_tmem_load_4[52]), "=f"(_tmem_load_4[53]), "=f"(_tmem_load_4[54]), "=f"(_tmem_load_4[55]), "=f"(_tmem_load_4[56]), "=f"(_tmem_load_4[57]), "=f"(_tmem_load_4[58]), "=f"(_tmem_load_4[59]), "=f"(_tmem_load_4[60]), "=f"(_tmem_load_4[61]), "=f"(_tmem_load_4[62]), "=f"(_tmem_load_4[63])
                    : "r"(dv_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int dv_stage_addr = dv_staging_addr + (unsigned int)(wg_idx * 16384);
                unsigned int _tmem_load_4_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int vec = 0; vec < 8; vec++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dv_stage_addr + (((warp - 4) % 4 * 32 + lane) * 128 + vec * 16 ^ (((warp - 4) % 4 * 32 + lane) * 128 + vec * 16 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[vec * 4]), "r"(_tmem_load_4_bf16[vec * 4 + 1]), "r"(_tmem_load_4_bf16[vec * 4 + 2]), "r"(_tmem_load_4_bf16[vec * 4 + 3]) : "memory");
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
                            :: "l"((&dV)), "r"(wg_idx * 64), "r"(off_kv_1), "r"(head_idx_1), "r"(batch_idx_1), "r"(dv_stage_addr), "l"(0ULL) : "memory");
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
                mbarrier_wait(dkv_full_addr + 8, _phase_dkv_full_1);
                _phase_dkv_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int dk_addr = taddr + 384 + (unsigned int)(wg_idx * 64) + (unsigned int)((warp - 4) % 4 * 32 << 16);
                float _tmem_load_5[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                    : "r"(dk_addr));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_5[32]), "=f"(_tmem_load_5[33]), "=f"(_tmem_load_5[34]), "=f"(_tmem_load_5[35]), "=f"(_tmem_load_5[36]), "=f"(_tmem_load_5[37]), "=f"(_tmem_load_5[38]), "=f"(_tmem_load_5[39]), "=f"(_tmem_load_5[40]), "=f"(_tmem_load_5[41]), "=f"(_tmem_load_5[42]), "=f"(_tmem_load_5[43]), "=f"(_tmem_load_5[44]), "=f"(_tmem_load_5[45]), "=f"(_tmem_load_5[46]), "=f"(_tmem_load_5[47]), "=f"(_tmem_load_5[48]), "=f"(_tmem_load_5[49]), "=f"(_tmem_load_5[50]), "=f"(_tmem_load_5[51]), "=f"(_tmem_load_5[52]), "=f"(_tmem_load_5[53]), "=f"(_tmem_load_5[54]), "=f"(_tmem_load_5[55]), "=f"(_tmem_load_5[56]), "=f"(_tmem_load_5[57]), "=f"(_tmem_load_5[58]), "=f"(_tmem_load_5[59]), "=f"(_tmem_load_5[60]), "=f"(_tmem_load_5[61]), "=f"(_tmem_load_5[62]), "=f"(_tmem_load_5[63])
                    : "r"(dk_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                const float2 _scale2_4 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_4);
                int dk_stage_addr = dk_staging_addr + (unsigned int)(wg_idx * 16384);
                unsigned int _tmem_load_5_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                    _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int vec_1 = 0; vec_1 < 8; vec_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dk_stage_addr + (((warp - 4) % 4 * 32 + lane) * 128 + vec_1 * 16 ^ (((warp - 4) % 4 * 32 + lane) * 128 + vec_1 * 16 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[vec_1 * 4]), "r"(_tmem_load_5_bf16[vec_1 * 4 + 1]), "r"(_tmem_load_5_bf16[vec_1 * 4 + 2]), "r"(_tmem_load_5_bf16[vec_1 * 4 + 3]) : "memory");
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
                            :: "l"((&dK)), "r"(wg_idx * 64), "r"(off_kv_1), "r"(head_idx_1), "r"(batch_idx_1), "r"(dk_stage_addr), "l"(0ULL) : "memory");
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
            }
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
            if (blockIdx.x < num_m_blocks) {
                int qo_m = blockIdx.x;
                int qo_head_r = blockIdx.y;
                int qo_batch_r = blockIdx.z;
                int qo_bh = blockIdx.z * num_heads + blockIdx.y;
                int qo_row_in_warp = lane / 16;
                int qo_lane_in_row = lane % 16;
                int qo_lane_col = qo_lane_in_row * 8;
                int qo_q_row0 = qo_batch_r * seqlen_q + qo_m * BLOCK_M;
                float qo_lse_nat = LSE[qo_bh * seqlen_q + qo_m * BLOCK_M + warp * 32 + lane];
                float qo_pre_o[32];
                float qo_pre_do[32];
                int qo_pre_row_0 = qo_q_row0 + warp * 2 + qo_row_in_warp;
                int qo_pre_base_0 = (qo_pre_row_0 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(O_gmem + qo_pre_base_0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_pre_o[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                            (&qo_pre_o[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(dO_gmem + qo_pre_base_0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_pre_do[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&qo_pre_do[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
                int qo_pre_row_1 = qo_q_row0 + 8 + warp * 2 + qo_row_in_warp;
                int qo_pre_base_1 = (qo_pre_row_1 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(O_gmem + qo_pre_base_1);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_pre_o[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&qo_pre_o[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(dO_gmem + qo_pre_base_1);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_pre_do[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&qo_pre_do[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
                int qo_pre_row_2 = qo_q_row0 + 16 + warp * 2 + qo_row_in_warp;
                int qo_pre_base_2 = (qo_pre_row_2 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(O_gmem + qo_pre_base_2);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_pre_o[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&qo_pre_o[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(dO_gmem + qo_pre_base_2);
                    uint4 _vld_5[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_5[_blk] = _vptr_5[_blk];
                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_pre_do[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                            (&qo_pre_do[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                        }
                    }
                }
                int qo_pre_row_3 = qo_q_row0 + 24 + warp * 2 + qo_row_in_warp;
                int qo_pre_base_3 = (qo_pre_row_3 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(O_gmem + qo_pre_base_3);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_pre_o[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                            (&qo_pre_o[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(dO_gmem + qo_pre_base_3);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_pre_do[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                            (&qo_pre_do[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float qo_lse_log2 = ((qo_lse_nat != -CUDART_INF_F) ? qo_lse_nat * 1.442695f : 0.0f);
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(lse_buf_addr + (unsigned int)((warp * 32 + lane) * 4)), "r"(__float_as_uint(qo_lse_log2)) : "memory");
                mbarrier_arrive(qo_lse_full_addr);
                float qo_dps[16];
                float qo_partial_0 = 0.0f;
                #pragma unroll
                for (int j_0 = 0; j_0 < 8; j_0++) {
                    qo_partial_0 = __fmaf_rn(qo_pre_o[j_0], qo_pre_do[j_0], qo_partial_0);
                }
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_0, 8);
                qo_partial_0 = qo_partial_0 + _shfl_xor_sync_8;
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_0, 4);
                qo_partial_0 = qo_partial_0 + _shfl_xor_sync_9;
                float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_0, 2);
                qo_partial_0 = qo_partial_0 + _shfl_xor_sync_10;
                float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_0, 1);
                qo_partial_0 = qo_partial_0 + _shfl_xor_sync_11;
                qo_dps[0] = qo_partial_0;
                float qo_partial_1 = 0.0f;
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 8; j_1_1++) {
                    qo_partial_1 = __fmaf_rn(qo_pre_o[8 + j_1_1], qo_pre_do[8 + j_1_1], qo_partial_1);
                }
                float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_1, 8);
                qo_partial_1 = qo_partial_1 + _shfl_xor_sync_12;
                float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_1, 4);
                qo_partial_1 = qo_partial_1 + _shfl_xor_sync_13;
                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_1, 2);
                qo_partial_1 = qo_partial_1 + _shfl_xor_sync_14;
                float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_1, 1);
                qo_partial_1 = qo_partial_1 + _shfl_xor_sync_15;
                qo_dps[1] = qo_partial_1;
                float qo_partial_2 = 0.0f;
                #pragma unroll
                for (int j_2_1 = 0; j_2_1 < 8; j_2_1++) {
                    qo_partial_2 = __fmaf_rn(qo_pre_o[16 + j_2_1], qo_pre_do[16 + j_2_1], qo_partial_2);
                }
                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_2, 8);
                qo_partial_2 = qo_partial_2 + _shfl_xor_sync_16;
                float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_2, 4);
                qo_partial_2 = qo_partial_2 + _shfl_xor_sync_17;
                float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_2, 2);
                qo_partial_2 = qo_partial_2 + _shfl_xor_sync_18;
                float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_2, 1);
                qo_partial_2 = qo_partial_2 + _shfl_xor_sync_19;
                qo_dps[2] = qo_partial_2;
                float qo_partial_3 = 0.0f;
                #pragma unroll
                for (int j_3_1 = 0; j_3_1 < 8; j_3_1++) {
                    qo_partial_3 = __fmaf_rn(qo_pre_o[24 + j_3_1], qo_pre_do[24 + j_3_1], qo_partial_3);
                }
                float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_3, 8);
                qo_partial_3 = qo_partial_3 + _shfl_xor_sync_20;
                float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_3, 4);
                qo_partial_3 = qo_partial_3 + _shfl_xor_sync_21;
                float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_3, 2);
                qo_partial_3 = qo_partial_3 + _shfl_xor_sync_22;
                float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_3, 1);
                qo_partial_3 = qo_partial_3 + _shfl_xor_sync_23;
                qo_dps[3] = qo_partial_3;
                float qo_partial_4 = 0.0f;
                float qo_row_o_0[8];
                float qo_row_do_0[8];
                int qo_late_row_0 = qo_q_row0 + 32 + warp * 2 + qo_row_in_warp;
                int qo_late_base_0 = (qo_late_row_0 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_24 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_0);
                    uint4 _vld_24[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_24[_blk] = _vptr_24[_blk];
                        uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
                            (&qo_row_o_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_25 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_0);
                    uint4 _vld_25[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_25[_blk] = _vptr_25[_blk];
                        uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
                            (&qo_row_do_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    qo_partial_4 = __fmaf_rn(qo_row_o_0[j_4], qo_row_do_0[j_4], qo_partial_4);
                }
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_4, 8);
                qo_partial_4 = qo_partial_4 + _shfl_xor_sync_26;
                float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_4, 4);
                qo_partial_4 = qo_partial_4 + _shfl_xor_sync_27;
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_4, 2);
                qo_partial_4 = qo_partial_4 + _shfl_xor_sync_28;
                float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_4, 1);
                qo_partial_4 = qo_partial_4 + _shfl_xor_sync_29;
                qo_dps[4] = qo_partial_4;
                float qo_partial_5 = 0.0f;
                float qo_row_o_1[8];
                float qo_row_do_1[8];
                int qo_late_row_1 = qo_q_row0 + 40 + warp * 2 + qo_row_in_warp;
                int qo_late_base_1 = (qo_late_row_1 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_30 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_1);
                    uint4 _vld_30[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_30[_blk] = _vptr_30[_blk];
                        uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
                            (&qo_row_o_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_31 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_1);
                    uint4 _vld_31[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_31[_blk] = _vptr_31[_blk];
                        uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                            (&qo_row_do_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_5 = 0; j_5 < 8; j_5++) {
                    qo_partial_5 = __fmaf_rn(qo_row_o_1[j_5], qo_row_do_1[j_5], qo_partial_5);
                }
                float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_5, 8);
                qo_partial_5 = qo_partial_5 + _shfl_xor_sync_32;
                float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_5, 4);
                qo_partial_5 = qo_partial_5 + _shfl_xor_sync_33;
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_5, 2);
                qo_partial_5 = qo_partial_5 + _shfl_xor_sync_34;
                float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_5, 1);
                qo_partial_5 = qo_partial_5 + _shfl_xor_sync_35;
                qo_dps[5] = qo_partial_5;
                float qo_partial_6 = 0.0f;
                float qo_row_o_2[8];
                float qo_row_do_2[8];
                int qo_late_row_2 = qo_q_row0 + 48 + warp * 2 + qo_row_in_warp;
                int qo_late_base_2 = (qo_late_row_2 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_36 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_2);
                    uint4 _vld_36[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_36[_blk] = _vptr_36[_blk];
                        uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                            (&qo_row_o_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_37 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_2);
                    uint4 _vld_37[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_37[_blk] = _vptr_37[_blk];
                        uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
                            (&qo_row_do_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_6 = 0; j_6 < 8; j_6++) {
                    qo_partial_6 = __fmaf_rn(qo_row_o_2[j_6], qo_row_do_2[j_6], qo_partial_6);
                }
                float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_6, 8);
                qo_partial_6 = qo_partial_6 + _shfl_xor_sync_38;
                float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_6, 4);
                qo_partial_6 = qo_partial_6 + _shfl_xor_sync_39;
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_6, 2);
                qo_partial_6 = qo_partial_6 + _shfl_xor_sync_40;
                float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_6, 1);
                qo_partial_6 = qo_partial_6 + _shfl_xor_sync_41;
                qo_dps[6] = qo_partial_6;
                float qo_partial_7 = 0.0f;
                float qo_row_o_3[8];
                float qo_row_do_3[8];
                int qo_late_row_3 = qo_q_row0 + 56 + warp * 2 + qo_row_in_warp;
                int qo_late_base_3 = (qo_late_row_3 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_42 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_3);
                    uint4 _vld_42[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_42[_blk] = _vptr_42[_blk];
                        uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                            (&qo_row_o_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_43 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_3);
                    uint4 _vld_43[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_43[_blk] = _vptr_43[_blk];
                        uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                            (&qo_row_do_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_7 = 0; j_7 < 8; j_7++) {
                    qo_partial_7 = __fmaf_rn(qo_row_o_3[j_7], qo_row_do_3[j_7], qo_partial_7);
                }
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_7, 8);
                qo_partial_7 = qo_partial_7 + _shfl_xor_sync_44;
                float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_7, 4);
                qo_partial_7 = qo_partial_7 + _shfl_xor_sync_45;
                float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_7, 2);
                qo_partial_7 = qo_partial_7 + _shfl_xor_sync_46;
                float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_7, 1);
                qo_partial_7 = qo_partial_7 + _shfl_xor_sync_47;
                qo_dps[7] = qo_partial_7;
                float qo_partial_8 = 0.0f;
                float qo_row_o_4[8];
                float qo_row_do_4[8];
                int qo_late_row_4 = qo_q_row0 + 64 + warp * 2 + qo_row_in_warp;
                int qo_late_base_4 = (qo_late_row_4 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_48 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_4);
                    uint4 _vld_48[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_48[_blk] = _vptr_48[_blk];
                        uint32_t* _vpairs_48 = reinterpret_cast<uint32_t*>(&_vld_48[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_48[_pair]) << 16);
                            (&qo_row_o_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_48[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_49 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_4);
                    uint4 _vld_49[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_49[_blk] = _vptr_49[_blk];
                        uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
                            (&qo_row_do_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_8 = 0; j_8 < 8; j_8++) {
                    qo_partial_8 = __fmaf_rn(qo_row_o_4[j_8], qo_row_do_4[j_8], qo_partial_8);
                }
                float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_8, 8);
                qo_partial_8 = qo_partial_8 + _shfl_xor_sync_50;
                float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_8, 4);
                qo_partial_8 = qo_partial_8 + _shfl_xor_sync_51;
                float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_8, 2);
                qo_partial_8 = qo_partial_8 + _shfl_xor_sync_52;
                float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_8, 1);
                qo_partial_8 = qo_partial_8 + _shfl_xor_sync_53;
                qo_dps[8] = qo_partial_8;
                float qo_partial_9 = 0.0f;
                float qo_row_o_5[8];
                float qo_row_do_5[8];
                int qo_late_row_5 = qo_q_row0 + 72 + warp * 2 + qo_row_in_warp;
                int qo_late_base_5 = (qo_late_row_5 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_54 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_5);
                    uint4 _vld_54[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_54[_blk] = _vptr_54[_blk];
                        uint32_t* _vpairs_54 = reinterpret_cast<uint32_t*>(&_vld_54[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) << 16);
                            (&qo_row_o_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_55 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_5);
                    uint4 _vld_55[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_55[_blk] = _vptr_55[_blk];
                        uint32_t* _vpairs_55 = reinterpret_cast<uint32_t*>(&_vld_55[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) << 16);
                            (&qo_row_do_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_9 = 0; j_9 < 8; j_9++) {
                    qo_partial_9 = __fmaf_rn(qo_row_o_5[j_9], qo_row_do_5[j_9], qo_partial_9);
                }
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_9, 8);
                qo_partial_9 = qo_partial_9 + _shfl_xor_sync_56;
                float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_9, 4);
                qo_partial_9 = qo_partial_9 + _shfl_xor_sync_57;
                float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_9, 2);
                qo_partial_9 = qo_partial_9 + _shfl_xor_sync_58;
                float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_9, 1);
                qo_partial_9 = qo_partial_9 + _shfl_xor_sync_59;
                qo_dps[9] = qo_partial_9;
                float qo_partial_10 = 0.0f;
                float qo_row_o_6[8];
                float qo_row_do_6[8];
                int qo_late_row_6 = qo_q_row0 + 80 + warp * 2 + qo_row_in_warp;
                int qo_late_base_6 = (qo_late_row_6 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_60 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_6);
                    uint4 _vld_60[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_60[_blk] = _vptr_60[_blk];
                        uint32_t* _vpairs_60 = reinterpret_cast<uint32_t*>(&_vld_60[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) << 16);
                            (&qo_row_o_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_61 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_6);
                    uint4 _vld_61[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_61[_blk] = _vptr_61[_blk];
                        uint32_t* _vpairs_61 = reinterpret_cast<uint32_t*>(&_vld_61[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) << 16);
                            (&qo_row_do_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_10 = 0; j_10 < 8; j_10++) {
                    qo_partial_10 = __fmaf_rn(qo_row_o_6[j_10], qo_row_do_6[j_10], qo_partial_10);
                }
                float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_10, 8);
                qo_partial_10 = qo_partial_10 + _shfl_xor_sync_62;
                float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_10, 4);
                qo_partial_10 = qo_partial_10 + _shfl_xor_sync_63;
                float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_10, 2);
                qo_partial_10 = qo_partial_10 + _shfl_xor_sync_64;
                float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_10, 1);
                qo_partial_10 = qo_partial_10 + _shfl_xor_sync_65;
                qo_dps[10] = qo_partial_10;
                float qo_partial_11 = 0.0f;
                float qo_row_o_7[8];
                float qo_row_do_7[8];
                int qo_late_row_7 = qo_q_row0 + 88 + warp * 2 + qo_row_in_warp;
                int qo_late_base_7 = (qo_late_row_7 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_66 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_7);
                    uint4 _vld_66[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_66[_blk] = _vptr_66[_blk];
                        uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
                            (&qo_row_o_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_67 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_7);
                    uint4 _vld_67[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_67[_blk] = _vptr_67[_blk];
                        uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) << 16);
                            (&qo_row_do_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_11 = 0; j_11 < 8; j_11++) {
                    qo_partial_11 = __fmaf_rn(qo_row_o_7[j_11], qo_row_do_7[j_11], qo_partial_11);
                }
                float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_11, 8);
                qo_partial_11 = qo_partial_11 + _shfl_xor_sync_68;
                float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_11, 4);
                qo_partial_11 = qo_partial_11 + _shfl_xor_sync_69;
                float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_11, 2);
                qo_partial_11 = qo_partial_11 + _shfl_xor_sync_70;
                float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_11, 1);
                qo_partial_11 = qo_partial_11 + _shfl_xor_sync_71;
                qo_dps[11] = qo_partial_11;
                float qo_partial_12 = 0.0f;
                float qo_row_o_8[8];
                float qo_row_do_8[8];
                int qo_late_row_8 = qo_q_row0 + 96 + warp * 2 + qo_row_in_warp;
                int qo_late_base_8 = (qo_late_row_8 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_72 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_8);
                    uint4 _vld_72[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_72[_blk] = _vptr_72[_blk];
                        uint32_t* _vpairs_72 = reinterpret_cast<uint32_t*>(&_vld_72[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) << 16);
                            (&qo_row_o_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_73 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_8);
                    uint4 _vld_73[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_73[_blk] = _vptr_73[_blk];
                        uint32_t* _vpairs_73 = reinterpret_cast<uint32_t*>(&_vld_73[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) << 16);
                            (&qo_row_do_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_12 = 0; j_12 < 8; j_12++) {
                    qo_partial_12 = __fmaf_rn(qo_row_o_8[j_12], qo_row_do_8[j_12], qo_partial_12);
                }
                float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_12, 8);
                qo_partial_12 = qo_partial_12 + _shfl_xor_sync_74;
                float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_12, 4);
                qo_partial_12 = qo_partial_12 + _shfl_xor_sync_75;
                float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_12, 2);
                qo_partial_12 = qo_partial_12 + _shfl_xor_sync_76;
                float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_12, 1);
                qo_partial_12 = qo_partial_12 + _shfl_xor_sync_77;
                qo_dps[12] = qo_partial_12;
                float qo_partial_13 = 0.0f;
                float qo_row_o_9[8];
                float qo_row_do_9[8];
                int qo_late_row_9 = qo_q_row0 + 104 + warp * 2 + qo_row_in_warp;
                int qo_late_base_9 = (qo_late_row_9 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_78 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_9);
                    uint4 _vld_78[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_78[_blk] = _vptr_78[_blk];
                        uint32_t* _vpairs_78 = reinterpret_cast<uint32_t*>(&_vld_78[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) << 16);
                            (&qo_row_o_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_79 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_9);
                    uint4 _vld_79[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_79[_blk] = _vptr_79[_blk];
                        uint32_t* _vpairs_79 = reinterpret_cast<uint32_t*>(&_vld_79[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) << 16);
                            (&qo_row_do_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_13 = 0; j_13 < 8; j_13++) {
                    qo_partial_13 = __fmaf_rn(qo_row_o_9[j_13], qo_row_do_9[j_13], qo_partial_13);
                }
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_13, 8);
                qo_partial_13 = qo_partial_13 + _shfl_xor_sync_80;
                float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_13, 4);
                qo_partial_13 = qo_partial_13 + _shfl_xor_sync_81;
                float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_13, 2);
                qo_partial_13 = qo_partial_13 + _shfl_xor_sync_82;
                float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_13, 1);
                qo_partial_13 = qo_partial_13 + _shfl_xor_sync_83;
                qo_dps[13] = qo_partial_13;
                float qo_partial_14 = 0.0f;
                float qo_row_o_10[8];
                float qo_row_do_10[8];
                int qo_late_row_10 = qo_q_row0 + 112 + warp * 2 + qo_row_in_warp;
                int qo_late_base_10 = (qo_late_row_10 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_84 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_10);
                    uint4 _vld_84[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_84[_blk] = _vptr_84[_blk];
                        uint32_t* _vpairs_84 = reinterpret_cast<uint32_t*>(&_vld_84[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_84[_pair]) << 16);
                            (&qo_row_o_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_84[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_85 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_10);
                    uint4 _vld_85[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_85[_blk] = _vptr_85[_blk];
                        uint32_t* _vpairs_85 = reinterpret_cast<uint32_t*>(&_vld_85[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) << 16);
                            (&qo_row_do_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_14 = 0; j_14 < 8; j_14++) {
                    qo_partial_14 = __fmaf_rn(qo_row_o_10[j_14], qo_row_do_10[j_14], qo_partial_14);
                }
                float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_14, 8);
                qo_partial_14 = qo_partial_14 + _shfl_xor_sync_86;
                float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_14, 4);
                qo_partial_14 = qo_partial_14 + _shfl_xor_sync_87;
                float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_14, 2);
                qo_partial_14 = qo_partial_14 + _shfl_xor_sync_88;
                float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_14, 1);
                qo_partial_14 = qo_partial_14 + _shfl_xor_sync_89;
                qo_dps[14] = qo_partial_14;
                float qo_partial_15 = 0.0f;
                float qo_row_o_11[8];
                float qo_row_do_11[8];
                int qo_late_row_11 = qo_q_row0 + 120 + warp * 2 + qo_row_in_warp;
                int qo_late_base_11 = (qo_late_row_11 * num_heads + qo_head_r) * HEAD_DIM + qo_lane_col;
                {
                    const uint4* _vptr_90 = reinterpret_cast<const uint4*>(O_gmem + qo_late_base_11);
                    uint4 _vld_90[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_90[_blk] = _vptr_90[_blk];
                        uint32_t* _vpairs_90 = reinterpret_cast<uint32_t*>(&_vld_90[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_o_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) << 16);
                            (&qo_row_o_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_91 = reinterpret_cast<const uint4*>(dO_gmem + qo_late_base_11);
                    uint4 _vld_91[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_91[_blk] = _vptr_91[_blk];
                        uint32_t* _vpairs_91 = reinterpret_cast<uint32_t*>(&_vld_91[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qo_row_do_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) << 16);
                            (&qo_row_do_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_15 = 0; j_15 < 8; j_15++) {
                    qo_partial_15 = __fmaf_rn(qo_row_o_11[j_15], qo_row_do_11[j_15], qo_partial_15);
                }
                float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_15, 8);
                qo_partial_15 = qo_partial_15 + _shfl_xor_sync_92;
                float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_15, 4);
                qo_partial_15 = qo_partial_15 + _shfl_xor_sync_93;
                float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_15, 2);
                qo_partial_15 = qo_partial_15 + _shfl_xor_sync_94;
                float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, qo_partial_15, 1);
                qo_partial_15 = qo_partial_15 + _shfl_xor_sync_95;
                qo_dps[15] = qo_partial_15;
                if (qo_lane_in_row == 0) {
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[0])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((8 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[1])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((16 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[2])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((24 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[3])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((32 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[4])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((40 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[5])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((48 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[6])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((56 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[7])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((64 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[8])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((72 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[9])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((80 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[10])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((88 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[11])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((96 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[12])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((104 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[13])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((112 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[14])) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((120 + warp * 2 + qo_row_in_warp) * 4)), "r"(__float_as_uint(qo_dps[15])) : "memory");
                }
                mbarrier_arrive(qo_dpsum_full_addr);
                mbarrier_wait(qo_dq_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float qo_dq_all[128];
                #pragma unroll
                for (int stage = 0; stage < HEAD_DIM / 32; stage++) {
                    int qo_dq_addr = taddr + 256 + (unsigned int)(stage * 32) + (unsigned int)(warp % 4 * 32 << 16);
                    float qo_dq_ld[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(qo_dq_ld[0]), "=f"(qo_dq_ld[1]), "=f"(qo_dq_ld[2]), "=f"(qo_dq_ld[3]), "=f"(qo_dq_ld[4]), "=f"(qo_dq_ld[5]), "=f"(qo_dq_ld[6]), "=f"(qo_dq_ld[7]), "=f"(qo_dq_ld[8]), "=f"(qo_dq_ld[9]), "=f"(qo_dq_ld[10]), "=f"(qo_dq_ld[11]), "=f"(qo_dq_ld[12]), "=f"(qo_dq_ld[13]), "=f"(qo_dq_ld[14]), "=f"(qo_dq_ld[15]), "=f"(qo_dq_ld[16]), "=f"(qo_dq_ld[17]), "=f"(qo_dq_ld[18]), "=f"(qo_dq_ld[19]), "=f"(qo_dq_ld[20]), "=f"(qo_dq_ld[21]), "=f"(qo_dq_ld[22]), "=f"(qo_dq_ld[23]), "=f"(qo_dq_ld[24]), "=f"(qo_dq_ld[25]), "=f"(qo_dq_ld[26]), "=f"(qo_dq_ld[27]), "=f"(qo_dq_ld[28]), "=f"(qo_dq_ld[29]), "=f"(qo_dq_ld[30]), "=f"(qo_dq_ld[31])
                        : "r"(qo_dq_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 32; j_16++) {
                        qo_dq_all[stage * 32 + j_16] = qo_dq_ld[j_16];
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(qo_dq_done_addr);
                }
                const float2 _scale2_96 = {softmax_scale, softmax_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 64; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qo_dq_all)[_ls], _scale2_96);
                unsigned int qo_dq_bf16[64];
                #pragma unroll
                for (int _lp = 0; _lp < 64; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qo_dq_all[_lp*2 + 0], qo_dq_all[_lp*2+1 + 0]));
                    qo_dq_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned long long qo_dq_base = (unsigned long long)(((qo_batch_r * seqlen_q + qo_m * BLOCK_M + (warp % 4 * 32 + lane)) * num_heads + qo_head_r) * HEAD_DIM);
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 0ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16)[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 8ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (4))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 16ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (8))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 24ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (12))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 32ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (16))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 40ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (20))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 48ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (24))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 56ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (28))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 64ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (32))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 72ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (36))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 80ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (40))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 88ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (44))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 96ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (48))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 104ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (52))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 112ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (56))[0];
                reinterpret_cast<int4*>(dQ + (qo_dq_base + 120ull))[0] = reinterpret_cast<int4*>(qo_dq_bf16 + (60))[0];
            }
            unsigned int _phase_lse_empty = 1;
            unsigned int _phase_dpsum_empty_0 = 1;
            if (blockIdx.x >= num_m_blocks) {
                int head_idx_2 = blockIdx.y;
                int batch_idx_2 = blockIdx.z;
                int bh = blockIdx.z * num_heads + blockIdx.y;
                int row_in_warp = lane / 16;
                int lane_in_row = lane % 16;
                int lane_col = lane_in_row * 8;
                unsigned int st_stage = 0;
                #pragma unroll 1
                for (int m_iter_3 = 0; m_iter_3 < num_m_blocks; m_iter_3++) {
                    int q_row0 = batch_idx_2 * seqlen_q + m_iter_3 * BLOCK_M;
                    float lse_nat = LSE[bh * seqlen_q + m_iter_3 * BLOCK_M + warp * 32 + lane];
                    float pre_o[32];
                    float pre_do[32];
                    int pre_row_0 = q_row0 + warp * 2 + row_in_warp;
                    int pre_base_0 = (pre_row_0 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_97 = reinterpret_cast<const uint4*>(O_gmem + pre_base_0);
                        uint4 _vld_97[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_97[_blk] = _vptr_97[_blk];
                            uint32_t* _vpairs_97 = reinterpret_cast<uint32_t*>(&_vld_97[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pre_o[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_97[_pair]) << 16);
                                (&pre_o[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_97[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_98 = reinterpret_cast<const uint4*>(dO_gmem + pre_base_0);
                        uint4 _vld_98[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_98[_blk] = _vptr_98[_blk];
                            uint32_t* _vpairs_98 = reinterpret_cast<uint32_t*>(&_vld_98[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pre_do[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_98[_pair]) << 16);
                                (&pre_do[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_98[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    int pre_row_1 = q_row0 + 8 + warp * 2 + row_in_warp;
                    int pre_base_1 = (pre_row_1 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_99 = reinterpret_cast<const uint4*>(O_gmem + pre_base_1);
                        uint4 _vld_99[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_99[_blk] = _vptr_99[_blk];
                            uint32_t* _vpairs_99 = reinterpret_cast<uint32_t*>(&_vld_99[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pre_o[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_99[_pair]) << 16);
                                (&pre_o[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_99[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_100 = reinterpret_cast<const uint4*>(dO_gmem + pre_base_1);
                        uint4 _vld_100[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_100[_blk] = _vptr_100[_blk];
                            uint32_t* _vpairs_100 = reinterpret_cast<uint32_t*>(&_vld_100[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pre_do[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_100[_pair]) << 16);
                                (&pre_do[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_100[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    int pre_row_2 = q_row0 + 16 + warp * 2 + row_in_warp;
                    int pre_base_2 = (pre_row_2 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_101 = reinterpret_cast<const uint4*>(O_gmem + pre_base_2);
                        uint4 _vld_101[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_101[_blk] = _vptr_101[_blk];
                            uint32_t* _vpairs_101 = reinterpret_cast<uint32_t*>(&_vld_101[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pre_o[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_101[_pair]) << 16);
                                (&pre_o[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_101[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_102 = reinterpret_cast<const uint4*>(dO_gmem + pre_base_2);
                        uint4 _vld_102[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_102[_blk] = _vptr_102[_blk];
                            uint32_t* _vpairs_102 = reinterpret_cast<uint32_t*>(&_vld_102[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pre_do[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_102[_pair]) << 16);
                                (&pre_do[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_102[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    int pre_row_3 = q_row0 + 24 + warp * 2 + row_in_warp;
                    int pre_base_3 = (pre_row_3 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_103 = reinterpret_cast<const uint4*>(O_gmem + pre_base_3);
                        uint4 _vld_103[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_103[_blk] = _vptr_103[_blk];
                            uint32_t* _vpairs_103 = reinterpret_cast<uint32_t*>(&_vld_103[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pre_o[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_103[_pair]) << 16);
                                (&pre_o[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_103[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_104 = reinterpret_cast<const uint4*>(dO_gmem + pre_base_3);
                        uint4 _vld_104[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_104[_blk] = _vptr_104[_blk];
                            uint32_t* _vpairs_104 = reinterpret_cast<uint32_t*>(&_vld_104[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&pre_do[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_104[_pair]) << 16);
                                (&pre_do[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_104[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float lse_log2 = ((lse_nat != -CUDART_INF_F) ? lse_nat * 1.442695f : 0.0f);
                    mbarrier_wait(lse_empty_addr + (st_stage) * 8, _phase_lse_empty);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(lse_buf_addr + st_stage * (unsigned int)BLOCK_M * 4 + (unsigned int)((warp * 32 + lane) * 4)), "r"(__float_as_uint(lse_log2)) : "memory");
                    mbarrier_arrive(lse_full_addr + (st_stage) * 8);
                    st_stage += 1;
                    if (st_stage == 2) { st_stage = 0; _phase_lse_empty ^= 1; }
                    float dps[16];
                    float partial_0 = 0.0f;
                    #pragma unroll
                    for (int j_16_1 = 0; j_16_1 < 8; j_16_1++) {
                        partial_0 = __fmaf_rn(pre_o[j_16_1], pre_do[j_16_1], partial_0);
                    }
                    float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, partial_0, 8);
                    partial_0 = partial_0 + _shfl_xor_sync_105;
                    float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, partial_0, 4);
                    partial_0 = partial_0 + _shfl_xor_sync_106;
                    float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, partial_0, 2);
                    partial_0 = partial_0 + _shfl_xor_sync_107;
                    float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, partial_0, 1);
                    partial_0 = partial_0 + _shfl_xor_sync_108;
                    dps[0] = partial_0;
                    float partial_1 = 0.0f;
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 8; j_17++) {
                        partial_1 = __fmaf_rn(pre_o[8 + j_17], pre_do[8 + j_17], partial_1);
                    }
                    float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, partial_1, 8);
                    partial_1 = partial_1 + _shfl_xor_sync_109;
                    float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, partial_1, 4);
                    partial_1 = partial_1 + _shfl_xor_sync_110;
                    float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, partial_1, 2);
                    partial_1 = partial_1 + _shfl_xor_sync_111;
                    float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, partial_1, 1);
                    partial_1 = partial_1 + _shfl_xor_sync_112;
                    dps[1] = partial_1;
                    float partial_2 = 0.0f;
                    #pragma unroll
                    for (int j_18 = 0; j_18 < 8; j_18++) {
                        partial_2 = __fmaf_rn(pre_o[16 + j_18], pre_do[16 + j_18], partial_2);
                    }
                    float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, partial_2, 8);
                    partial_2 = partial_2 + _shfl_xor_sync_113;
                    float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, partial_2, 4);
                    partial_2 = partial_2 + _shfl_xor_sync_114;
                    float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, partial_2, 2);
                    partial_2 = partial_2 + _shfl_xor_sync_115;
                    float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, partial_2, 1);
                    partial_2 = partial_2 + _shfl_xor_sync_116;
                    dps[2] = partial_2;
                    float partial_3 = 0.0f;
                    #pragma unroll
                    for (int j_19 = 0; j_19 < 8; j_19++) {
                        partial_3 = __fmaf_rn(pre_o[24 + j_19], pre_do[24 + j_19], partial_3);
                    }
                    float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, partial_3, 8);
                    partial_3 = partial_3 + _shfl_xor_sync_117;
                    float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, partial_3, 4);
                    partial_3 = partial_3 + _shfl_xor_sync_118;
                    float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, partial_3, 2);
                    partial_3 = partial_3 + _shfl_xor_sync_119;
                    float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, partial_3, 1);
                    partial_3 = partial_3 + _shfl_xor_sync_120;
                    dps[3] = partial_3;
                    float partial_4 = 0.0f;
                    float row_o_0[8];
                    float row_do_0[8];
                    int late_row_0 = q_row0 + 32 + warp * 2 + row_in_warp;
                    int late_base_0 = (late_row_0 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_121 = reinterpret_cast<const uint4*>(O_gmem + late_base_0);
                        uint4 _vld_121[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_121[_blk] = _vptr_121[_blk];
                            uint32_t* _vpairs_121 = reinterpret_cast<uint32_t*>(&_vld_121[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_121[_pair]) << 16);
                                (&row_o_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_121[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_122 = reinterpret_cast<const uint4*>(dO_gmem + late_base_0);
                        uint4 _vld_122[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_122[_blk] = _vptr_122[_blk];
                            uint32_t* _vpairs_122 = reinterpret_cast<uint32_t*>(&_vld_122[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_122[_pair]) << 16);
                                (&row_do_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_122[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 8; j_20++) {
                        partial_4 = __fmaf_rn(row_o_0[j_20], row_do_0[j_20], partial_4);
                    }
                    float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, partial_4, 8);
                    partial_4 = partial_4 + _shfl_xor_sync_123;
                    float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, partial_4, 4);
                    partial_4 = partial_4 + _shfl_xor_sync_124;
                    float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, partial_4, 2);
                    partial_4 = partial_4 + _shfl_xor_sync_125;
                    float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, partial_4, 1);
                    partial_4 = partial_4 + _shfl_xor_sync_126;
                    dps[4] = partial_4;
                    float partial_5 = 0.0f;
                    float row_o_1[8];
                    float row_do_1[8];
                    int late_row_1 = q_row0 + 40 + warp * 2 + row_in_warp;
                    int late_base_1 = (late_row_1 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_127 = reinterpret_cast<const uint4*>(O_gmem + late_base_1);
                        uint4 _vld_127[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_127[_blk] = _vptr_127[_blk];
                            uint32_t* _vpairs_127 = reinterpret_cast<uint32_t*>(&_vld_127[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_127[_pair]) << 16);
                                (&row_o_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_127[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_128 = reinterpret_cast<const uint4*>(dO_gmem + late_base_1);
                        uint4 _vld_128[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_128[_blk] = _vptr_128[_blk];
                            uint32_t* _vpairs_128 = reinterpret_cast<uint32_t*>(&_vld_128[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_128[_pair]) << 16);
                                (&row_do_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_128[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_21 = 0; j_21 < 8; j_21++) {
                        partial_5 = __fmaf_rn(row_o_1[j_21], row_do_1[j_21], partial_5);
                    }
                    float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, partial_5, 8);
                    partial_5 = partial_5 + _shfl_xor_sync_129;
                    float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, partial_5, 4);
                    partial_5 = partial_5 + _shfl_xor_sync_130;
                    float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, partial_5, 2);
                    partial_5 = partial_5 + _shfl_xor_sync_131;
                    float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, partial_5, 1);
                    partial_5 = partial_5 + _shfl_xor_sync_132;
                    dps[5] = partial_5;
                    float partial_6 = 0.0f;
                    float row_o_2[8];
                    float row_do_2[8];
                    int late_row_2 = q_row0 + 48 + warp * 2 + row_in_warp;
                    int late_base_2 = (late_row_2 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_133 = reinterpret_cast<const uint4*>(O_gmem + late_base_2);
                        uint4 _vld_133[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_133[_blk] = _vptr_133[_blk];
                            uint32_t* _vpairs_133 = reinterpret_cast<uint32_t*>(&_vld_133[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_133[_pair]) << 16);
                                (&row_o_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_133[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_134 = reinterpret_cast<const uint4*>(dO_gmem + late_base_2);
                        uint4 _vld_134[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_134[_blk] = _vptr_134[_blk];
                            uint32_t* _vpairs_134 = reinterpret_cast<uint32_t*>(&_vld_134[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_134[_pair]) << 16);
                                (&row_do_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_134[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_22 = 0; j_22 < 8; j_22++) {
                        partial_6 = __fmaf_rn(row_o_2[j_22], row_do_2[j_22], partial_6);
                    }
                    float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, partial_6, 8);
                    partial_6 = partial_6 + _shfl_xor_sync_135;
                    float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, partial_6, 4);
                    partial_6 = partial_6 + _shfl_xor_sync_136;
                    float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, partial_6, 2);
                    partial_6 = partial_6 + _shfl_xor_sync_137;
                    float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, partial_6, 1);
                    partial_6 = partial_6 + _shfl_xor_sync_138;
                    dps[6] = partial_6;
                    float partial_7 = 0.0f;
                    float row_o_3[8];
                    float row_do_3[8];
                    int late_row_3 = q_row0 + 56 + warp * 2 + row_in_warp;
                    int late_base_3 = (late_row_3 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_139 = reinterpret_cast<const uint4*>(O_gmem + late_base_3);
                        uint4 _vld_139[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_139[_blk] = _vptr_139[_blk];
                            uint32_t* _vpairs_139 = reinterpret_cast<uint32_t*>(&_vld_139[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_139[_pair]) << 16);
                                (&row_o_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_139[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_140 = reinterpret_cast<const uint4*>(dO_gmem + late_base_3);
                        uint4 _vld_140[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_140[_blk] = _vptr_140[_blk];
                            uint32_t* _vpairs_140 = reinterpret_cast<uint32_t*>(&_vld_140[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_140[_pair]) << 16);
                                (&row_do_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_140[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_23 = 0; j_23 < 8; j_23++) {
                        partial_7 = __fmaf_rn(row_o_3[j_23], row_do_3[j_23], partial_7);
                    }
                    float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, partial_7, 8);
                    partial_7 = partial_7 + _shfl_xor_sync_141;
                    float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, partial_7, 4);
                    partial_7 = partial_7 + _shfl_xor_sync_142;
                    float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, partial_7, 2);
                    partial_7 = partial_7 + _shfl_xor_sync_143;
                    float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, partial_7, 1);
                    partial_7 = partial_7 + _shfl_xor_sync_144;
                    dps[7] = partial_7;
                    float partial_8 = 0.0f;
                    float row_o_4[8];
                    float row_do_4[8];
                    int late_row_4 = q_row0 + 64 + warp * 2 + row_in_warp;
                    int late_base_4 = (late_row_4 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_145 = reinterpret_cast<const uint4*>(O_gmem + late_base_4);
                        uint4 _vld_145[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_145[_blk] = _vptr_145[_blk];
                            uint32_t* _vpairs_145 = reinterpret_cast<uint32_t*>(&_vld_145[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_145[_pair]) << 16);
                                (&row_o_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_145[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_146 = reinterpret_cast<const uint4*>(dO_gmem + late_base_4);
                        uint4 _vld_146[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_146[_blk] = _vptr_146[_blk];
                            uint32_t* _vpairs_146 = reinterpret_cast<uint32_t*>(&_vld_146[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_146[_pair]) << 16);
                                (&row_do_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_146[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_24 = 0; j_24 < 8; j_24++) {
                        partial_8 = __fmaf_rn(row_o_4[j_24], row_do_4[j_24], partial_8);
                    }
                    float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, partial_8, 8);
                    partial_8 = partial_8 + _shfl_xor_sync_147;
                    float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, partial_8, 4);
                    partial_8 = partial_8 + _shfl_xor_sync_148;
                    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, partial_8, 2);
                    partial_8 = partial_8 + _shfl_xor_sync_149;
                    float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, partial_8, 1);
                    partial_8 = partial_8 + _shfl_xor_sync_150;
                    dps[8] = partial_8;
                    float partial_9 = 0.0f;
                    float row_o_5[8];
                    float row_do_5[8];
                    int late_row_5 = q_row0 + 72 + warp * 2 + row_in_warp;
                    int late_base_5 = (late_row_5 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_151 = reinterpret_cast<const uint4*>(O_gmem + late_base_5);
                        uint4 _vld_151[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_151[_blk] = _vptr_151[_blk];
                            uint32_t* _vpairs_151 = reinterpret_cast<uint32_t*>(&_vld_151[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_151[_pair]) << 16);
                                (&row_o_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_151[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_152 = reinterpret_cast<const uint4*>(dO_gmem + late_base_5);
                        uint4 _vld_152[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_152[_blk] = _vptr_152[_blk];
                            uint32_t* _vpairs_152 = reinterpret_cast<uint32_t*>(&_vld_152[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_152[_pair]) << 16);
                                (&row_do_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_152[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_25 = 0; j_25 < 8; j_25++) {
                        partial_9 = __fmaf_rn(row_o_5[j_25], row_do_5[j_25], partial_9);
                    }
                    float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, partial_9, 8);
                    partial_9 = partial_9 + _shfl_xor_sync_153;
                    float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, partial_9, 4);
                    partial_9 = partial_9 + _shfl_xor_sync_154;
                    float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, partial_9, 2);
                    partial_9 = partial_9 + _shfl_xor_sync_155;
                    float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, partial_9, 1);
                    partial_9 = partial_9 + _shfl_xor_sync_156;
                    dps[9] = partial_9;
                    float partial_10 = 0.0f;
                    float row_o_6[8];
                    float row_do_6[8];
                    int late_row_6 = q_row0 + 80 + warp * 2 + row_in_warp;
                    int late_base_6 = (late_row_6 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_157 = reinterpret_cast<const uint4*>(O_gmem + late_base_6);
                        uint4 _vld_157[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_157[_blk] = _vptr_157[_blk];
                            uint32_t* _vpairs_157 = reinterpret_cast<uint32_t*>(&_vld_157[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_157[_pair]) << 16);
                                (&row_o_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_157[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_158 = reinterpret_cast<const uint4*>(dO_gmem + late_base_6);
                        uint4 _vld_158[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_158[_blk] = _vptr_158[_blk];
                            uint32_t* _vpairs_158 = reinterpret_cast<uint32_t*>(&_vld_158[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_158[_pair]) << 16);
                                (&row_do_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_158[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_26 = 0; j_26 < 8; j_26++) {
                        partial_10 = __fmaf_rn(row_o_6[j_26], row_do_6[j_26], partial_10);
                    }
                    float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, partial_10, 8);
                    partial_10 = partial_10 + _shfl_xor_sync_159;
                    float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, partial_10, 4);
                    partial_10 = partial_10 + _shfl_xor_sync_160;
                    float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, partial_10, 2);
                    partial_10 = partial_10 + _shfl_xor_sync_161;
                    float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, partial_10, 1);
                    partial_10 = partial_10 + _shfl_xor_sync_162;
                    dps[10] = partial_10;
                    float partial_11 = 0.0f;
                    float row_o_7[8];
                    float row_do_7[8];
                    int late_row_7 = q_row0 + 88 + warp * 2 + row_in_warp;
                    int late_base_7 = (late_row_7 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_163 = reinterpret_cast<const uint4*>(O_gmem + late_base_7);
                        uint4 _vld_163[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_163[_blk] = _vptr_163[_blk];
                            uint32_t* _vpairs_163 = reinterpret_cast<uint32_t*>(&_vld_163[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_163[_pair]) << 16);
                                (&row_o_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_163[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_164 = reinterpret_cast<const uint4*>(dO_gmem + late_base_7);
                        uint4 _vld_164[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_164[_blk] = _vptr_164[_blk];
                            uint32_t* _vpairs_164 = reinterpret_cast<uint32_t*>(&_vld_164[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_164[_pair]) << 16);
                                (&row_do_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_164[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_27 = 0; j_27 < 8; j_27++) {
                        partial_11 = __fmaf_rn(row_o_7[j_27], row_do_7[j_27], partial_11);
                    }
                    float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, partial_11, 8);
                    partial_11 = partial_11 + _shfl_xor_sync_165;
                    float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, partial_11, 4);
                    partial_11 = partial_11 + _shfl_xor_sync_166;
                    float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, partial_11, 2);
                    partial_11 = partial_11 + _shfl_xor_sync_167;
                    float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, partial_11, 1);
                    partial_11 = partial_11 + _shfl_xor_sync_168;
                    dps[11] = partial_11;
                    float partial_12 = 0.0f;
                    float row_o_8[8];
                    float row_do_8[8];
                    int late_row_8 = q_row0 + 96 + warp * 2 + row_in_warp;
                    int late_base_8 = (late_row_8 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_169 = reinterpret_cast<const uint4*>(O_gmem + late_base_8);
                        uint4 _vld_169[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_169[_blk] = _vptr_169[_blk];
                            uint32_t* _vpairs_169 = reinterpret_cast<uint32_t*>(&_vld_169[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_169[_pair]) << 16);
                                (&row_o_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_169[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_170 = reinterpret_cast<const uint4*>(dO_gmem + late_base_8);
                        uint4 _vld_170[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_170[_blk] = _vptr_170[_blk];
                            uint32_t* _vpairs_170 = reinterpret_cast<uint32_t*>(&_vld_170[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_170[_pair]) << 16);
                                (&row_do_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_170[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_28 = 0; j_28 < 8; j_28++) {
                        partial_12 = __fmaf_rn(row_o_8[j_28], row_do_8[j_28], partial_12);
                    }
                    float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, partial_12, 8);
                    partial_12 = partial_12 + _shfl_xor_sync_171;
                    float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, partial_12, 4);
                    partial_12 = partial_12 + _shfl_xor_sync_172;
                    float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, partial_12, 2);
                    partial_12 = partial_12 + _shfl_xor_sync_173;
                    float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, partial_12, 1);
                    partial_12 = partial_12 + _shfl_xor_sync_174;
                    dps[12] = partial_12;
                    float partial_13 = 0.0f;
                    float row_o_9[8];
                    float row_do_9[8];
                    int late_row_9 = q_row0 + 104 + warp * 2 + row_in_warp;
                    int late_base_9 = (late_row_9 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_175 = reinterpret_cast<const uint4*>(O_gmem + late_base_9);
                        uint4 _vld_175[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_175[_blk] = _vptr_175[_blk];
                            uint32_t* _vpairs_175 = reinterpret_cast<uint32_t*>(&_vld_175[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_175[_pair]) << 16);
                                (&row_o_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_175[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_176 = reinterpret_cast<const uint4*>(dO_gmem + late_base_9);
                        uint4 _vld_176[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_176[_blk] = _vptr_176[_blk];
                            uint32_t* _vpairs_176 = reinterpret_cast<uint32_t*>(&_vld_176[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_176[_pair]) << 16);
                                (&row_do_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_176[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_29 = 0; j_29 < 8; j_29++) {
                        partial_13 = __fmaf_rn(row_o_9[j_29], row_do_9[j_29], partial_13);
                    }
                    float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, partial_13, 8);
                    partial_13 = partial_13 + _shfl_xor_sync_177;
                    float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, partial_13, 4);
                    partial_13 = partial_13 + _shfl_xor_sync_178;
                    float _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, partial_13, 2);
                    partial_13 = partial_13 + _shfl_xor_sync_179;
                    float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, partial_13, 1);
                    partial_13 = partial_13 + _shfl_xor_sync_180;
                    dps[13] = partial_13;
                    float partial_14 = 0.0f;
                    float row_o_10[8];
                    float row_do_10[8];
                    int late_row_10 = q_row0 + 112 + warp * 2 + row_in_warp;
                    int late_base_10 = (late_row_10 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_181 = reinterpret_cast<const uint4*>(O_gmem + late_base_10);
                        uint4 _vld_181[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_181[_blk] = _vptr_181[_blk];
                            uint32_t* _vpairs_181 = reinterpret_cast<uint32_t*>(&_vld_181[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_181[_pair]) << 16);
                                (&row_o_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_181[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_182 = reinterpret_cast<const uint4*>(dO_gmem + late_base_10);
                        uint4 _vld_182[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_182[_blk] = _vptr_182[_blk];
                            uint32_t* _vpairs_182 = reinterpret_cast<uint32_t*>(&_vld_182[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_182[_pair]) << 16);
                                (&row_do_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_182[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_30 = 0; j_30 < 8; j_30++) {
                        partial_14 = __fmaf_rn(row_o_10[j_30], row_do_10[j_30], partial_14);
                    }
                    float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, partial_14, 8);
                    partial_14 = partial_14 + _shfl_xor_sync_183;
                    float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, partial_14, 4);
                    partial_14 = partial_14 + _shfl_xor_sync_184;
                    float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, partial_14, 2);
                    partial_14 = partial_14 + _shfl_xor_sync_185;
                    float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, partial_14, 1);
                    partial_14 = partial_14 + _shfl_xor_sync_186;
                    dps[14] = partial_14;
                    float partial_15 = 0.0f;
                    float row_o_11[8];
                    float row_do_11[8];
                    int late_row_11 = q_row0 + 120 + warp * 2 + row_in_warp;
                    int late_base_11 = (late_row_11 * num_heads + head_idx_2) * HEAD_DIM + lane_col;
                    {
                        const uint4* _vptr_187 = reinterpret_cast<const uint4*>(O_gmem + late_base_11);
                        uint4 _vld_187[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_187[_blk] = _vptr_187[_blk];
                            uint32_t* _vpairs_187 = reinterpret_cast<uint32_t*>(&_vld_187[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_o_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_187[_pair]) << 16);
                                (&row_o_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_187[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    {
                        const uint4* _vptr_188 = reinterpret_cast<const uint4*>(dO_gmem + late_base_11);
                        uint4 _vld_188[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_188[_blk] = _vptr_188[_blk];
                            uint32_t* _vpairs_188 = reinterpret_cast<uint32_t*>(&_vld_188[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&row_do_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_188[_pair]) << 16);
                                (&row_do_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_188[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_31 = 0; j_31 < 8; j_31++) {
                        partial_15 = __fmaf_rn(row_o_11[j_31], row_do_11[j_31], partial_15);
                    }
                    float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, partial_15, 8);
                    partial_15 = partial_15 + _shfl_xor_sync_189;
                    float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, partial_15, 4);
                    partial_15 = partial_15 + _shfl_xor_sync_190;
                    float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, partial_15, 2);
                    partial_15 = partial_15 + _shfl_xor_sync_191;
                    float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, partial_15, 1);
                    partial_15 = partial_15 + _shfl_xor_sync_192;
                    dps[15] = partial_15;
                    mbarrier_wait(dpsum_empty_addr, _phase_dpsum_empty_0);
                    _phase_dpsum_empty_0 ^= 1;
                    if (lane_in_row == 0) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[0])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((8 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[1])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((16 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[2])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((24 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[3])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((32 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[4])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((40 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[5])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((48 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[6])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((56 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[7])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((64 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[8])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((72 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[9])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((80 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[10])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((88 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[11])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((96 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[12])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((104 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[13])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((112 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[14])) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dpsum_buf_addr + (unsigned int)((120 + warp * 2 + row_in_warp) * 4)), "r"(__float_as_uint(dps[15])) : "memory");
                    }
                    mbarrier_arrive(dpsum_full_addr);
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
