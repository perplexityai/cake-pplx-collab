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

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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




__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}



__device__ __forceinline__ void tcgen05_commit_cg2_multicast_leader(
    int mbar_addr, uint16_t cta_mask, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask), "r"(leader) : "memory");
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_fmha_train_cd2c9b670b9df03ff0b7(float* __restrict__ dQ_accum, float* __restrict__ LSE_log2_gmem, float* __restrict__ dPsum_gmem, const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap dO, const __grid_constant__ CUtensorMap dK_part, const __grid_constant__ CUtensorMap dV_part, int seqlen_q, int seqlen_kv, float softmax_scale_log2, int num_m_blocks, int num_heads)
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
    #define kv_full_addr (mbar_base + 0)
    #define kt_full_addr (mbar_base + 8)
    #define q_full_addr (mbar_base + 16)
    #define q_empty_addr (mbar_base + 24)
    #define qt_full_addr (mbar_base + 32)
    #define qt_empty_addr (mbar_base + 40)
    #define do_full_addr (mbar_base + 48)
    #define do_empty_addr (mbar_base + 56)
    #define lse_full_addr (mbar_base + 64)
    #define lse_empty_addr (mbar_base + 72)
    #define dpsum_full_addr (mbar_base + 80)
    #define dpsum_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 96)
    #define dp_full_addr (mbar_base + 104)
    #define sp_empty_addr (mbar_base + 112)
    #define dp_consumed_addr (mbar_base + 120)
    #define ds_full_addr (mbar_base + 128)
    #define ds_empty_addr (mbar_base + 136)
    #define xchg_full_addr (mbar_base + 144)
    #define xchg_done_addr (mbar_base + 152)
    #define dq_full_addr (mbar_base + 160)
    #define dq_empty_addr (mbar_base + 168)
    #define dv_done_addr (mbar_base + 176)
    #define dk_done_addr (mbar_base + 184)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sk_addr = smem + 1024;
    __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int sv_addr = smem + 33792;
    __nv_bfloat16* skt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int skt_addr = smem + 66560;
    __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int sq_addr = smem + 99328;
    __nv_bfloat16* sqt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int sqt_addr = smem + 115712;
    __nv_bfloat16* sdo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int sdo_addr = smem + 132096;
    __nv_bfloat16* sdot = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int sdot_addr = smem + 148480;
    __nv_bfloat16* sds = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int sds_addr = smem + 164864;
    __nv_bfloat16* sxchg = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int sxchg_addr = smem + 197632;
    __nv_bfloat16* dk_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int dk_staging_addr = smem + 1024;
    __nv_bfloat16* dv_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int dv_staging_addr = smem + 33792;
    float* dvp_staging = reinterpret_cast<float*>(smem_raw + 1024);
    const int dvp_staging_addr = smem + 1024;
    float* dkp_staging = reinterpret_cast<float*>(smem_raw + 99328);
    const int dkp_staging_addr = smem + 99328;
    float* sdq = reinterpret_cast<float*>(smem_raw + 214016);
    const int sdq_addr = smem + 214016;
    float* slse = reinterpret_cast<float*>(smem_raw + 230400);
    const int slse_addr = smem + 230400;
    float* sdps = reinterpret_cast<float*>(smem_raw + 230912);
    const int sdps_addr = smem + 230912;

    // Kernel setup ops
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dO))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dV_part))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dK_part))) : "memory"); }

    // Mbarrier init (24 barriers at smem_raw[0..192))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // kv_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // kt_full: 1 barriers, init_count=2
            mbarrier_init(smem + 8, 2);
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=2
            mbarrier_init(smem + 16, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'qt_pipe' ---
            // qt_full: 1 barriers, init_count=2
            mbarrier_init(smem + 32, 2);
            // qt_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'do_pipe' ---
            // do_full: 1 barriers, init_count=2
            mbarrier_init(smem + 48, 2);
            // do_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // lse_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // lse_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 72, 256);
            // dpsum_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // dpsum_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // dp_full: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // sp_empty: 1 barriers, init_count=512
            mbarrier_init(smem + 112, 512);
            // dp_consumed: 1 barriers, init_count=512
            mbarrier_init(smem + 120, 512);
            // ds_full: 1 barriers, init_count=512
            mbarrier_init(smem + 128, 512);
            // ds_empty: 1 barriers, init_count=2
            mbarrier_init(smem + 136, 2);
            // xchg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // xchg_done: 1 barriers, init_count=2
            mbarrier_init(smem + 152, 2);
            // dq_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // dq_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 168, 256);
            // dv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // dk_done: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 12) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_s = taddr;
    const unsigned int tmem_p = taddr;
    const unsigned int tmem_dq = taddr + 64;
    const unsigned int tmem_dv = taddr + 128;
    const unsigned int tmem_dp = taddr + 256;
    const unsigned int tmem_ds = taddr + 256;
    const unsigned int tmem_dk = taddr + 384;

    // Kernel post-init ops
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    }

    // ---- Role: reduce ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // reduce_main
            int rank = cta_rank & 1;
            int bh = blockIdx.z / 2 * num_heads + blockIdx.y;
            int m_start = 0;
            int n_trips = (num_m_blocks - blockIdx.z % 2 + 1) / 2;
            const int quarter = warp % 4;
            const int row64 = (quarter & 1) * 32 + lane;
            const int half = quarter >> 1;
            const unsigned int lane_base = (unsigned int)(quarter * 32 << 16);
            int t = 0;
            #pragma unroll 1
            for (int m = m_start; m < n_trips; m++) {
                int block = (bh * seqlen_q + (blockIdx.z % 2 + m * 2) * 128) * 128 + rank * 8192;
                int _parity_dq_full = (int)((unsigned int)t % 2);
                mbarrier_wait(dq_full_addr, _parity_dq_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float dq[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(dq[0]), "=f"(dq[1]), "=f"(dq[2]), "=f"(dq[3]), "=f"(dq[4]), "=f"(dq[5]), "=f"(dq[6]), "=f"(dq[7]), "=f"(dq[8]), "=f"(dq[9]), "=f"(dq[10]), "=f"(dq[11]), "=f"(dq[12]), "=f"(dq[13]), "=f"(dq[14]), "=f"(dq[15]), "=f"(dq[16]), "=f"(dq[17]), "=f"(dq[18]), "=f"(dq[19]), "=f"(dq[20]), "=f"(dq[21]), "=f"(dq[22]), "=f"(dq[23]), "=f"(dq[24]), "=f"(dq[25]), "=f"(dq[26]), "=f"(dq[27]), "=f"(dq[28]), "=f"(dq[29]), "=f"(dq[30]), "=f"(dq[31]), "=f"(dq[32]), "=f"(dq[33]), "=f"(dq[34]), "=f"(dq[35]), "=f"(dq[36]), "=f"(dq[37]), "=f"(dq[38]), "=f"(dq[39]), "=f"(dq[40]), "=f"(dq[41]), "=f"(dq[42]), "=f"(dq[43]), "=f"(dq[44]), "=f"(dq[45]), "=f"(dq[46]), "=f"(dq[47]), "=f"(dq[48]), "=f"(dq[49]), "=f"(dq[50]), "=f"(dq[51]), "=f"(dq[52]), "=f"(dq[53]), "=f"(dq[54]), "=f"(dq[55]), "=f"(dq[56]), "=f"(dq[57]), "=f"(dq[58]), "=f"(dq[59]), "=f"(dq[60]), "=f"(dq[61]), "=f"(dq[62]), "=f"(dq[63])
                    : "r"(taddr + 64 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
                    :: "r"((dq_empty_addr) & 0xFEFFFFFF) : "memory");
                unsigned int stage_base_0 = sdq_addr + (unsigned int)(half * 4096);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_0 + (unsigned int)(row64 * 16)), "f"(dq[0]), "f"(dq[1]), "f"(dq[2]), "f"(dq[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_0 + (unsigned int)(1024 + row64 * 16)), "f"(dq[4]), "f"(dq[5]), "f"(dq[6]), "f"(dq[7]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_0 + (unsigned int)(2048 + row64 * 16)), "f"(dq[8]), "f"(dq[9]), "f"(dq[10]), "f"(dq[11]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_0 + (unsigned int)(3072 + row64 * 16)), "f"(dq[12]), "f"(dq[13]), "f"(dq[14]), "f"(dq[15]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                if (quarter % 2 == 0) {
                    if (elect_sync()) {
                        {
                            void* _cpred_dst_0 = reinterpret_cast<void*>(dQ_accum + (block + half * 4 * 1024));
                            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                                " [%0], [%1], %2;"
                                :: "l"(_cpred_dst_0), "r"(stage_base_0), "r"((uint32_t)(4096))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                unsigned int stage_base_1 = sdq_addr + (unsigned int)(8192 + half * 4096);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_1 + (unsigned int)(row64 * 16)), "f"(dq[16]), "f"(dq[17]), "f"(dq[18]), "f"(dq[19]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_1 + (unsigned int)(1024 + row64 * 16)), "f"(dq[20]), "f"(dq[21]), "f"(dq[22]), "f"(dq[23]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_1 + (unsigned int)(2048 + row64 * 16)), "f"(dq[24]), "f"(dq[25]), "f"(dq[26]), "f"(dq[27]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_1 + (unsigned int)(3072 + row64 * 16)), "f"(dq[28]), "f"(dq[29]), "f"(dq[30]), "f"(dq[31]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                if (quarter % 2 == 0) {
                    if (elect_sync()) {
                        {
                            void* _cpred_dst_1 = reinterpret_cast<void*>(dQ_accum + (block + (half * 4 + 1) * 1024));
                            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                                " [%0], [%1], %2;"
                                :: "l"(_cpred_dst_1), "r"(stage_base_1), "r"((uint32_t)(4096))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                unsigned int stage_base_2 = sdq_addr + (unsigned int)(half * 4096);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_2 + (unsigned int)(row64 * 16)), "f"(dq[32]), "f"(dq[33]), "f"(dq[34]), "f"(dq[35]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_2 + (unsigned int)(1024 + row64 * 16)), "f"(dq[36]), "f"(dq[37]), "f"(dq[38]), "f"(dq[39]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_2 + (unsigned int)(2048 + row64 * 16)), "f"(dq[40]), "f"(dq[41]), "f"(dq[42]), "f"(dq[43]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_2 + (unsigned int)(3072 + row64 * 16)), "f"(dq[44]), "f"(dq[45]), "f"(dq[46]), "f"(dq[47]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                if (quarter % 2 == 0) {
                    if (elect_sync()) {
                        {
                            void* _cpred_dst_2 = reinterpret_cast<void*>(dQ_accum + (block + (half * 4 + 2) * 1024));
                            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                                " [%0], [%1], %2;"
                                :: "l"(_cpred_dst_2), "r"(stage_base_2), "r"((uint32_t)(4096))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                unsigned int stage_base_3 = sdq_addr + (unsigned int)(8192 + half * 4096);
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_3 + (unsigned int)(row64 * 16)), "f"(dq[48]), "f"(dq[49]), "f"(dq[50]), "f"(dq[51]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_3 + (unsigned int)(1024 + row64 * 16)), "f"(dq[52]), "f"(dq[53]), "f"(dq[54]), "f"(dq[55]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_3 + (unsigned int)(2048 + row64 * 16)), "f"(dq[56]), "f"(dq[57]), "f"(dq[58]), "f"(dq[59]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_base_3 + (unsigned int)(3072 + row64 * 16)), "f"(dq[60]), "f"(dq[61]), "f"(dq[62]), "f"(dq[63]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                if (quarter % 2 == 0) {
                    if (elect_sync()) {
                        {
                            void* _cpred_dst_3 = reinterpret_cast<void*>(dQ_accum + (block + (half * 4 + 3) * 1024));
                            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                                " [%0], [%1], %2;"
                                :: "l"(_cpred_dst_3), "r"(stage_base_3), "r"((uint32_t)(4096))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                t += 1;
            }
            if (quarter % 2 == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
        }
    }
    // ---- Role: compute ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
        { // compute_main
            int rank_1 = cta_rank & 1;
            int bh_1 = blockIdx.z / 2 * num_heads + blockIdx.y;
            int m_start_1 = 0;
            int n_trips_1 = (num_m_blocks - blockIdx.z % 2 + 1) / 2;
            int head_idx = blockIdx.y;
            int batch_idx = blockIdx.z;
            int n_block = blockIdx.x;
            int wg_idx = 1 - (warp - 4) / 4;
            int kv_row = (warp - 4) % 4 * 32 + lane;
            bool copy_warp = warp == 4;
            unsigned int lane_off = (unsigned int)((warp - 4) % 4 * 32 << 16);
            int t_1 = 0;
            int _parity_lse_full = (int)((unsigned int)t_1 % 2);
            mbarrier_wait(lse_full_addr, _parity_lse_full);
            int _parity_s_full = (int)((unsigned int)t_1 % 2);
            mbarrier_wait(s_full_addr, _parity_s_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float s0[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(s0[0]), "=f"(s0[1]), "=f"(s0[2]), "=f"(s0[3]), "=f"(s0[4]), "=f"(s0[5]), "=f"(s0[6]), "=f"(s0[7]), "=f"(s0[8]), "=f"(s0[9]), "=f"(s0[10]), "=f"(s0[11]), "=f"(s0[12]), "=f"(s0[13]), "=f"(s0[14]), "=f"(s0[15]), "=f"(s0[16]), "=f"(s0[17]), "=f"(s0[18]), "=f"(s0[19]), "=f"(s0[20]), "=f"(s0[21]), "=f"(s0[22]), "=f"(s0[23]), "=f"(s0[24]), "=f"(s0[25]), "=f"(s0[26]), "=f"(s0[27]), "=f"(s0[28]), "=f"(s0[29]), "=f"(s0[30]), "=f"(s0[31])
                : "r"(taddr + (unsigned int)(wg_idx * 32) + lane_off));
            float s1[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(s1[0]), "=f"(s1[1]), "=f"(s1[2]), "=f"(s1[3]), "=f"(s1[4]), "=f"(s1[5]), "=f"(s1[6]), "=f"(s1[7]), "=f"(s1[8]), "=f"(s1[9]), "=f"(s1[10]), "=f"(s1[11]), "=f"(s1[12]), "=f"(s1[13]), "=f"(s1[14]), "=f"(s1[15]), "=f"(s1[16]), "=f"(s1[17]), "=f"(s1[18]), "=f"(s1[19]), "=f"(s1[20]), "=f"(s1[21]), "=f"(s1[22]), "=f"(s1[23]), "=f"(s1[24]), "=f"(s1[25]), "=f"(s1[26]), "=f"(s1[27]), "=f"(s1[28]), "=f"(s1[29]), "=f"(s1[30]), "=f"(s1[31])
                : "r"(taddr + 64 + (unsigned int)(wg_idx * 32) + lane_off));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float lse0[32];
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                lse0[j] = slse[wg_idx * 32 + j];
            }
            float2 scale2 = make_float2(softmax_scale_log2, softmax_scale_log2);
            #pragma unroll
            for (int pair = 0; pair < 16; pair++) {
                float2 affine_pair = fma_sub_f32x2_rn_ftz(make_float2(s0[pair * 2], s0[pair * 2 + 1]), scale2, make_float2(lse0[pair * 2], lse0[pair * 2 + 1]));
                s0[pair * 2] = affine_pair.x;
                s0[pair * 2 + 1] = affine_pair.y;
            }
            unsigned int p0[16];
            softmax_frag_exp2_cast(&s0[0], p0, 0);
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 3, 256;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                :: "r"(taddr + (unsigned int)(wg_idx * 2 * 8) + lane_off), "r"(p0[0]), "r"(p0[1]), "r"(p0[2]), "r"(p0[3]), "r"(p0[4]), "r"(p0[5]), "r"(p0[6]), "r"(p0[7]), "r"(p0[8]), "r"(p0[9]), "r"(p0[10]), "r"(p0[11]), "r"(p0[12]), "r"(p0[13]), "r"(p0[14]), "r"(p0[15]));
            float lse1[32];
            #pragma unroll
            for (int j_1 = 0; j_1 < 32; j_1++) {
                lse1[j_1] = slse[wg_idx * 32 + 64 + j_1];
            }
            float2 scale2_0 = make_float2(softmax_scale_log2, softmax_scale_log2);
            #pragma unroll
            for (int pair_1 = 0; pair_1 < 16; pair_1++) {
                float2 affine_pair_1 = fma_sub_f32x2_rn_ftz(make_float2(s1[pair_1 * 2], s1[pair_1 * 2 + 1]), scale2_0, make_float2(lse1[pair_1 * 2], lse1[pair_1 * 2 + 1]));
                s1[pair_1 * 2] = affine_pair_1.x;
                s1[pair_1 * 2 + 1] = affine_pair_1.y;
            }
            unsigned int p1[16];
            softmax_frag_exp2_cast(&s1[0], p1, 0);
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                :: "r"(taddr + (unsigned int)((wg_idx * 2 + 4) * 8) + lane_off), "r"(p1[0]), "r"(p1[1]), "r"(p1[2]), "r"(p1[3]), "r"(p1[4]), "r"(p1[5]), "r"(p1[6]), "r"(p1[7]), "r"(p1[8]), "r"(p1[9]), "r"(p1[10]), "r"(p1[11]), "r"(p1[12]), "r"(p1[13]), "r"(p1[14]), "r"(p1[15]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"((sp_empty_addr) & 0xFEFFFFFF) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(lse_empty_addr);
            int _parity_dpsum_full = (int)((unsigned int)t_1 % 2);
            mbarrier_wait(dpsum_full_addr, _parity_dpsum_full);
            int _parity_dp_full = (int)((unsigned int)t_1 % 2);
            mbarrier_wait(dp_full_addr, _parity_dp_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float dp0[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(dp0[0]), "=f"(dp0[1]), "=f"(dp0[2]), "=f"(dp0[3]), "=f"(dp0[4]), "=f"(dp0[5]), "=f"(dp0[6]), "=f"(dp0[7]), "=f"(dp0[8]), "=f"(dp0[9]), "=f"(dp0[10]), "=f"(dp0[11]), "=f"(dp0[12]), "=f"(dp0[13]), "=f"(dp0[14]), "=f"(dp0[15]), "=f"(dp0[16]), "=f"(dp0[17]), "=f"(dp0[18]), "=f"(dp0[19]), "=f"(dp0[20]), "=f"(dp0[21]), "=f"(dp0[22]), "=f"(dp0[23]), "=f"(dp0[24]), "=f"(dp0[25]), "=f"(dp0[26]), "=f"(dp0[27]), "=f"(dp0[28]), "=f"(dp0[29]), "=f"(dp0[30]), "=f"(dp0[31])
                : "r"(taddr + 256 + (unsigned int)(wg_idx * 32) + lane_off));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 3, 256;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            float dpsum[32];
            #pragma unroll
            for (int j_2 = 0; j_2 < 32; j_2++) {
                dpsum[j_2] = sdps[wg_idx * 32 + j_2];
            }
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                sub_f32x2_inplace(&reinterpret_cast<float2*>(dp0)[_ls], reinterpret_cast<const float2*>(dpsum)[_ls]);
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dp0)[_ls], reinterpret_cast<const float2*>(s0)[_ls]);
            unsigned int ds0[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dp0[_lp*2 + 0], dp0[_lp*2+1 + 0]));
                ds0[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                :: "r"(taddr + 256 + (unsigned int)(wg_idx * 2 * 8) + lane_off), "r"(ds0[0]), "r"(ds0[1]), "r"(ds0[2]), "r"(ds0[3]), "r"(ds0[4]), "r"(ds0[5]), "r"(ds0[6]), "r"(ds0[7]), "r"(ds0[8]), "r"(ds0[9]), "r"(ds0[10]), "r"(ds0[11]), "r"(ds0[12]), "r"(ds0[13]), "r"(ds0[14]), "r"(ds0[15]));
            float dp1[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(dp1[0]), "=f"(dp1[1]), "=f"(dp1[2]), "=f"(dp1[3]), "=f"(dp1[4]), "=f"(dp1[5]), "=f"(dp1[6]), "=f"(dp1[7]), "=f"(dp1[8]), "=f"(dp1[9]), "=f"(dp1[10]), "=f"(dp1[11]), "=f"(dp1[12]), "=f"(dp1[13]), "=f"(dp1[14]), "=f"(dp1[15]), "=f"(dp1[16]), "=f"(dp1[17]), "=f"(dp1[18]), "=f"(dp1[19]), "=f"(dp1[20]), "=f"(dp1[21]), "=f"(dp1[22]), "=f"(dp1[23]), "=f"(dp1[24]), "=f"(dp1[25]), "=f"(dp1[26]), "=f"(dp1[27]), "=f"(dp1[28]), "=f"(dp1[29]), "=f"(dp1[30]), "=f"(dp1[31])
                : "r"(taddr + 320 + (unsigned int)(wg_idx * 32) + lane_off));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 3, 256;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            float dpsum_0[32];
            #pragma unroll
            for (int j_3 = 0; j_3 < 32; j_3++) {
                dpsum_0[j_3] = sdps[wg_idx * 32 + 64 + j_3];
            }
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                sub_f32x2_inplace(&reinterpret_cast<float2*>(dp1)[_ls], reinterpret_cast<const float2*>(dpsum_0)[_ls]);
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dp1)[_ls], reinterpret_cast<const float2*>(s1)[_ls]);
            unsigned int ds1[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dp1[_lp*2 + 0], dp1[_lp*2+1 + 0]));
                ds1[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                :: "r"(taddr + 256 + (unsigned int)((wg_idx * 2 + 4) * 8) + lane_off), "r"(ds1[0]), "r"(ds1[1]), "r"(ds1[2]), "r"(ds1[3]), "r"(ds1[4]), "r"(ds1[5]), "r"(ds1[6]), "r"(ds1[7]), "r"(ds1[8]), "r"(ds1[9]), "r"(ds1[10]), "r"(ds1[11]), "r"(ds1[12]), "r"(ds1[13]), "r"(ds1[14]), "r"(ds1[15]));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"((dp_consumed_addr) & 0xFEFFFFFF) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(dpsum_empty_addr);
            int _parity_ds_empty = (int)((unsigned int)t_1 % 2) ^ 1;
            mbarrier_wait_cluster_hint(ds_empty_addr, _parity_ds_empty, 10000000);
            if (rank_1 == 0) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds0[0]), "r"(ds0[1]), "r"(ds0[2]), "r"(ds0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 1 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds0[4]), "r"(ds0[5]), "r"(ds0[6]), "r"(ds0[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 2 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds0[8]), "r"(ds0[9]), "r"(ds0[10]), "r"(ds0[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 3 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds0[12]), "r"(ds0[13]), "r"(ds0[14]), "r"(ds0[15]) : "memory");
            } else {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 ^ kv_row & 7) * 16))), "r"(ds0[0]), "r"(ds0[1]), "r"(ds0[2]), "r"(ds0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 1 ^ kv_row & 7) * 16))), "r"(ds0[4]), "r"(ds0[5]), "r"(ds0[6]), "r"(ds0[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 2 ^ kv_row & 7) * 16))), "r"(ds0[8]), "r"(ds0[9]), "r"(ds0[10]), "r"(ds0[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 3 ^ kv_row & 7) * 16))), "r"(ds0[12]), "r"(ds0[13]), "r"(ds0[14]), "r"(ds0[15]) : "memory");
            }
            if (rank_1 == 1) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds1[0]), "r"(ds1[1]), "r"(ds1[2]), "r"(ds1[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 1 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds1[4]), "r"(ds1[5]), "r"(ds1[6]), "r"(ds1[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 2 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds1[8]), "r"(ds1[9]), "r"(ds1[10]), "r"(ds1[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 3 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds1[12]), "r"(ds1[13]), "r"(ds1[14]), "r"(ds1[15]) : "memory");
            } else {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 ^ kv_row & 7) * 16))), "r"(ds1[0]), "r"(ds1[1]), "r"(ds1[2]), "r"(ds1[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 1 ^ kv_row & 7) * 16))), "r"(ds1[4]), "r"(ds1[5]), "r"(ds1[6]), "r"(ds1[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 2 ^ kv_row & 7) * 16))), "r"(ds1[8]), "r"(ds1[9]), "r"(ds1[10]), "r"(ds1[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 3 ^ kv_row & 7) * 16))), "r"(ds1[12]), "r"(ds1[13]), "r"(ds1[14]), "r"(ds1[15]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 3, 256;" ::: "memory");
            if (copy_warp) {
                if (elect_sync()) {
                    asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                        "}"
                        :: "r"(xchg_full_addr), "r"(1 - rank_1), "r"((uint32_t)(16384)) : "memory");
                    uint32_t _mbar_peer_0;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_0) : "r"(xchg_full_addr), "r"(1 - rank_1));
                    uint32_t _cluster_mapa_1;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_1) : "r"(sds_addr + (unsigned int)(rank_1 * 2 * 8192)), "r"(1 - rank_1));
                    asm volatile(
                        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                        " [%0], [%1], %2, [%3];"
                        :: "r"(_cluster_mapa_1), "r"(sxchg_addr), "r"((uint32_t)(16384)), "r"(_mbar_peer_0)
                        : "memory");
                }
            }
            t_1 += 1;
            #pragma unroll 1
            for (int m_1 = m_start_1 + 1; m_1 < n_trips_1; m_1++) {
                int _parity_lse_full_1 = (int)((unsigned int)t_1 % 2);
                mbarrier_wait(lse_full_addr, _parity_lse_full_1);
                int _parity_s_full_1 = (int)((unsigned int)t_1 % 2);
                mbarrier_wait(s_full_addr, _parity_s_full_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float s0_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(s0_0[0]), "=f"(s0_0[1]), "=f"(s0_0[2]), "=f"(s0_0[3]), "=f"(s0_0[4]), "=f"(s0_0[5]), "=f"(s0_0[6]), "=f"(s0_0[7]), "=f"(s0_0[8]), "=f"(s0_0[9]), "=f"(s0_0[10]), "=f"(s0_0[11]), "=f"(s0_0[12]), "=f"(s0_0[13]), "=f"(s0_0[14]), "=f"(s0_0[15]), "=f"(s0_0[16]), "=f"(s0_0[17]), "=f"(s0_0[18]), "=f"(s0_0[19]), "=f"(s0_0[20]), "=f"(s0_0[21]), "=f"(s0_0[22]), "=f"(s0_0[23]), "=f"(s0_0[24]), "=f"(s0_0[25]), "=f"(s0_0[26]), "=f"(s0_0[27]), "=f"(s0_0[28]), "=f"(s0_0[29]), "=f"(s0_0[30]), "=f"(s0_0[31])
                    : "r"(taddr + (unsigned int)(wg_idx * 32) + lane_off));
                float s1_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(s1_0[0]), "=f"(s1_0[1]), "=f"(s1_0[2]), "=f"(s1_0[3]), "=f"(s1_0[4]), "=f"(s1_0[5]), "=f"(s1_0[6]), "=f"(s1_0[7]), "=f"(s1_0[8]), "=f"(s1_0[9]), "=f"(s1_0[10]), "=f"(s1_0[11]), "=f"(s1_0[12]), "=f"(s1_0[13]), "=f"(s1_0[14]), "=f"(s1_0[15]), "=f"(s1_0[16]), "=f"(s1_0[17]), "=f"(s1_0[18]), "=f"(s1_0[19]), "=f"(s1_0[20]), "=f"(s1_0[21]), "=f"(s1_0[22]), "=f"(s1_0[23]), "=f"(s1_0[24]), "=f"(s1_0[25]), "=f"(s1_0[26]), "=f"(s1_0[27]), "=f"(s1_0[28]), "=f"(s1_0[29]), "=f"(s1_0[30]), "=f"(s1_0[31])
                    : "r"(taddr + 64 + (unsigned int)(wg_idx * 32) + lane_off));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
                    :: "r"((ds_full_addr) & 0xFEFFFFFF) : "memory");
                float lse0_0[32];
                #pragma unroll
                for (int j_4 = 0; j_4 < 32; j_4++) {
                    lse0_0[j_4] = slse[wg_idx * 32 + j_4];
                }
                float2 scale2_1 = make_float2(softmax_scale_log2, softmax_scale_log2);
                #pragma unroll
                for (int pair_2 = 0; pair_2 < 16; pair_2++) {
                    float2 affine_pair_2 = fma_sub_f32x2_rn_ftz(make_float2(s0_0[pair_2 * 2], s0_0[pair_2 * 2 + 1]), scale2_1, make_float2(lse0_0[pair_2 * 2], lse0_0[pair_2 * 2 + 1]));
                    s0_0[pair_2 * 2] = affine_pair_2.x;
                    s0_0[pair_2 * 2 + 1] = affine_pair_2.y;
                }
                unsigned int p0_0[16];
                softmax_frag_exp2_cast(&s0_0[0], p0_0, 0);
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 3, 256;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)(wg_idx * 2 * 8) + lane_off), "r"(p0_0[0]), "r"(p0_0[1]), "r"(p0_0[2]), "r"(p0_0[3]), "r"(p0_0[4]), "r"(p0_0[5]), "r"(p0_0[6]), "r"(p0_0[7]), "r"(p0_0[8]), "r"(p0_0[9]), "r"(p0_0[10]), "r"(p0_0[11]), "r"(p0_0[12]), "r"(p0_0[13]), "r"(p0_0[14]), "r"(p0_0[15]));
                float lse1_0[32];
                #pragma unroll
                for (int j_5 = 0; j_5 < 32; j_5++) {
                    lse1_0[j_5] = slse[wg_idx * 32 + 64 + j_5];
                }
                float2 scale2_2 = make_float2(softmax_scale_log2, softmax_scale_log2);
                #pragma unroll
                for (int pair_3 = 0; pair_3 < 16; pair_3++) {
                    float2 affine_pair_3 = fma_sub_f32x2_rn_ftz(make_float2(s1_0[pair_3 * 2], s1_0[pair_3 * 2 + 1]), scale2_2, make_float2(lse1_0[pair_3 * 2], lse1_0[pair_3 * 2 + 1]));
                    s1_0[pair_3 * 2] = affine_pair_3.x;
                    s1_0[pair_3 * 2 + 1] = affine_pair_3.y;
                }
                unsigned int p1_0[16];
                softmax_frag_exp2_cast(&s1_0[0], p1_0, 0);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)((wg_idx * 2 + 4) * 8) + lane_off), "r"(p1_0[0]), "r"(p1_0[1]), "r"(p1_0[2]), "r"(p1_0[3]), "r"(p1_0[4]), "r"(p1_0[5]), "r"(p1_0[6]), "r"(p1_0[7]), "r"(p1_0[8]), "r"(p1_0[9]), "r"(p1_0[10]), "r"(p1_0[11]), "r"(p1_0[12]), "r"(p1_0[13]), "r"(p1_0[14]), "r"(p1_0[15]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((sp_empty_addr) & 0xFEFFFFFF) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(lse_empty_addr);
                int _parity_dpsum_full_1 = (int)((unsigned int)t_1 % 2);
                mbarrier_wait(dpsum_full_addr, _parity_dpsum_full_1);
                int _parity_dp_full_1 = (int)((unsigned int)t_1 % 2);
                mbarrier_wait(dp_full_addr, _parity_dp_full_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float dp0_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(dp0_0[0]), "=f"(dp0_0[1]), "=f"(dp0_0[2]), "=f"(dp0_0[3]), "=f"(dp0_0[4]), "=f"(dp0_0[5]), "=f"(dp0_0[6]), "=f"(dp0_0[7]), "=f"(dp0_0[8]), "=f"(dp0_0[9]), "=f"(dp0_0[10]), "=f"(dp0_0[11]), "=f"(dp0_0[12]), "=f"(dp0_0[13]), "=f"(dp0_0[14]), "=f"(dp0_0[15]), "=f"(dp0_0[16]), "=f"(dp0_0[17]), "=f"(dp0_0[18]), "=f"(dp0_0[19]), "=f"(dp0_0[20]), "=f"(dp0_0[21]), "=f"(dp0_0[22]), "=f"(dp0_0[23]), "=f"(dp0_0[24]), "=f"(dp0_0[25]), "=f"(dp0_0[26]), "=f"(dp0_0[27]), "=f"(dp0_0[28]), "=f"(dp0_0[29]), "=f"(dp0_0[30]), "=f"(dp0_0[31])
                    : "r"(taddr + 256 + (unsigned int)(wg_idx * 32) + lane_off));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 3, 256;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float dpsum_1[32];
                #pragma unroll
                for (int j_6 = 0; j_6 < 32; j_6++) {
                    dpsum_1[j_6] = sdps[wg_idx * 32 + j_6];
                }
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    sub_f32x2_inplace(&reinterpret_cast<float2*>(dp0_0)[_ls], reinterpret_cast<const float2*>(dpsum_1)[_ls]);
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dp0_0)[_ls], reinterpret_cast<const float2*>(s0_0)[_ls]);
                unsigned int ds0_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dp0_0[_lp*2 + 0], dp0_0[_lp*2+1 + 0]));
                    ds0_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + 256 + (unsigned int)(wg_idx * 2 * 8) + lane_off), "r"(ds0_0[0]), "r"(ds0_0[1]), "r"(ds0_0[2]), "r"(ds0_0[3]), "r"(ds0_0[4]), "r"(ds0_0[5]), "r"(ds0_0[6]), "r"(ds0_0[7]), "r"(ds0_0[8]), "r"(ds0_0[9]), "r"(ds0_0[10]), "r"(ds0_0[11]), "r"(ds0_0[12]), "r"(ds0_0[13]), "r"(ds0_0[14]), "r"(ds0_0[15]));
                float dp1_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(dp1_0[0]), "=f"(dp1_0[1]), "=f"(dp1_0[2]), "=f"(dp1_0[3]), "=f"(dp1_0[4]), "=f"(dp1_0[5]), "=f"(dp1_0[6]), "=f"(dp1_0[7]), "=f"(dp1_0[8]), "=f"(dp1_0[9]), "=f"(dp1_0[10]), "=f"(dp1_0[11]), "=f"(dp1_0[12]), "=f"(dp1_0[13]), "=f"(dp1_0[14]), "=f"(dp1_0[15]), "=f"(dp1_0[16]), "=f"(dp1_0[17]), "=f"(dp1_0[18]), "=f"(dp1_0[19]), "=f"(dp1_0[20]), "=f"(dp1_0[21]), "=f"(dp1_0[22]), "=f"(dp1_0[23]), "=f"(dp1_0[24]), "=f"(dp1_0[25]), "=f"(dp1_0[26]), "=f"(dp1_0[27]), "=f"(dp1_0[28]), "=f"(dp1_0[29]), "=f"(dp1_0[30]), "=f"(dp1_0[31])
                    : "r"(taddr + 320 + (unsigned int)(wg_idx * 32) + lane_off));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 3, 256;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float dpsum_2[32];
                #pragma unroll
                for (int j_7 = 0; j_7 < 32; j_7++) {
                    dpsum_2[j_7] = sdps[wg_idx * 32 + 64 + j_7];
                }
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    sub_f32x2_inplace(&reinterpret_cast<float2*>(dp1_0)[_ls], reinterpret_cast<const float2*>(dpsum_2)[_ls]);
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dp1_0)[_ls], reinterpret_cast<const float2*>(s1_0)[_ls]);
                unsigned int ds1_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dp1_0[_lp*2 + 0], dp1_0[_lp*2+1 + 0]));
                    ds1_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + 256 + (unsigned int)((wg_idx * 2 + 4) * 8) + lane_off), "r"(ds1_0[0]), "r"(ds1_0[1]), "r"(ds1_0[2]), "r"(ds1_0[3]), "r"(ds1_0[4]), "r"(ds1_0[5]), "r"(ds1_0[6]), "r"(ds1_0[7]), "r"(ds1_0[8]), "r"(ds1_0[9]), "r"(ds1_0[10]), "r"(ds1_0[11]), "r"(ds1_0[12]), "r"(ds1_0[13]), "r"(ds1_0[14]), "r"(ds1_0[15]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((dp_consumed_addr) & 0xFEFFFFFF) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(dpsum_empty_addr);
                int _parity_ds_empty_1 = (int)((unsigned int)t_1 % 2) ^ 1;
                mbarrier_wait_cluster_hint(ds_empty_addr, _parity_ds_empty_1, 10000000);
                if (rank_1 == 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds0_0[0]), "r"(ds0_0[1]), "r"(ds0_0[2]), "r"(ds0_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 1 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds0_0[4]), "r"(ds0_0[5]), "r"(ds0_0[6]), "r"(ds0_0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 2 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds0_0[8]), "r"(ds0_0[9]), "r"(ds0_0[10]), "r"(ds0_0[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 3 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds0_0[12]), "r"(ds0_0[13]), "r"(ds0_0[14]), "r"(ds0_0[15]) : "memory");
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 ^ kv_row & 7) * 16))), "r"(ds0_0[0]), "r"(ds0_0[1]), "r"(ds0_0[2]), "r"(ds0_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 1 ^ kv_row & 7) * 16))), "r"(ds0_0[4]), "r"(ds0_0[5]), "r"(ds0_0[6]), "r"(ds0_0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 2 ^ kv_row & 7) * 16))), "r"(ds0_0[8]), "r"(ds0_0[9]), "r"(ds0_0[10]), "r"(ds0_0[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 3 ^ kv_row & 7) * 16))), "r"(ds0_0[12]), "r"(ds0_0[13]), "r"(ds0_0[14]), "r"(ds0_0[15]) : "memory");
                }
                if (rank_1 == 1) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds1_0[0]), "r"(ds1_0[1]), "r"(ds1_0[2]), "r"(ds1_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 1 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds1_0[4]), "r"(ds1_0[5]), "r"(ds1_0[6]), "r"(ds1_0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 2 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds1_0[8]), "r"(ds1_0[9]), "r"(ds1_0[10]), "r"(ds1_0[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sds_addr + (unsigned int)((rank_1 * 128 + kv_row) * 128 + (wg_idx * 4 + 3 ^ rank_1 * 128 + kv_row & 7) * 16))), "r"(ds1_0[12]), "r"(ds1_0[13]), "r"(ds1_0[14]), "r"(ds1_0[15]) : "memory");
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 ^ kv_row & 7) * 16))), "r"(ds1_0[0]), "r"(ds1_0[1]), "r"(ds1_0[2]), "r"(ds1_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 1 ^ kv_row & 7) * 16))), "r"(ds1_0[4]), "r"(ds1_0[5]), "r"(ds1_0[6]), "r"(ds1_0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 2 ^ kv_row & 7) * 16))), "r"(ds1_0[8]), "r"(ds1_0[9]), "r"(ds1_0[10]), "r"(ds1_0[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sxchg_addr + (unsigned int)(kv_row * 128 + (wg_idx * 4 + 3 ^ kv_row & 7) * 16))), "r"(ds1_0[12]), "r"(ds1_0[13]), "r"(ds1_0[14]), "r"(ds1_0[15]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 3, 256;" ::: "memory");
                if (copy_warp) {
                    if (elect_sync()) {
                        asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(xchg_full_addr), "r"(1 - rank_1), "r"((uint32_t)(16384)) : "memory");
                        uint32_t _mbar_peer_2;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_2) : "r"(xchg_full_addr), "r"(1 - rank_1));
                        uint32_t _cluster_mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_3) : "r"(sds_addr + (unsigned int)(rank_1 * 2 * 8192)), "r"(1 - rank_1));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_cluster_mapa_3), "r"(sxchg_addr), "r"((uint32_t)(16384)), "r"(_mbar_peer_2)
                            : "memory");
                    }
                }
                t_1 += 1;
            }
            asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
            asm volatile(
                "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
                :: "r"((ds_full_addr) & 0xFEFFFFFF) : "memory");
            int _parity_ds_empty_2 = (int)((unsigned int)t_1 % 2) ^ 1;
            mbarrier_wait_cluster_hint(ds_empty_addr, _parity_ds_empty_2, 10000000);
            int off_kv = n_block * 128;
            unsigned int lane_base_1 = (unsigned int)((warp - 4) % 4 * 32 << 16);
            mbarrier_wait(dv_done_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float acc[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15]), "=f"(acc[16]), "=f"(acc[17]), "=f"(acc[18]), "=f"(acc[19]), "=f"(acc[20]), "=f"(acc[21]), "=f"(acc[22]), "=f"(acc[23]), "=f"(acc[24]), "=f"(acc[25]), "=f"(acc[26]), "=f"(acc[27]), "=f"(acc[28]), "=f"(acc[29]), "=f"(acc[30]), "=f"(acc[31]), "=f"(acc[32]), "=f"(acc[33]), "=f"(acc[34]), "=f"(acc[35]), "=f"(acc[36]), "=f"(acc[37]), "=f"(acc[38]), "=f"(acc[39]), "=f"(acc[40]), "=f"(acc[41]), "=f"(acc[42]), "=f"(acc[43]), "=f"(acc[44]), "=f"(acc[45]), "=f"(acc[46]), "=f"(acc[47]), "=f"(acc[48]), "=f"(acc[49]), "=f"(acc[50]), "=f"(acc[51]), "=f"(acc[52]), "=f"(acc[53]), "=f"(acc[54]), "=f"(acc[55]), "=f"(acc[56]), "=f"(acc[57]), "=f"(acc[58]), "=f"(acc[59]), "=f"(acc[60]), "=f"(acc[61]), "=f"(acc[62]), "=f"(acc[63])
                : "r"(taddr + (unsigned int)(128 + wg_idx * 64) + lane_base_1));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int stage_addr = dvp_staging_addr + (unsigned int)(wg_idx * 2 * 16384);
            #pragma unroll
            for (int vec_0 = 0; vec_0 < 8; vec_0++) {
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((stage_addr + (kv_row * 128 + vec_0 * 16 ^ (kv_row * 128 + vec_0 * 16 >> 7 & 7) << 4))), "f"(acc[vec_0 * 4]), "f"(acc[vec_0 * 4 + 1]), "f"(acc[vec_0 * 4 + 2]), "f"(acc[vec_0 * 4 + 3]) : "memory");
            }
            #pragma unroll
            for (int vec_1 = 0; vec_1 < 8; vec_1++) {
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((stage_addr + 16384 + (kv_row * 128 + vec_1 * 16 ^ (kv_row * 128 + vec_1 * 16 >> 7 & 7) << 4))), "f"(acc[32 + vec_1 * 4]), "f"(acc[32 + vec_1 * 4 + 1]), "f"(acc[32 + vec_1 * 4 + 2]), "f"(acc[32 + vec_1 * 4 + 3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (wg_idx == 0) {
                asm volatile("barrier.sync 1, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 2, 128;" ::: "memory");
            }
            if (warp % 4 == 0) {
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                        " [%0, {%1, %2, %3, %4}], [%5], %6;"
                        :: "l"((&dV_part)), "r"(wg_idx * 64), "r"(head_idx), "r"(off_kv), "r"(batch_idx), "r"(stage_addr), "l"(0ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                        " [%0, {%1, %2, %3, %4}], [%5], %6;"
                        :: "l"((&dV_part)), "r"(wg_idx * 64 + 32), "r"(head_idx), "r"(off_kv), "r"(batch_idx), "r"(stage_addr + 16384), "l"(0ULL) : "memory");
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
            mbarrier_wait(dk_done_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float acc_0[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                : "=f"(acc_0[0]), "=f"(acc_0[1]), "=f"(acc_0[2]), "=f"(acc_0[3]), "=f"(acc_0[4]), "=f"(acc_0[5]), "=f"(acc_0[6]), "=f"(acc_0[7]), "=f"(acc_0[8]), "=f"(acc_0[9]), "=f"(acc_0[10]), "=f"(acc_0[11]), "=f"(acc_0[12]), "=f"(acc_0[13]), "=f"(acc_0[14]), "=f"(acc_0[15]), "=f"(acc_0[16]), "=f"(acc_0[17]), "=f"(acc_0[18]), "=f"(acc_0[19]), "=f"(acc_0[20]), "=f"(acc_0[21]), "=f"(acc_0[22]), "=f"(acc_0[23]), "=f"(acc_0[24]), "=f"(acc_0[25]), "=f"(acc_0[26]), "=f"(acc_0[27]), "=f"(acc_0[28]), "=f"(acc_0[29]), "=f"(acc_0[30]), "=f"(acc_0[31]), "=f"(acc_0[32]), "=f"(acc_0[33]), "=f"(acc_0[34]), "=f"(acc_0[35]), "=f"(acc_0[36]), "=f"(acc_0[37]), "=f"(acc_0[38]), "=f"(acc_0[39]), "=f"(acc_0[40]), "=f"(acc_0[41]), "=f"(acc_0[42]), "=f"(acc_0[43]), "=f"(acc_0[44]), "=f"(acc_0[45]), "=f"(acc_0[46]), "=f"(acc_0[47]), "=f"(acc_0[48]), "=f"(acc_0[49]), "=f"(acc_0[50]), "=f"(acc_0[51]), "=f"(acc_0[52]), "=f"(acc_0[53]), "=f"(acc_0[54]), "=f"(acc_0[55]), "=f"(acc_0[56]), "=f"(acc_0[57]), "=f"(acc_0[58]), "=f"(acc_0[59]), "=f"(acc_0[60]), "=f"(acc_0[61]), "=f"(acc_0[62]), "=f"(acc_0[63])
                : "r"(taddr + (unsigned int)(384 + wg_idx * 64) + lane_base_1));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int stage_addr_0 = dkp_staging_addr + (unsigned int)(wg_idx * 2 * 16384);
            #pragma unroll
            for (int vec_2 = 0; vec_2 < 8; vec_2++) {
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((stage_addr_0 + (kv_row * 128 + vec_2 * 16 ^ (kv_row * 128 + vec_2 * 16 >> 7 & 7) << 4))), "f"(acc_0[vec_2 * 4]), "f"(acc_0[vec_2 * 4 + 1]), "f"(acc_0[vec_2 * 4 + 2]), "f"(acc_0[vec_2 * 4 + 3]) : "memory");
            }
            #pragma unroll
            for (int vec_3 = 0; vec_3 < 8; vec_3++) {
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((stage_addr_0 + 16384 + (kv_row * 128 + vec_3 * 16 ^ (kv_row * 128 + vec_3 * 16 >> 7 & 7) << 4))), "f"(acc_0[32 + vec_3 * 4]), "f"(acc_0[32 + vec_3 * 4 + 1]), "f"(acc_0[32 + vec_3 * 4 + 2]), "f"(acc_0[32 + vec_3 * 4 + 3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (wg_idx == 0) {
                asm volatile("barrier.sync 1, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 2, 128;" ::: "memory");
            }
            if (warp % 4 == 0) {
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                        " [%0, {%1, %2, %3, %4}], [%5], %6;"
                        :: "l"((&dK_part)), "r"(wg_idx * 64), "r"(head_idx), "r"(off_kv), "r"(batch_idx), "r"(stage_addr_0), "l"(0ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                        " [%0, {%1, %2, %3, %4}], [%5], %6;"
                        :: "l"((&dK_part)), "r"(wg_idx * 64 + 32), "r"(head_idx), "r"(off_kv), "r"(batch_idx), "r"(stage_addr_0 + 16384), "l"(0ULL) : "memory");
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int rank_2 = cta_rank & 1;
            int bh_2 = blockIdx.z / 2 * num_heads + blockIdx.y;
            int m_start_2 = 0;
            int n_trips_2 = (num_m_blocks - blockIdx.z % 2 + 1) / 2;
            int sc = 0;
            int kc = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_do_full = 0;
            unsigned int _phase_qt_full = 0;
            if ((cta_rank & 1) == 0) {
                mbarrier_wait_cluster(kv_full_addr, 0);
                mbarrier_wait_cluster(q_full_addr, _phase_q_full);
                int _parity_dq_empty = (int)((unsigned int)kc % 2) ^ 1;
                mbarrier_wait_cluster(dq_empty_addr, _parity_dq_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_0 = elect_sync();
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_1 = ((sk_addr) >> 4) & 0x7FFF;
                int _smem_desc_lo_2 = ((sq_addr) >> 4) & 0x7FFF;
#else
                int _smem_desc_lo_1 = ((sk_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_2 = ((sq_addr) >> 4) & 0x3FFF;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_1), "r"(_smem_desc_lo_2), "r"(tmem_s), "r"(0), "r"(_elected_0) : "memory");
                tcgen05_commit_cg2_multicast_leader(s_full_addr, (uint16_t)(3), _elected_0);
                tcgen05_commit_cg2_multicast_leader(q_empty_addr, (uint16_t)(3), _elected_0);
                _phase_q_full ^= 1;
                mbarrier_wait_cluster(do_full_addr, _phase_do_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_3 = elect_sync();
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_4 = ((sv_addr) >> 4) & 0x7FFF;
                int _smem_desc_lo_5 = ((sdo_addr) >> 4) & 0x7FFF;
#else
                int _smem_desc_lo_4 = ((sv_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_5 = ((sdo_addr) >> 4) & 0x3FFF;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_dp), "r"(0), "r"(_elected_3) : "memory");
                tcgen05_commit_cg2_multicast_leader(dp_full_addr, (uint16_t)(3), _elected_3);
                int _parity_sp_empty = (int)((unsigned int)sc % 2);
                mbarrier_wait_cluster(sp_empty_addr, _parity_sp_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_6 = elect_sync();
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_7 = (((sdot_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                int _smem_desc_lo_7 = (((sdot_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_dv), "r"(_smem_desc_lo_7), "r"(tmem_p), "r"(0), "r"(_elected_6) : "memory");
                tcgen05_commit_cg2_multicast_leader(do_empty_addr, (uint16_t)(3), _elected_6);
                _phase_do_full ^= 1;
                sc += 1;
                mbarrier_wait_cluster(kt_full_addr, 0);
                #pragma unroll 1
                for (int m_2 = m_start_2 + 1; m_2 < n_trips_2; m_2++) {
                    mbarrier_wait_cluster(q_full_addr, _phase_q_full);
                    int _parity_dq_empty_1 = (int)((unsigned int)kc % 2) ^ 1;
                    mbarrier_wait_cluster(dq_empty_addr, _parity_dq_empty_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_8 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_9 = ((sk_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_10 = ((sq_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_9 = ((sk_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_10 = ((sq_addr) >> 4) & 0x3FFF;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_9), "r"(_smem_desc_lo_10), "r"(tmem_s), "r"(0), "r"(_elected_8) : "memory");
                    tcgen05_commit_cg2_multicast_leader(s_full_addr, (uint16_t)(3), _elected_8);
                    tcgen05_commit_cg2_multicast_leader(q_empty_addr, (uint16_t)(3), _elected_8);
                    _phase_q_full ^= 1;
                    int first_dk = ((kc == 0) ? 1 : 0);
                    mbarrier_wait_cluster(qt_full_addr, _phase_qt_full);
                    int _parity_dp_consumed = (int)((unsigned int)kc % 2);
                    mbarrier_wait_cluster(dp_consumed_addr, _parity_dp_consumed);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_11 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_12 = (((sqt_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_12 = (((sqt_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_dk), "r"(_smem_desc_lo_12), "r"(tmem_ds), "r"(((first_dk) ? 0 : 1)), "r"(_elected_11) : "memory");
                    tcgen05_commit_cg2_multicast_leader(qt_empty_addr, (uint16_t)(3), _elected_11);
                    _phase_qt_full ^= 1;
                    mbarrier_wait_cluster(do_full_addr, _phase_do_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_13 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_14 = ((sv_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_15 = ((sdo_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_14 = ((sv_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_15 = ((sdo_addr) >> 4) & 0x3FFF;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"(tmem_dp), "r"(0), "r"(_elected_13) : "memory");
                    tcgen05_commit_cg2_multicast_leader(dp_full_addr, (uint16_t)(3), _elected_13);
                    int _parity_ds_full = (int)((unsigned int)kc % 2);
                    mbarrier_wait_cluster(ds_full_addr, _parity_ds_full);
                    int _parity_xchg_done = (int)((unsigned int)kc % 2);
                    mbarrier_wait_cluster(xchg_done_addr, _parity_xchg_done);
                    int _parity_ds_empty_3 = (int)((unsigned int)kc % 2) ^ 1;
                    mbarrier_wait_cluster(ds_empty_addr, _parity_ds_empty_3);
                    int _parity_dq_empty_2 = (int)((unsigned int)kc % 2) ^ 1;
                    mbarrier_wait_cluster(dq_empty_addr, _parity_dq_empty_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_16 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_17 = (((sds_addr) >> 4) & 0x7FFF) | 0x8000000;
                    int _smem_desc_lo_18 = (((skt_addr) >> 4) & 0x7FFF) | 0x8000000;
#else
                    int _smem_desc_lo_17 = (((sds_addr) >> 4) & 0x3FFF) | 0x8000000;
                    int _smem_desc_lo_18 = (((skt_addr) >> 4) & 0x3FFF) | 0x8000000;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136414352;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_17), "r"(_smem_desc_lo_18), "r"(tmem_dq), "r"(0), "r"(_elected_16) : "memory");
                    tcgen05_commit_cg2_multicast_leader(dq_full_addr, (uint16_t)(3), _elected_16);
                    tcgen05_commit_cg2_multicast_leader(ds_empty_addr, (uint16_t)(3), _elected_16);
                    kc += 1;
                    int _parity_sp_empty_1 = (int)((unsigned int)sc % 2);
                    mbarrier_wait_cluster(sp_empty_addr, _parity_sp_empty_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_19 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_20 = (((sdot_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_20 = (((sdot_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_dv), "r"(_smem_desc_lo_20), "r"(tmem_p), "r"(1), "r"(_elected_19) : "memory");
                    tcgen05_commit_cg2_multicast_leader(do_empty_addr, (uint16_t)(3), _elected_19);
                    _phase_do_full ^= 1;
                    sc += 1;
                }
                elect_commit_cg2_multicast(dv_done_addr, (uint16_t)(3));
                int first_dk_1 = ((kc == 0) ? 1 : 0);
                mbarrier_wait_cluster(qt_full_addr, _phase_qt_full);
                int _parity_dp_consumed_1 = (int)((unsigned int)kc % 2);
                mbarrier_wait_cluster(dp_consumed_addr, _parity_dp_consumed_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_21 = elect_sync();
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_22 = (((sqt_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                int _smem_desc_lo_22 = (((sqt_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_dk), "r"(_smem_desc_lo_22), "r"(tmem_ds), "r"(((first_dk_1) ? 0 : 1)), "r"(_elected_21) : "memory");
                tcgen05_commit_cg2_multicast_leader(qt_empty_addr, (uint16_t)(3), _elected_21);
                _phase_qt_full ^= 1;
                elect_commit_cg2_multicast(dk_done_addr, (uint16_t)(3));
                int _parity_ds_full_1 = (int)((unsigned int)kc % 2);
                mbarrier_wait_cluster(ds_full_addr, _parity_ds_full_1);
                int _parity_xchg_done_1 = (int)((unsigned int)kc % 2);
                mbarrier_wait_cluster(xchg_done_addr, _parity_xchg_done_1);
                int _parity_ds_empty_4 = (int)((unsigned int)kc % 2) ^ 1;
                mbarrier_wait_cluster(ds_empty_addr, _parity_ds_empty_4);
                int _parity_dq_empty_3 = (int)((unsigned int)kc % 2) ^ 1;
                mbarrier_wait_cluster(dq_empty_addr, _parity_dq_empty_3);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_23 = elect_sync();
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_24 = (((sds_addr) >> 4) & 0x7FFF) | 0x8000000;
                int _smem_desc_lo_25 = (((skt_addr) >> 4) & 0x7FFF) | 0x8000000;
#else
                int _smem_desc_lo_24 = (((sds_addr) >> 4) & 0x3FFF) | 0x8000000;
                int _smem_desc_lo_25 = (((skt_addr) >> 4) & 0x3FFF) | 0x8000000;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136414352;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_24), "r"(_smem_desc_lo_25), "r"(tmem_dq), "r"(0), "r"(_elected_23) : "memory");
                tcgen05_commit_cg2_multicast_leader(dq_full_addr, (uint16_t)(3), _elected_23);
                tcgen05_commit_cg2_multicast_leader(ds_empty_addr, (uint16_t)(3), _elected_23);
                kc += 1;
                int _parity_dq_empty_4 = (int)((unsigned int)kc % 2) ^ 1;
                mbarrier_wait_cluster(dq_empty_addr, _parity_dq_empty_4);
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            int rank_3 = cta_rank & 1;
            int bh_3 = blockIdx.z / 2 * num_heads + blockIdx.y;
            int m_start_3 = 0;
            int n_trips_3 = (num_m_blocks - blockIdx.z % 2 + 1) / 2;
            int head_idx_1 = blockIdx.y;
            int batch_idx_1 = blockIdx.z / 2;
            int off_kv_1 = blockIdx.x * 128;
            int pair_kv = (blockIdx.x >> 1) * 256;
            int stats_base = bh_3 * seqlen_q;
            int t_2 = 0;
            if (elect_sync()) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                    :: "r"((kv_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                tma_4d_gmem2smem_cta2(sk_addr, (&K), 0, off_kv_1, head_idx_1, batch_idx_1, ((kv_full_addr) & 0xFEFFFFFF));
                tma_4d_gmem2smem_cta2(sv_addr, (&V), 0, off_kv_1, head_idx_1, batch_idx_1, ((kv_full_addr) & 0xFEFFFFFF));
                tma_4d_gmem2smem_cta2(sk_addr + 16384, (&K), 64, off_kv_1, head_idx_1, batch_idx_1, ((kv_full_addr) & 0xFEFFFFFF));
                tma_4d_gmem2smem_cta2(sv_addr + 16384, (&V), 64, off_kv_1, head_idx_1, batch_idx_1, ((kv_full_addr) & 0xFEFFFFFF));
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                    :: "r"((kt_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                tma_4d_gmem2smem_cta2(skt_addr, (&K), rank_3 * 64, pair_kv, head_idx_1, batch_idx_1, ((kt_full_addr) & 0xFEFFFFFF));
                tma_4d_gmem2smem_cta2(skt_addr + 16384, (&K), rank_3 * 64, pair_kv + 128, head_idx_1, batch_idx_1, ((kt_full_addr) & 0xFEFFFFFF));
            }
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_do_empty = 1;
            unsigned int _phase_qt_empty = 1;
            #pragma unroll 1
            for (int m_3 = m_start_3; m_3 < n_trips_3; m_3++) {
                int off_q = (blockIdx.z % 2 + m_3 * 2) * 128;
                int stats_row = stats_base + off_q;
                mbarrier_wait(q_empty_addr, _phase_q_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    tma_4d_gmem2smem_cta2(sq_addr, (&Q), 0, off_q + rank_3 * 64, head_idx_1, batch_idx_1, ((q_full_addr) & 0xFEFFFFFF));
                    tma_4d_gmem2smem_cta2(sq_addr + 8192, (&Q), 64, off_q + rank_3 * 64, head_idx_1, batch_idx_1, ((q_full_addr) & 0xFEFFFFFF));
                }
                _phase_q_empty ^= 1;
                int _parity_lse_empty = (int)((unsigned int)t_2 % 2) ^ 1;
                mbarrier_wait(lse_empty_addr, _parity_lse_empty);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(lse_full_addr, 512);
                    cp_async_bulk_gmem2smem(slse_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(LSE_log2_gmem) + (unsigned long long)stats_row * 4ull), 512, lse_full_addr);
                }
                mbarrier_wait(do_empty_addr, _phase_do_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((do_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                    tma_4d_gmem2smem_cta2(sdo_addr, (&dO), 0, off_q + rank_3 * 64, head_idx_1, batch_idx_1, ((do_full_addr) & 0xFEFFFFFF));
                    tma_4d_gmem2smem_cta2(sdot_addr, (&dO), rank_3 * 64, off_q, head_idx_1, batch_idx_1, ((do_full_addr) & 0xFEFFFFFF));
                    tma_4d_gmem2smem_cta2(sdo_addr + 8192, (&dO), 64, off_q + rank_3 * 64, head_idx_1, batch_idx_1, ((do_full_addr) & 0xFEFFFFFF));
                    tma_4d_gmem2smem_cta2(sdot_addr + 8192, (&dO), rank_3 * 64, off_q + 64, head_idx_1, batch_idx_1, ((do_full_addr) & 0xFEFFFFFF));
                }
                _phase_do_empty ^= 1;
                int _parity_dpsum_empty = (int)((unsigned int)t_2 % 2) ^ 1;
                mbarrier_wait(dpsum_empty_addr, _parity_dpsum_empty);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(dpsum_full_addr, 512);
                    cp_async_bulk_gmem2smem(sdps_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dPsum_gmem) + (unsigned long long)stats_row * 4ull), 512, dpsum_full_addr);
                }
                mbarrier_wait(qt_empty_addr, _phase_qt_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((qt_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    tma_4d_gmem2smem_cta2(sqt_addr, (&Q), rank_3 * 64, off_q, head_idx_1, batch_idx_1, ((qt_full_addr) & 0xFEFFFFFF));
                    tma_4d_gmem2smem_cta2(sqt_addr + 8192, (&Q), rank_3 * 64, off_q + 64, head_idx_1, batch_idx_1, ((qt_full_addr) & 0xFEFFFFFF));
                }
                _phase_qt_empty ^= 1;
                t_2 += 1;
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty);
            mbarrier_wait(qt_empty_addr, _phase_qt_empty);
            mbarrier_wait(do_empty_addr, _phase_do_empty);
        }
    }
    // ---- Role: relay ----
    if (warp == 14) {
        { // relay_main
            int rank_4 = cta_rank & 1;
            int bh_4 = blockIdx.z / 2 * num_heads + blockIdx.y;
            int m_start_4 = 0;
            int n_trips_4 = (num_m_blocks - blockIdx.z % 2 + 1) / 2;
            int t_3 = 0;
            #pragma unroll 1
            for (int m_4 = m_start_4; m_4 < n_trips_4; m_4++) {
                int _parity_xchg_full = (int)((unsigned int)t_3 % 2);
                mbarrier_wait_cluster_hint(xchg_full_addr, _parity_xchg_full, 10000000);
                if (elect_sync()) {
                    asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
                        :: "r"((xchg_done_addr) & 0xFEFFFFFF) : "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(ds_empty_addr), "r"(1 - rank_4) : "memory");
                }
                t_3 += 1;
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 12) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
