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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}




__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
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




__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
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


#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
__device__ __forceinline__ float2 mul_nz_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}
#define CAKE_F32X2_MUL_NZ(a, b, nz) mul_nz_f32x2_rn_noftz(a, b)
#else
#define CAKE_F32X2_MUL_NZ(a, b, nz) fma_f32x2_rn_noftz(a, b, nz)
#endif








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



extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_dsa_h64_det_train_adfe34021dbfb2cffe71(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap q_rope, const __grid_constant__ CUtensorMap kv_latent, const __grid_constant__ CUtensorMap k_rope, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ lse, int* __restrict__ indices, int* __restrict__ topk_length, int num_queries, int num_heads, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int derive_length, float scale_log2)
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
    #define q_full_bar_addr (mbar_base + 0)
    #define q_copied_addr (mbar_base + 8)
    #define k_a_addr (mbar_base + 16)
    #define k_b_addr (mbar_base + 32)
    #define v_a_addr (mbar_base + 48)
    #define v_b_addr (mbar_base + 64)
    #define qk_part_addr (mbar_base + 80)
    #define qk_done_addr (mbar_base + 96)
    #define pv_a_addr (mbar_base + 112)
    #define pv_b_addr (mbar_base + 128)
    #define s_free_addr (mbar_base + 144)
    #define so_ready_addr (mbar_base + 160)
    #define o_free_addr (mbar_base + 176)
    #define valid_ready_addr (mbar_base + 184)
    #define valid_free_addr (mbar_base + 200)
    #define len_ready_addr (mbar_base + 216)
    #define len_free_addr (mbar_base + 232)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* q_full = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_full_addr = smem + 1024;
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int q_smem_addr = smem + 33792;
    __nv_bfloat16* prob = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int prob_addr = smem + 1024;
    __nv_bfloat16* k_slot = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int k_slot_addr = smem + 74752;
    __nv_bfloat16* v_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int v_buf_addr = smem + 148480;
    uint8_t* validity = reinterpret_cast<uint8_t*>(smem_raw + 214016);
    const int validity_addr = smem + 214016;
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + 214016);
    const int validity32_addr = smem + 214016;
    float* stats = reinterpret_cast<float*>(smem_raw + 214144);
    const int stats_addr = smem + 214144;
    int* len_cells = reinterpret_cast<int*>(smem_raw + 215680);
    const int len_cells_addr = smem + 215680;

    // Kernel setup ops
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_rope))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&k_rope))) : "memory"); }

    // Mbarrier init (31 barriers at smem_raw[0..248))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_copied: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // k_a: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // k_b: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // v_a: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // v_b: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // qk_part: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // qk_done: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // pv_a: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // pv_b: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // s_free: 2 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            // so_ready: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // o_free: 1 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            // valid_ready: 2 barriers, init_count=16
            mbarrier_init(smem + 184, 16);
            mbarrier_init(smem + 192, 16);
            // valid_free: 2 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            // len_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // len_free: 2 barriers, init_count=153
            mbarrier_init(smem + 232, 153);
            mbarrier_init(smem + 240, 153);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 248);
    if (warp == 0) {
        int _tmem_hold = smem + 248;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem = taddr;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int cid = blockIdx.x / 2;
            int nclu = (int)gridDim.x / 2;
            int ntok = (num_queries - cid + nclu - 1) / nclu;
            int base = 0;
            int row = warp * 32 + lane;
            int kh = row / 64;
            int hrow = cta_rank * 64 + row % 64;
            int live = hrow < num_heads;
            unsigned int tmrow = tmem_tmem + (unsigned int)(warp % 4 * 32 << 16);
            if (ntok > 0) {
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(q_full_addr), "l"((&q_latent)), "r"(0), "r"(cta_rank * 64), "r"(0), "r"(cid),
                               "r"(((q_full_bar_addr) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(q_full_addr + (unsigned int)(8 * 8192)), "l"((&q_rope)), "r"(0), "r"(cta_rank * 64), "r"(0), "r"(cid),
                               "r"(((q_full_bar_addr) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            #pragma unroll 1
            for (int it = 0; it < ntok; it++) {
                int query = cid + it * nclu;
                int _stage_len_ready = (int)((unsigned int)it % 2);
                int _parity_len_ready = (int)((unsigned int)it / 2 % 2);
                mbarrier_wait(len_ready_addr + (_stage_len_ready) * 8, _parity_len_ready);
                int _max_0 = ((len_cells[it % 2]) > (0) ? (len_cells[it % 2]) : (0));
                int active = ((_max_0) < (topk) ? (_max_0) : (topk));
                mbarrier_arrive(len_free_addr + ((int)((unsigned int)it % 2)) * 8);
                int tiles = (((active + 127) / 128) > (1) ? ((active + 127) / 128) : (1));
                int _parity_q_copied = (int)((unsigned int)it % 2);
                mbarrier_wait(q_copied_addr, _parity_q_copied);
                float mi = -1e+30f;
                float li = 0.0f;
                float real_mi = -CUDART_INF_F;
                #pragma unroll 1
                for (int n = 0; n < tiles; n++) {
                    int _stage_qk_done = (int)((unsigned int)(base + n) % 2);
                    int _parity_qk_done = (int)((unsigned int)(base + n) / 2 % 2);
                    mbarrier_wait(qk_done_addr + (_stage_qk_done) * 8, _parity_qk_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float p[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]), "=f"(p[4]), "=f"(p[5]), "=f"(p[6]), "=f"(p[7]), "=f"(p[8]), "=f"(p[9]), "=f"(p[10]), "=f"(p[11]), "=f"(p[12]), "=f"(p[13]), "=f"(p[14]), "=f"(p[15]), "=f"(p[16]), "=f"(p[17]), "=f"(p[18]), "=f"(p[19]), "=f"(p[20]), "=f"(p[21]), "=f"(p[22]), "=f"(p[23]), "=f"(p[24]), "=f"(p[25]), "=f"(p[26]), "=f"(p[27]), "=f"(p[28]), "=f"(p[29]), "=f"(p[30]), "=f"(p[31]), "=f"(p[32]), "=f"(p[33]), "=f"(p[34]), "=f"(p[35]), "=f"(p[36]), "=f"(p[37]), "=f"(p[38]), "=f"(p[39]), "=f"(p[40]), "=f"(p[41]), "=f"(p[42]), "=f"(p[43]), "=f"(p[44]), "=f"(p[45]), "=f"(p[46]), "=f"(p[47]), "=f"(p[48]), "=f"(p[49]), "=f"(p[50]), "=f"(p[51]), "=f"(p[52]), "=f"(p[53]), "=f"(p[54]), "=f"(p[55]), "=f"(p[56]), "=f"(p[57]), "=f"(p[58]), "=f"(p[59]), "=f"(p[60]), "=f"(p[61]), "=f"(p[62]), "=f"(p[63])
                        : "r"(tmrow + (unsigned int)(256 + (base + n) % 2 * 64)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_free_addr + ((int)((unsigned int)(base + n) % 2)) * 8), "r"(0) : "memory");
                    int _stage_valid_ready = (int)((unsigned int)(base + n) % 2);
                    int _parity_valid_ready = (int)((unsigned int)(base + n) / 2 % 2);
                    mbarrier_wait(valid_ready_addr + (_stage_valid_ready) * 8, _parity_valid_ready);
                    unsigned int mask_lo = validity32[(base + n) % 2 * 4 + kh * 2];
                    unsigned int mask_hi = validity32[(base + n) % 2 * 4 + kh * 2 + 1];
                    mbarrier_arrive(valid_free_addr + ((int)((unsigned int)(base + n) % 2)) * 8);
                    if ((mask_lo & 1) == 0) {
                        p[0] = -CUDART_INF_F;
                    }
                    if ((mask_hi & 1) == 0) {
                        p[32] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 1 & 1) == 0) {
                        p[1] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 1 & 1) == 0) {
                        p[33] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 2 & 1) == 0) {
                        p[2] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 2 & 1) == 0) {
                        p[34] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 3 & 1) == 0) {
                        p[3] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 3 & 1) == 0) {
                        p[35] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 4 & 1) == 0) {
                        p[4] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 4 & 1) == 0) {
                        p[36] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 5 & 1) == 0) {
                        p[5] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 5 & 1) == 0) {
                        p[37] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 6 & 1) == 0) {
                        p[6] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 6 & 1) == 0) {
                        p[38] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 7 & 1) == 0) {
                        p[7] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 7 & 1) == 0) {
                        p[39] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 8 & 1) == 0) {
                        p[8] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 8 & 1) == 0) {
                        p[40] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 9 & 1) == 0) {
                        p[9] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 9 & 1) == 0) {
                        p[41] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 10 & 1) == 0) {
                        p[10] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 10 & 1) == 0) {
                        p[42] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 11 & 1) == 0) {
                        p[11] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 11 & 1) == 0) {
                        p[43] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 12 & 1) == 0) {
                        p[12] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 12 & 1) == 0) {
                        p[44] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 13 & 1) == 0) {
                        p[13] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 13 & 1) == 0) {
                        p[45] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 14 & 1) == 0) {
                        p[14] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 14 & 1) == 0) {
                        p[46] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 15 & 1) == 0) {
                        p[15] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 15 & 1) == 0) {
                        p[47] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 16 & 1) == 0) {
                        p[16] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 16 & 1) == 0) {
                        p[48] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 17 & 1) == 0) {
                        p[17] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 17 & 1) == 0) {
                        p[49] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 18 & 1) == 0) {
                        p[18] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 18 & 1) == 0) {
                        p[50] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 19 & 1) == 0) {
                        p[19] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 19 & 1) == 0) {
                        p[51] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 20 & 1) == 0) {
                        p[20] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 20 & 1) == 0) {
                        p[52] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 21 & 1) == 0) {
                        p[21] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 21 & 1) == 0) {
                        p[53] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 22 & 1) == 0) {
                        p[22] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 22 & 1) == 0) {
                        p[54] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 23 & 1) == 0) {
                        p[23] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 23 & 1) == 0) {
                        p[55] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 24 & 1) == 0) {
                        p[24] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 24 & 1) == 0) {
                        p[56] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 25 & 1) == 0) {
                        p[25] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 25 & 1) == 0) {
                        p[57] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 26 & 1) == 0) {
                        p[26] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 26 & 1) == 0) {
                        p[58] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 27 & 1) == 0) {
                        p[27] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 27 & 1) == 0) {
                        p[59] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 28 & 1) == 0) {
                        p[28] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 28 & 1) == 0) {
                        p[60] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 29 & 1) == 0) {
                        p[29] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 29 & 1) == 0) {
                        p[61] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 30 & 1) == 0) {
                        p[30] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 30 & 1) == 0) {
                        p[62] = -CUDART_INF_F;
                    }
                    if ((mask_lo >> 31 & 1) == 0) {
                        p[31] = -CUDART_INF_F;
                    }
                    if ((mask_hi >> 31 & 1) == 0) {
                        p[63] = -CUDART_INF_F;
                    }
                    float local_max0 = -CUDART_INF_F;
                    float local_max1 = -CUDART_INF_F;
                    local_max0 = max_noftz(local_max0, p[0]);
                    local_max0 = max_noftz(local_max0, p[1]);
                    local_max0 = max_noftz(local_max0, p[2]);
                    local_max0 = max_noftz(local_max0, p[3]);
                    local_max0 = max_noftz(local_max0, p[4]);
                    local_max0 = max_noftz(local_max0, p[5]);
                    local_max0 = max_noftz(local_max0, p[6]);
                    local_max0 = max_noftz(local_max0, p[7]);
                    local_max0 = max_noftz(local_max0, p[8]);
                    local_max0 = max_noftz(local_max0, p[9]);
                    local_max0 = max_noftz(local_max0, p[10]);
                    local_max0 = max_noftz(local_max0, p[11]);
                    local_max0 = max_noftz(local_max0, p[12]);
                    local_max0 = max_noftz(local_max0, p[13]);
                    local_max0 = max_noftz(local_max0, p[14]);
                    local_max0 = max_noftz(local_max0, p[15]);
                    local_max0 = max_noftz(local_max0, p[16]);
                    local_max0 = max_noftz(local_max0, p[17]);
                    local_max0 = max_noftz(local_max0, p[18]);
                    local_max0 = max_noftz(local_max0, p[19]);
                    local_max0 = max_noftz(local_max0, p[20]);
                    local_max0 = max_noftz(local_max0, p[21]);
                    local_max0 = max_noftz(local_max0, p[22]);
                    local_max0 = max_noftz(local_max0, p[23]);
                    local_max0 = max_noftz(local_max0, p[24]);
                    local_max0 = max_noftz(local_max0, p[25]);
                    local_max0 = max_noftz(local_max0, p[26]);
                    local_max0 = max_noftz(local_max0, p[27]);
                    local_max0 = max_noftz(local_max0, p[28]);
                    local_max0 = max_noftz(local_max0, p[29]);
                    local_max0 = max_noftz(local_max0, p[30]);
                    local_max0 = max_noftz(local_max0, p[31]);
                    local_max1 = max_noftz(local_max1, p[32]);
                    local_max1 = max_noftz(local_max1, p[33]);
                    local_max1 = max_noftz(local_max1, p[34]);
                    local_max1 = max_noftz(local_max1, p[35]);
                    local_max1 = max_noftz(local_max1, p[36]);
                    local_max1 = max_noftz(local_max1, p[37]);
                    local_max1 = max_noftz(local_max1, p[38]);
                    local_max1 = max_noftz(local_max1, p[39]);
                    local_max1 = max_noftz(local_max1, p[40]);
                    local_max1 = max_noftz(local_max1, p[41]);
                    local_max1 = max_noftz(local_max1, p[42]);
                    local_max1 = max_noftz(local_max1, p[43]);
                    local_max1 = max_noftz(local_max1, p[44]);
                    local_max1 = max_noftz(local_max1, p[45]);
                    local_max1 = max_noftz(local_max1, p[46]);
                    local_max1 = max_noftz(local_max1, p[47]);
                    local_max1 = max_noftz(local_max1, p[48]);
                    local_max1 = max_noftz(local_max1, p[49]);
                    local_max1 = max_noftz(local_max1, p[50]);
                    local_max1 = max_noftz(local_max1, p[51]);
                    local_max1 = max_noftz(local_max1, p[52]);
                    local_max1 = max_noftz(local_max1, p[53]);
                    local_max1 = max_noftz(local_max1, p[54]);
                    local_max1 = max_noftz(local_max1, p[55]);
                    local_max1 = max_noftz(local_max1, p[56]);
                    local_max1 = max_noftz(local_max1, p[57]);
                    local_max1 = max_noftz(local_max1, p[58]);
                    local_max1 = max_noftz(local_max1, p[59]);
                    local_max1 = max_noftz(local_max1, p[60]);
                    local_max1 = max_noftz(local_max1, p[61]);
                    local_max1 = max_noftz(local_max1, p[62]);
                    local_max1 = max_noftz(local_max1, p[63]);
                    float cur_max = max_noftz(local_max0, local_max1) * scale_log2;
                    stats[(base + n) % 2 * 128 + row] = cur_max;
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    cur_max = max_noftz(cur_max, stats[(base + n) % 2 * 128 + (row ^ 64)]);
                    real_mi = max_noftz(real_mi, cur_max);
                    int rescale = __any_sync(0xFFFFFFFF, cur_max - mi > 6.0f);
                    float factor = 1.0f;
                    float next_max = mi;
                    if (rescale != 0) {
                        next_max = max_noftz(cur_max, mi);
                        float _exp2_1 = approx_exp2(mi - next_max);
                        factor = _exp2_1;
                    }
                    mi = next_max;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_0 = make_float2(p[0], p[1]);
                    float2 _f32x2_mul_b_scaled_0 = make_float2(scale_log2, scale_log2);
                    float2 scaled_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_0));
                    #else
                    float2 scaled_0 = CAKE_F32X2_MUL_NZ(make_float2(p[0], p[1]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[0] = scaled_0.x;
                    p[1] = scaled_0.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_1 = make_float2(p[2], p[3]);
                    float2 _f32x2_mul_b_scaled_1 = make_float2(scale_log2, scale_log2);
                    float2 scaled_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_1));
                    #else
                    float2 scaled_1 = CAKE_F32X2_MUL_NZ(make_float2(p[2], p[3]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[2] = scaled_1.x;
                    p[3] = scaled_1.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_2 = make_float2(p[4], p[5]);
                    float2 _f32x2_mul_b_scaled_2 = make_float2(scale_log2, scale_log2);
                    float2 scaled_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_2));
                    #else
                    float2 scaled_2 = CAKE_F32X2_MUL_NZ(make_float2(p[4], p[5]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[4] = scaled_2.x;
                    p[5] = scaled_2.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_3 = make_float2(p[6], p[7]);
                    float2 _f32x2_mul_b_scaled_3 = make_float2(scale_log2, scale_log2);
                    float2 scaled_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_3));
                    #else
                    float2 scaled_3 = CAKE_F32X2_MUL_NZ(make_float2(p[6], p[7]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[6] = scaled_3.x;
                    p[7] = scaled_3.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_4 = make_float2(p[8], p[9]);
                    float2 _f32x2_mul_b_scaled_4 = make_float2(scale_log2, scale_log2);
                    float2 scaled_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_4));
                    #else
                    float2 scaled_4 = CAKE_F32X2_MUL_NZ(make_float2(p[8], p[9]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[8] = scaled_4.x;
                    p[9] = scaled_4.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_5 = make_float2(p[10], p[11]);
                    float2 _f32x2_mul_b_scaled_5 = make_float2(scale_log2, scale_log2);
                    float2 scaled_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_5));
                    #else
                    float2 scaled_5 = CAKE_F32X2_MUL_NZ(make_float2(p[10], p[11]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[10] = scaled_5.x;
                    p[11] = scaled_5.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_6 = make_float2(p[12], p[13]);
                    float2 _f32x2_mul_b_scaled_6 = make_float2(scale_log2, scale_log2);
                    float2 scaled_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_6));
                    #else
                    float2 scaled_6 = CAKE_F32X2_MUL_NZ(make_float2(p[12], p[13]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[12] = scaled_6.x;
                    p[13] = scaled_6.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_7 = make_float2(p[14], p[15]);
                    float2 _f32x2_mul_b_scaled_7 = make_float2(scale_log2, scale_log2);
                    float2 scaled_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_7));
                    #else
                    float2 scaled_7 = CAKE_F32X2_MUL_NZ(make_float2(p[14], p[15]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[14] = scaled_7.x;
                    p[15] = scaled_7.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_8 = make_float2(p[16], p[17]);
                    float2 _f32x2_mul_b_scaled_8 = make_float2(scale_log2, scale_log2);
                    float2 scaled_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_8));
                    #else
                    float2 scaled_8 = CAKE_F32X2_MUL_NZ(make_float2(p[16], p[17]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[16] = scaled_8.x;
                    p[17] = scaled_8.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_9 = make_float2(p[18], p[19]);
                    float2 _f32x2_mul_b_scaled_9 = make_float2(scale_log2, scale_log2);
                    float2 scaled_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_9));
                    #else
                    float2 scaled_9 = CAKE_F32X2_MUL_NZ(make_float2(p[18], p[19]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[18] = scaled_9.x;
                    p[19] = scaled_9.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_10 = make_float2(p[20], p[21]);
                    float2 _f32x2_mul_b_scaled_10 = make_float2(scale_log2, scale_log2);
                    float2 scaled_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_10));
                    #else
                    float2 scaled_10 = CAKE_F32X2_MUL_NZ(make_float2(p[20], p[21]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[20] = scaled_10.x;
                    p[21] = scaled_10.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_11 = make_float2(p[22], p[23]);
                    float2 _f32x2_mul_b_scaled_11 = make_float2(scale_log2, scale_log2);
                    float2 scaled_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_11));
                    #else
                    float2 scaled_11 = CAKE_F32X2_MUL_NZ(make_float2(p[22], p[23]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[22] = scaled_11.x;
                    p[23] = scaled_11.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_12 = make_float2(p[24], p[25]);
                    float2 _f32x2_mul_b_scaled_12 = make_float2(scale_log2, scale_log2);
                    float2 scaled_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_12));
                    #else
                    float2 scaled_12 = CAKE_F32X2_MUL_NZ(make_float2(p[24], p[25]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[24] = scaled_12.x;
                    p[25] = scaled_12.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_13 = make_float2(p[26], p[27]);
                    float2 _f32x2_mul_b_scaled_13 = make_float2(scale_log2, scale_log2);
                    float2 scaled_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_13));
                    #else
                    float2 scaled_13 = CAKE_F32X2_MUL_NZ(make_float2(p[26], p[27]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[26] = scaled_13.x;
                    p[27] = scaled_13.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_14 = make_float2(p[28], p[29]);
                    float2 _f32x2_mul_b_scaled_14 = make_float2(scale_log2, scale_log2);
                    float2 scaled_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_14));
                    #else
                    float2 scaled_14 = CAKE_F32X2_MUL_NZ(make_float2(p[28], p[29]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[28] = scaled_14.x;
                    p[29] = scaled_14.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_15 = make_float2(p[30], p[31]);
                    float2 _f32x2_mul_b_scaled_15 = make_float2(scale_log2, scale_log2);
                    float2 scaled_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_15));
                    #else
                    float2 scaled_15 = CAKE_F32X2_MUL_NZ(make_float2(p[30], p[31]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[30] = scaled_15.x;
                    p[31] = scaled_15.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_16 = make_float2(p[32], p[33]);
                    float2 _f32x2_mul_b_scaled_16 = make_float2(scale_log2, scale_log2);
                    float2 scaled_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_16));
                    #else
                    float2 scaled_16 = CAKE_F32X2_MUL_NZ(make_float2(p[32], p[33]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[32] = scaled_16.x;
                    p[33] = scaled_16.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_17 = make_float2(p[34], p[35]);
                    float2 _f32x2_mul_b_scaled_17 = make_float2(scale_log2, scale_log2);
                    float2 scaled_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_17));
                    #else
                    float2 scaled_17 = CAKE_F32X2_MUL_NZ(make_float2(p[34], p[35]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[34] = scaled_17.x;
                    p[35] = scaled_17.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_18 = make_float2(p[36], p[37]);
                    float2 _f32x2_mul_b_scaled_18 = make_float2(scale_log2, scale_log2);
                    float2 scaled_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_18));
                    #else
                    float2 scaled_18 = CAKE_F32X2_MUL_NZ(make_float2(p[36], p[37]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[36] = scaled_18.x;
                    p[37] = scaled_18.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_19 = make_float2(p[38], p[39]);
                    float2 _f32x2_mul_b_scaled_19 = make_float2(scale_log2, scale_log2);
                    float2 scaled_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_19));
                    #else
                    float2 scaled_19 = CAKE_F32X2_MUL_NZ(make_float2(p[38], p[39]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[38] = scaled_19.x;
                    p[39] = scaled_19.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_20 = make_float2(p[40], p[41]);
                    float2 _f32x2_mul_b_scaled_20 = make_float2(scale_log2, scale_log2);
                    float2 scaled_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_20));
                    #else
                    float2 scaled_20 = CAKE_F32X2_MUL_NZ(make_float2(p[40], p[41]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[40] = scaled_20.x;
                    p[41] = scaled_20.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_21 = make_float2(p[42], p[43]);
                    float2 _f32x2_mul_b_scaled_21 = make_float2(scale_log2, scale_log2);
                    float2 scaled_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_21));
                    #else
                    float2 scaled_21 = CAKE_F32X2_MUL_NZ(make_float2(p[42], p[43]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[42] = scaled_21.x;
                    p[43] = scaled_21.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_22 = make_float2(p[44], p[45]);
                    float2 _f32x2_mul_b_scaled_22 = make_float2(scale_log2, scale_log2);
                    float2 scaled_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_22));
                    #else
                    float2 scaled_22 = CAKE_F32X2_MUL_NZ(make_float2(p[44], p[45]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[44] = scaled_22.x;
                    p[45] = scaled_22.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_23 = make_float2(p[46], p[47]);
                    float2 _f32x2_mul_b_scaled_23 = make_float2(scale_log2, scale_log2);
                    float2 scaled_23;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_23));
                    #else
                    float2 scaled_23 = CAKE_F32X2_MUL_NZ(make_float2(p[46], p[47]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[46] = scaled_23.x;
                    p[47] = scaled_23.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_24 = make_float2(p[48], p[49]);
                    float2 _f32x2_mul_b_scaled_24 = make_float2(scale_log2, scale_log2);
                    float2 scaled_24;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_24));
                    #else
                    float2 scaled_24 = CAKE_F32X2_MUL_NZ(make_float2(p[48], p[49]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[48] = scaled_24.x;
                    p[49] = scaled_24.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_25 = make_float2(p[50], p[51]);
                    float2 _f32x2_mul_b_scaled_25 = make_float2(scale_log2, scale_log2);
                    float2 scaled_25;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_25));
                    #else
                    float2 scaled_25 = CAKE_F32X2_MUL_NZ(make_float2(p[50], p[51]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[50] = scaled_25.x;
                    p[51] = scaled_25.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_26 = make_float2(p[52], p[53]);
                    float2 _f32x2_mul_b_scaled_26 = make_float2(scale_log2, scale_log2);
                    float2 scaled_26;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_26));
                    #else
                    float2 scaled_26 = CAKE_F32X2_MUL_NZ(make_float2(p[52], p[53]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[52] = scaled_26.x;
                    p[53] = scaled_26.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_27 = make_float2(p[54], p[55]);
                    float2 _f32x2_mul_b_scaled_27 = make_float2(scale_log2, scale_log2);
                    float2 scaled_27;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_27));
                    #else
                    float2 scaled_27 = CAKE_F32X2_MUL_NZ(make_float2(p[54], p[55]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[54] = scaled_27.x;
                    p[55] = scaled_27.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_28 = make_float2(p[56], p[57]);
                    float2 _f32x2_mul_b_scaled_28 = make_float2(scale_log2, scale_log2);
                    float2 scaled_28;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_28));
                    #else
                    float2 scaled_28 = CAKE_F32X2_MUL_NZ(make_float2(p[56], p[57]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[56] = scaled_28.x;
                    p[57] = scaled_28.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_29 = make_float2(p[58], p[59]);
                    float2 _f32x2_mul_b_scaled_29 = make_float2(scale_log2, scale_log2);
                    float2 scaled_29;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_29));
                    #else
                    float2 scaled_29 = CAKE_F32X2_MUL_NZ(make_float2(p[58], p[59]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[58] = scaled_29.x;
                    p[59] = scaled_29.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_30 = make_float2(p[60], p[61]);
                    float2 _f32x2_mul_b_scaled_30 = make_float2(scale_log2, scale_log2);
                    float2 scaled_30;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_30));
                    #else
                    float2 scaled_30 = CAKE_F32X2_MUL_NZ(make_float2(p[60], p[61]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[60] = scaled_30.x;
                    p[61] = scaled_30.y;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
                    float2 _f32x2_mul_a_scaled_31 = make_float2(p[62], p[63]);
                    float2 _f32x2_mul_b_scaled_31 = make_float2(scale_log2, scale_log2);
                    float2 scaled_31;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_scaled_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_scaled_31));
                    #else
                    float2 scaled_31 = CAKE_F32X2_MUL_NZ(make_float2(p[62], p[63]), make_float2(scale_log2, scale_log2), make_float2(__uint_as_float((unsigned int)topk >> 31 | 2147483648u), __uint_as_float((unsigned int)topk >> 31 | 2147483648u)));
                    #endif
                    p[62] = scaled_31.x;
                    p[63] = scaled_31.y;
                    const float2 _sub2_2 = {mi, mi};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        sub_f32x2_inplace(&reinterpret_cast<float2*>(p)[_ls], _sub2_2);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        p[_le] = approx_exp2(p[_le]);
                    }
                    float2 sum_pair = make_float2(0.0f, 0.0f);
                    sum_pair = add_f32x2(sum_pair, make_float2(p[0], p[1]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[2], p[3]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[4], p[5]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[6], p[7]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[8], p[9]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[10], p[11]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[12], p[13]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[14], p[15]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[16], p[17]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[18], p[19]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[20], p[21]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[22], p[23]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[24], p[25]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[26], p[27]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[28], p[29]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[30], p[31]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[32], p[33]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[34], p[35]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[36], p[37]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[38], p[39]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[40], p[41]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[42], p[43]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[44], p[45]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[46], p[47]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[48], p[49]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[50], p[51]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[52], p[53]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[54], p[55]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[56], p[57]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[58], p[59]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[60], p[61]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[62], p[63]));
                    float cur_sum = sum_pair.x + sum_pair.y;
                    li = __fmaf_rn(li, factor, cur_sum);
                    int _stage_pv_b = (int)((unsigned int)(base + n) % 2);
                    int _parity_pv_b = (int)((unsigned int)(base + n) / 2 % 2) ^ 1;
                    mbarrier_wait(pv_b_addr + (_stage_pv_b) * 8, _parity_pv_b);
                    unsigned int pk_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                        pk_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(prob_addr + (unsigned int)((base + n) % 2 * 16384 + (row % 64 + 64 * (kh * 8)) * 16)), "r"(pk_0[0]), "r"(pk_0[1]), "r"(pk_0[2]), "r"(pk_0[3]) : "memory");
                    unsigned int pk_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 8], p[_lp*2+1 + 8]));
                        pk_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(prob_addr + (unsigned int)((base + n) % 2 * 16384 + (row % 64 + 64 * (kh * 8 + 1)) * 16)), "r"(pk_1[0]), "r"(pk_1[1]), "r"(pk_1[2]), "r"(pk_1[3]) : "memory");
                    unsigned int pk_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 16], p[_lp*2+1 + 16]));
                        pk_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(prob_addr + (unsigned int)((base + n) % 2 * 16384 + (row % 64 + 64 * (kh * 8 + 2)) * 16)), "r"(pk_2[0]), "r"(pk_2[1]), "r"(pk_2[2]), "r"(pk_2[3]) : "memory");
                    unsigned int pk_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 24], p[_lp*2+1 + 24]));
                        pk_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(prob_addr + (unsigned int)((base + n) % 2 * 16384 + (row % 64 + 64 * (kh * 8 + 3)) * 16)), "r"(pk_3[0]), "r"(pk_3[1]), "r"(pk_3[2]), "r"(pk_3[3]) : "memory");
                    unsigned int pk_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 32], p[_lp*2+1 + 32]));
                        pk_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(prob_addr + (unsigned int)((base + n) % 2 * 16384 + (row % 64 + 64 * (kh * 8 + 4)) * 16)), "r"(pk_4[0]), "r"(pk_4[1]), "r"(pk_4[2]), "r"(pk_4[3]) : "memory");
                    unsigned int pk_5[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 40], p[_lp*2+1 + 40]));
                        pk_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(prob_addr + (unsigned int)((base + n) % 2 * 16384 + (row % 64 + 64 * (kh * 8 + 5)) * 16)), "r"(pk_5[0]), "r"(pk_5[1]), "r"(pk_5[2]), "r"(pk_5[3]) : "memory");
                    unsigned int pk_6[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 48], p[_lp*2+1 + 48]));
                        pk_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(prob_addr + (unsigned int)((base + n) % 2 * 16384 + (row % 64 + 64 * (kh * 8 + 6)) * 16)), "r"(pk_6[0]), "r"(pk_6[1]), "r"(pk_6[2]), "r"(pk_6[3]) : "memory");
                    unsigned int pk_7[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 56], p[_lp*2+1 + 56]));
                        pk_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(prob_addr + (unsigned int)((base + n) % 2 * 16384 + (row % 64 + 64 * (kh * 8 + 7)) * 16)), "r"(pk_7[0]), "r"(pk_7[1]), "r"(pk_7[2]), "r"(pk_7[3]) : "memory");
                    if (n > 0) {
                        int _stage_pv_b_1 = (int)((unsigned int)(base + n - 1) % 2);
                        int _parity_pv_b_1 = (int)((unsigned int)(base + n - 1) / 2 % 2);
                        mbarrier_wait(pv_b_addr + (_stage_pv_b_1) * 8, _parity_pv_b_1);
                    }
                    if (n > 0 && rescale != 0) {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(tmrow));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_3 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_3);
                        tmem_st_x32_f32(tmrow, _tmem_load_0);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(tmrow + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_4 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_4);
                        tmem_st_x32_f32(tmrow + 32, _tmem_load_1);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                            : "r"(tmrow + 64));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_5 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_5);
                        tmem_st_x32_f32(tmrow + 64, _tmem_load_2);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                            : "r"(tmrow + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_6 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_6);
                        tmem_st_x32_f32(tmrow + 96, _tmem_load_3);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                            : "r"(tmrow + 128));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_7 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_7);
                        tmem_st_x32_f32(tmrow + 128, _tmem_load_4);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                            : "r"(tmrow + 160));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_8 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_8);
                        tmem_st_x32_f32(tmrow + 160, _tmem_load_5);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                            : "r"(tmrow + 192));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_9 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_9);
                        tmem_st_x32_f32(tmrow + 192, _tmem_load_6);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_7[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                            : "r"(tmrow + 224));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_10 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_10);
                        tmem_st_x32_f32(tmrow + 224, _tmem_load_7);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(so_ready_addr + ((int)((unsigned int)(base + n) % 2)) * 8), "r"(0) : "memory");
                }
                base += tiles;
                int _stage_pv_b_2 = (int)((unsigned int)(base - 1) % 2);
                int _parity_pv_b_2 = (int)((unsigned int)(base - 1) / 2 % 2);
                mbarrier_wait(pv_b_addr + (_stage_pv_b_2) * 8, _parity_pv_b_2);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (it + 1 < ntok) {
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(q_full_addr), "l"((&q_latent)), "r"(0), "r"(cta_rank * 64), "r"(0), "r"(query + nclu),
                                   "r"(((q_full_bar_addr) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(q_full_addr + (unsigned int)(8 * 8192)), "l"((&q_rope)), "r"(0), "r"(cta_rank * 64), "r"(0), "r"(query + nclu),
                                   "r"(((q_full_bar_addr) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (real_mi == -CUDART_INF_F) {
                    li = 0.0f;
                    mi = -CUDART_INF_F;
                }
                stats[256 + row] = li;
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                li = li + stats[256 + (row ^ 64)];
                if (kh == 0 && live != 0) {
                    float _log_11 = logf(li);
                    float cur_lse = __fmaf_rn(mi, 0.6931472f, _log_11);
                    if (real_mi == -CUDART_INF_F) {
                        cur_lse = -CUDART_INF_F;
                    }
                    lse[query * num_heads + hrow] = cur_lse;
                }
                float output_scale = 1.0f / li;
                int have_valid = __any_sync(0xFFFFFFFF, li != 0.0f);
                if (have_valid == 0) {
                    output_scale = 1.0f;
                }
                long long row_out = ((long long)query * (long long)num_heads + (long long)hrow) * 512ll + (long long)(kh * 256);
                float ov_0[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_0[0]), "=f"(ov_0[1]), "=f"(ov_0[2]), "=f"(ov_0[3]), "=f"(ov_0[4]), "=f"(ov_0[5]), "=f"(ov_0[6]), "=f"(ov_0[7]), "=f"(ov_0[8]), "=f"(ov_0[9]), "=f"(ov_0[10]), "=f"(ov_0[11]), "=f"(ov_0[12]), "=f"(ov_0[13]), "=f"(ov_0[14]), "=f"(ov_0[15]), "=f"(ov_0[16]), "=f"(ov_0[17]), "=f"(ov_0[18]), "=f"(ov_0[19]), "=f"(ov_0[20]), "=f"(ov_0[21]), "=f"(ov_0[22]), "=f"(ov_0[23]), "=f"(ov_0[24]), "=f"(ov_0[25]), "=f"(ov_0[26]), "=f"(ov_0[27]), "=f"(ov_0[28]), "=f"(ov_0[29]), "=f"(ov_0[30]), "=f"(ov_0[31]), "=f"(ov_0[32]), "=f"(ov_0[33]), "=f"(ov_0[34]), "=f"(ov_0[35]), "=f"(ov_0[36]), "=f"(ov_0[37]), "=f"(ov_0[38]), "=f"(ov_0[39]), "=f"(ov_0[40]), "=f"(ov_0[41]), "=f"(ov_0[42]), "=f"(ov_0[43]), "=f"(ov_0[44]), "=f"(ov_0[45]), "=f"(ov_0[46]), "=f"(ov_0[47]), "=f"(ov_0[48]), "=f"(ov_0[49]), "=f"(ov_0[50]), "=f"(ov_0[51]), "=f"(ov_0[52]), "=f"(ov_0[53]), "=f"(ov_0[54]), "=f"(ov_0[55]), "=f"(ov_0[56]), "=f"(ov_0[57]), "=f"(ov_0[58]), "=f"(ov_0[59]), "=f"(ov_0[60]), "=f"(ov_0[61]), "=f"(ov_0[62]), "=f"(ov_0[63])
                        : "r"(tmrow));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_0[0] = 0.0f;
                    ov_0[1] = 0.0f;
                    ov_0[2] = 0.0f;
                    ov_0[3] = 0.0f;
                    ov_0[4] = 0.0f;
                    ov_0[5] = 0.0f;
                    ov_0[6] = 0.0f;
                    ov_0[7] = 0.0f;
                    ov_0[8] = 0.0f;
                    ov_0[9] = 0.0f;
                    ov_0[10] = 0.0f;
                    ov_0[11] = 0.0f;
                    ov_0[12] = 0.0f;
                    ov_0[13] = 0.0f;
                    ov_0[14] = 0.0f;
                    ov_0[15] = 0.0f;
                    ov_0[16] = 0.0f;
                    ov_0[17] = 0.0f;
                    ov_0[18] = 0.0f;
                    ov_0[19] = 0.0f;
                    ov_0[20] = 0.0f;
                    ov_0[21] = 0.0f;
                    ov_0[22] = 0.0f;
                    ov_0[23] = 0.0f;
                    ov_0[24] = 0.0f;
                    ov_0[25] = 0.0f;
                    ov_0[26] = 0.0f;
                    ov_0[27] = 0.0f;
                    ov_0[28] = 0.0f;
                    ov_0[29] = 0.0f;
                    ov_0[30] = 0.0f;
                    ov_0[31] = 0.0f;
                    ov_0[32] = 0.0f;
                    ov_0[33] = 0.0f;
                    ov_0[34] = 0.0f;
                    ov_0[35] = 0.0f;
                    ov_0[36] = 0.0f;
                    ov_0[37] = 0.0f;
                    ov_0[38] = 0.0f;
                    ov_0[39] = 0.0f;
                    ov_0[40] = 0.0f;
                    ov_0[41] = 0.0f;
                    ov_0[42] = 0.0f;
                    ov_0[43] = 0.0f;
                    ov_0[44] = 0.0f;
                    ov_0[45] = 0.0f;
                    ov_0[46] = 0.0f;
                    ov_0[47] = 0.0f;
                    ov_0[48] = 0.0f;
                    ov_0[49] = 0.0f;
                    ov_0[50] = 0.0f;
                    ov_0[51] = 0.0f;
                    ov_0[52] = 0.0f;
                    ov_0[53] = 0.0f;
                    ov_0[54] = 0.0f;
                    ov_0[55] = 0.0f;
                    ov_0[56] = 0.0f;
                    ov_0[57] = 0.0f;
                    ov_0[58] = 0.0f;
                    ov_0[59] = 0.0f;
                    ov_0[60] = 0.0f;
                    ov_0[61] = 0.0f;
                    ov_0[62] = 0.0f;
                    ov_0[63] = 0.0f;
                }
                const float2 _scale2_12 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_0)[_ls], _scale2_12);
                unsigned int hi[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 0], ov_0[_lp*2+1 + 0]));
                    hi[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (0)), "r"(hi[(0) + 0]), "r"(hi[(0) + 1]), "r"(hi[(0) + 2]), "r"(hi[(0) + 3]), "r"(hi[(0) + 4]), "r"(hi[(0) + 5]), "r"(hi[(0) + 6]), "r"(hi[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi[_pair]) << 16);
                    (&unpacked[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi[_pair]) & 0xffff0000u);
                }
                float res[16];
                res[0] = ov_0[0] - unpacked[0];
                res[1] = ov_0[1] - unpacked[1];
                res[2] = ov_0[2] - unpacked[2];
                res[3] = ov_0[3] - unpacked[3];
                res[4] = ov_0[4] - unpacked[4];
                res[5] = ov_0[5] - unpacked[5];
                res[6] = ov_0[6] - unpacked[6];
                res[7] = ov_0[7] - unpacked[7];
                res[8] = ov_0[8] - unpacked[8];
                res[9] = ov_0[9] - unpacked[9];
                res[10] = ov_0[10] - unpacked[10];
                res[11] = ov_0[11] - unpacked[11];
                res[12] = ov_0[12] - unpacked[12];
                res[13] = ov_0[13] - unpacked[13];
                res[14] = ov_0[14] - unpacked[14];
                res[15] = ov_0[15] - unpacked[15];
                unsigned int lo[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 0], res[_lp*2+1 + 0]));
                    lo[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (0)), "r"(lo[(0) + 0]), "r"(lo[(0) + 1]), "r"(lo[(0) + 2]), "r"(lo[(0) + 3]), "r"(lo[(0) + 4]), "r"(lo[(0) + 5]), "r"(lo[(0) + 6]), "r"(lo[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 16], ov_0[_lp*2+1 + 16]));
                    hi_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (16)), "r"(hi_0[(0) + 0]), "r"(hi_0[(0) + 1]), "r"(hi_0[(0) + 2]), "r"(hi_0[(0) + 3]), "r"(hi_0[(0) + 4]), "r"(hi_0[(0) + 5]), "r"(hi_0[(0) + 6]), "r"(hi_0[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_0[_pair]) << 16);
                    (&unpacked_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_0[_pair]) & 0xffff0000u);
                }
                float res_0[16];
                res_0[0] = ov_0[16] - unpacked_0[0];
                res_0[1] = ov_0[17] - unpacked_0[1];
                res_0[2] = ov_0[18] - unpacked_0[2];
                res_0[3] = ov_0[19] - unpacked_0[3];
                res_0[4] = ov_0[20] - unpacked_0[4];
                res_0[5] = ov_0[21] - unpacked_0[5];
                res_0[6] = ov_0[22] - unpacked_0[6];
                res_0[7] = ov_0[23] - unpacked_0[7];
                res_0[8] = ov_0[24] - unpacked_0[8];
                res_0[9] = ov_0[25] - unpacked_0[9];
                res_0[10] = ov_0[26] - unpacked_0[10];
                res_0[11] = ov_0[27] - unpacked_0[11];
                res_0[12] = ov_0[28] - unpacked_0[12];
                res_0[13] = ov_0[29] - unpacked_0[13];
                res_0[14] = ov_0[30] - unpacked_0[14];
                res_0[15] = ov_0[31] - unpacked_0[15];
                unsigned int lo_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 0], res_0[_lp*2+1 + 0]));
                    lo_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (16)), "r"(lo_0[(0) + 0]), "r"(lo_0[(0) + 1]), "r"(lo_0[(0) + 2]), "r"(lo_0[(0) + 3]), "r"(lo_0[(0) + 4]), "r"(lo_0[(0) + 5]), "r"(lo_0[(0) + 6]), "r"(lo_0[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 32], ov_0[_lp*2+1 + 32]));
                    hi_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (32)), "r"(hi_1[(0) + 0]), "r"(hi_1[(0) + 1]), "r"(hi_1[(0) + 2]), "r"(hi_1[(0) + 3]), "r"(hi_1[(0) + 4]), "r"(hi_1[(0) + 5]), "r"(hi_1[(0) + 6]), "r"(hi_1[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_1[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_1[_pair]) << 16);
                    (&unpacked_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_1[_pair]) & 0xffff0000u);
                }
                float res_1[16];
                res_1[0] = ov_0[32] - unpacked_1[0];
                res_1[1] = ov_0[33] - unpacked_1[1];
                res_1[2] = ov_0[34] - unpacked_1[2];
                res_1[3] = ov_0[35] - unpacked_1[3];
                res_1[4] = ov_0[36] - unpacked_1[4];
                res_1[5] = ov_0[37] - unpacked_1[5];
                res_1[6] = ov_0[38] - unpacked_1[6];
                res_1[7] = ov_0[39] - unpacked_1[7];
                res_1[8] = ov_0[40] - unpacked_1[8];
                res_1[9] = ov_0[41] - unpacked_1[9];
                res_1[10] = ov_0[42] - unpacked_1[10];
                res_1[11] = ov_0[43] - unpacked_1[11];
                res_1[12] = ov_0[44] - unpacked_1[12];
                res_1[13] = ov_0[45] - unpacked_1[13];
                res_1[14] = ov_0[46] - unpacked_1[14];
                res_1[15] = ov_0[47] - unpacked_1[15];
                unsigned int lo_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 0], res_1[_lp*2+1 + 0]));
                    lo_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (32)), "r"(lo_1[(0) + 0]), "r"(lo_1[(0) + 1]), "r"(lo_1[(0) + 2]), "r"(lo_1[(0) + 3]), "r"(lo_1[(0) + 4]), "r"(lo_1[(0) + 5]), "r"(lo_1[(0) + 6]), "r"(lo_1[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 48], ov_0[_lp*2+1 + 48]));
                    hi_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (48)), "r"(hi_2[(0) + 0]), "r"(hi_2[(0) + 1]), "r"(hi_2[(0) + 2]), "r"(hi_2[(0) + 3]), "r"(hi_2[(0) + 4]), "r"(hi_2[(0) + 5]), "r"(hi_2[(0) + 6]), "r"(hi_2[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_2[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_2[_pair]) << 16);
                    (&unpacked_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_2[_pair]) & 0xffff0000u);
                }
                float res_2[16];
                res_2[0] = ov_0[48] - unpacked_2[0];
                res_2[1] = ov_0[49] - unpacked_2[1];
                res_2[2] = ov_0[50] - unpacked_2[2];
                res_2[3] = ov_0[51] - unpacked_2[3];
                res_2[4] = ov_0[52] - unpacked_2[4];
                res_2[5] = ov_0[53] - unpacked_2[5];
                res_2[6] = ov_0[54] - unpacked_2[6];
                res_2[7] = ov_0[55] - unpacked_2[7];
                res_2[8] = ov_0[56] - unpacked_2[8];
                res_2[9] = ov_0[57] - unpacked_2[9];
                res_2[10] = ov_0[58] - unpacked_2[10];
                res_2[11] = ov_0[59] - unpacked_2[11];
                res_2[12] = ov_0[60] - unpacked_2[12];
                res_2[13] = ov_0[61] - unpacked_2[13];
                res_2[14] = ov_0[62] - unpacked_2[14];
                res_2[15] = ov_0[63] - unpacked_2[15];
                unsigned int lo_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 0], res_2[_lp*2+1 + 0]));
                    lo_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (48)), "r"(lo_2[(0) + 0]), "r"(lo_2[(0) + 1]), "r"(lo_2[(0) + 2]), "r"(lo_2[(0) + 3]), "r"(lo_2[(0) + 4]), "r"(lo_2[(0) + 5]), "r"(lo_2[(0) + 6]), "r"(lo_2[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float ov_1[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_1[0]), "=f"(ov_1[1]), "=f"(ov_1[2]), "=f"(ov_1[3]), "=f"(ov_1[4]), "=f"(ov_1[5]), "=f"(ov_1[6]), "=f"(ov_1[7]), "=f"(ov_1[8]), "=f"(ov_1[9]), "=f"(ov_1[10]), "=f"(ov_1[11]), "=f"(ov_1[12]), "=f"(ov_1[13]), "=f"(ov_1[14]), "=f"(ov_1[15]), "=f"(ov_1[16]), "=f"(ov_1[17]), "=f"(ov_1[18]), "=f"(ov_1[19]), "=f"(ov_1[20]), "=f"(ov_1[21]), "=f"(ov_1[22]), "=f"(ov_1[23]), "=f"(ov_1[24]), "=f"(ov_1[25]), "=f"(ov_1[26]), "=f"(ov_1[27]), "=f"(ov_1[28]), "=f"(ov_1[29]), "=f"(ov_1[30]), "=f"(ov_1[31]), "=f"(ov_1[32]), "=f"(ov_1[33]), "=f"(ov_1[34]), "=f"(ov_1[35]), "=f"(ov_1[36]), "=f"(ov_1[37]), "=f"(ov_1[38]), "=f"(ov_1[39]), "=f"(ov_1[40]), "=f"(ov_1[41]), "=f"(ov_1[42]), "=f"(ov_1[43]), "=f"(ov_1[44]), "=f"(ov_1[45]), "=f"(ov_1[46]), "=f"(ov_1[47]), "=f"(ov_1[48]), "=f"(ov_1[49]), "=f"(ov_1[50]), "=f"(ov_1[51]), "=f"(ov_1[52]), "=f"(ov_1[53]), "=f"(ov_1[54]), "=f"(ov_1[55]), "=f"(ov_1[56]), "=f"(ov_1[57]), "=f"(ov_1[58]), "=f"(ov_1[59]), "=f"(ov_1[60]), "=f"(ov_1[61]), "=f"(ov_1[62]), "=f"(ov_1[63])
                        : "r"(tmrow + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_1[0] = 0.0f;
                    ov_1[1] = 0.0f;
                    ov_1[2] = 0.0f;
                    ov_1[3] = 0.0f;
                    ov_1[4] = 0.0f;
                    ov_1[5] = 0.0f;
                    ov_1[6] = 0.0f;
                    ov_1[7] = 0.0f;
                    ov_1[8] = 0.0f;
                    ov_1[9] = 0.0f;
                    ov_1[10] = 0.0f;
                    ov_1[11] = 0.0f;
                    ov_1[12] = 0.0f;
                    ov_1[13] = 0.0f;
                    ov_1[14] = 0.0f;
                    ov_1[15] = 0.0f;
                    ov_1[16] = 0.0f;
                    ov_1[17] = 0.0f;
                    ov_1[18] = 0.0f;
                    ov_1[19] = 0.0f;
                    ov_1[20] = 0.0f;
                    ov_1[21] = 0.0f;
                    ov_1[22] = 0.0f;
                    ov_1[23] = 0.0f;
                    ov_1[24] = 0.0f;
                    ov_1[25] = 0.0f;
                    ov_1[26] = 0.0f;
                    ov_1[27] = 0.0f;
                    ov_1[28] = 0.0f;
                    ov_1[29] = 0.0f;
                    ov_1[30] = 0.0f;
                    ov_1[31] = 0.0f;
                    ov_1[32] = 0.0f;
                    ov_1[33] = 0.0f;
                    ov_1[34] = 0.0f;
                    ov_1[35] = 0.0f;
                    ov_1[36] = 0.0f;
                    ov_1[37] = 0.0f;
                    ov_1[38] = 0.0f;
                    ov_1[39] = 0.0f;
                    ov_1[40] = 0.0f;
                    ov_1[41] = 0.0f;
                    ov_1[42] = 0.0f;
                    ov_1[43] = 0.0f;
                    ov_1[44] = 0.0f;
                    ov_1[45] = 0.0f;
                    ov_1[46] = 0.0f;
                    ov_1[47] = 0.0f;
                    ov_1[48] = 0.0f;
                    ov_1[49] = 0.0f;
                    ov_1[50] = 0.0f;
                    ov_1[51] = 0.0f;
                    ov_1[52] = 0.0f;
                    ov_1[53] = 0.0f;
                    ov_1[54] = 0.0f;
                    ov_1[55] = 0.0f;
                    ov_1[56] = 0.0f;
                    ov_1[57] = 0.0f;
                    ov_1[58] = 0.0f;
                    ov_1[59] = 0.0f;
                    ov_1[60] = 0.0f;
                    ov_1[61] = 0.0f;
                    ov_1[62] = 0.0f;
                    ov_1[63] = 0.0f;
                }
                const float2 _scale2_13 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_13);
                unsigned int hi_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 0], ov_1[_lp*2+1 + 0]));
                    hi_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (64)), "r"(hi_3[(0) + 0]), "r"(hi_3[(0) + 1]), "r"(hi_3[(0) + 2]), "r"(hi_3[(0) + 3]), "r"(hi_3[(0) + 4]), "r"(hi_3[(0) + 5]), "r"(hi_3[(0) + 6]), "r"(hi_3[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_3[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_3[_pair]) << 16);
                    (&unpacked_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_3[_pair]) & 0xffff0000u);
                }
                float res_3[16];
                res_3[0] = ov_1[0] - unpacked_3[0];
                res_3[1] = ov_1[1] - unpacked_3[1];
                res_3[2] = ov_1[2] - unpacked_3[2];
                res_3[3] = ov_1[3] - unpacked_3[3];
                res_3[4] = ov_1[4] - unpacked_3[4];
                res_3[5] = ov_1[5] - unpacked_3[5];
                res_3[6] = ov_1[6] - unpacked_3[6];
                res_3[7] = ov_1[7] - unpacked_3[7];
                res_3[8] = ov_1[8] - unpacked_3[8];
                res_3[9] = ov_1[9] - unpacked_3[9];
                res_3[10] = ov_1[10] - unpacked_3[10];
                res_3[11] = ov_1[11] - unpacked_3[11];
                res_3[12] = ov_1[12] - unpacked_3[12];
                res_3[13] = ov_1[13] - unpacked_3[13];
                res_3[14] = ov_1[14] - unpacked_3[14];
                res_3[15] = ov_1[15] - unpacked_3[15];
                unsigned int lo_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
                    lo_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (64)), "r"(lo_3[(0) + 0]), "r"(lo_3[(0) + 1]), "r"(lo_3[(0) + 2]), "r"(lo_3[(0) + 3]), "r"(lo_3[(0) + 4]), "r"(lo_3[(0) + 5]), "r"(lo_3[(0) + 6]), "r"(lo_3[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 16], ov_1[_lp*2+1 + 16]));
                    hi_4[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (80)), "r"(hi_4[(0) + 0]), "r"(hi_4[(0) + 1]), "r"(hi_4[(0) + 2]), "r"(hi_4[(0) + 3]), "r"(hi_4[(0) + 4]), "r"(hi_4[(0) + 5]), "r"(hi_4[(0) + 6]), "r"(hi_4[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_4[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_4[_pair]) << 16);
                    (&unpacked_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_4[_pair]) & 0xffff0000u);
                }
                float res_4[16];
                res_4[0] = ov_1[16] - unpacked_4[0];
                res_4[1] = ov_1[17] - unpacked_4[1];
                res_4[2] = ov_1[18] - unpacked_4[2];
                res_4[3] = ov_1[19] - unpacked_4[3];
                res_4[4] = ov_1[20] - unpacked_4[4];
                res_4[5] = ov_1[21] - unpacked_4[5];
                res_4[6] = ov_1[22] - unpacked_4[6];
                res_4[7] = ov_1[23] - unpacked_4[7];
                res_4[8] = ov_1[24] - unpacked_4[8];
                res_4[9] = ov_1[25] - unpacked_4[9];
                res_4[10] = ov_1[26] - unpacked_4[10];
                res_4[11] = ov_1[27] - unpacked_4[11];
                res_4[12] = ov_1[28] - unpacked_4[12];
                res_4[13] = ov_1[29] - unpacked_4[13];
                res_4[14] = ov_1[30] - unpacked_4[14];
                res_4[15] = ov_1[31] - unpacked_4[15];
                unsigned int lo_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                    lo_4[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (80)), "r"(lo_4[(0) + 0]), "r"(lo_4[(0) + 1]), "r"(lo_4[(0) + 2]), "r"(lo_4[(0) + 3]), "r"(lo_4[(0) + 4]), "r"(lo_4[(0) + 5]), "r"(lo_4[(0) + 6]), "r"(lo_4[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 32], ov_1[_lp*2+1 + 32]));
                    hi_5[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (96)), "r"(hi_5[(0) + 0]), "r"(hi_5[(0) + 1]), "r"(hi_5[(0) + 2]), "r"(hi_5[(0) + 3]), "r"(hi_5[(0) + 4]), "r"(hi_5[(0) + 5]), "r"(hi_5[(0) + 6]), "r"(hi_5[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_5[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_5[_pair]) << 16);
                    (&unpacked_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_5[_pair]) & 0xffff0000u);
                }
                float res_5[16];
                res_5[0] = ov_1[32] - unpacked_5[0];
                res_5[1] = ov_1[33] - unpacked_5[1];
                res_5[2] = ov_1[34] - unpacked_5[2];
                res_5[3] = ov_1[35] - unpacked_5[3];
                res_5[4] = ov_1[36] - unpacked_5[4];
                res_5[5] = ov_1[37] - unpacked_5[5];
                res_5[6] = ov_1[38] - unpacked_5[6];
                res_5[7] = ov_1[39] - unpacked_5[7];
                res_5[8] = ov_1[40] - unpacked_5[8];
                res_5[9] = ov_1[41] - unpacked_5[9];
                res_5[10] = ov_1[42] - unpacked_5[10];
                res_5[11] = ov_1[43] - unpacked_5[11];
                res_5[12] = ov_1[44] - unpacked_5[12];
                res_5[13] = ov_1[45] - unpacked_5[13];
                res_5[14] = ov_1[46] - unpacked_5[14];
                res_5[15] = ov_1[47] - unpacked_5[15];
                unsigned int lo_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_5[_lp*2 + 0], res_5[_lp*2+1 + 0]));
                    lo_5[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (96)), "r"(lo_5[(0) + 0]), "r"(lo_5[(0) + 1]), "r"(lo_5[(0) + 2]), "r"(lo_5[(0) + 3]), "r"(lo_5[(0) + 4]), "r"(lo_5[(0) + 5]), "r"(lo_5[(0) + 6]), "r"(lo_5[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 48], ov_1[_lp*2+1 + 48]));
                    hi_6[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (112)), "r"(hi_6[(0) + 0]), "r"(hi_6[(0) + 1]), "r"(hi_6[(0) + 2]), "r"(hi_6[(0) + 3]), "r"(hi_6[(0) + 4]), "r"(hi_6[(0) + 5]), "r"(hi_6[(0) + 6]), "r"(hi_6[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_6[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_6[_pair]) << 16);
                    (&unpacked_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_6[_pair]) & 0xffff0000u);
                }
                float res_6[16];
                res_6[0] = ov_1[48] - unpacked_6[0];
                res_6[1] = ov_1[49] - unpacked_6[1];
                res_6[2] = ov_1[50] - unpacked_6[2];
                res_6[3] = ov_1[51] - unpacked_6[3];
                res_6[4] = ov_1[52] - unpacked_6[4];
                res_6[5] = ov_1[53] - unpacked_6[5];
                res_6[6] = ov_1[54] - unpacked_6[6];
                res_6[7] = ov_1[55] - unpacked_6[7];
                res_6[8] = ov_1[56] - unpacked_6[8];
                res_6[9] = ov_1[57] - unpacked_6[9];
                res_6[10] = ov_1[58] - unpacked_6[10];
                res_6[11] = ov_1[59] - unpacked_6[11];
                res_6[12] = ov_1[60] - unpacked_6[12];
                res_6[13] = ov_1[61] - unpacked_6[13];
                res_6[14] = ov_1[62] - unpacked_6[14];
                res_6[15] = ov_1[63] - unpacked_6[15];
                unsigned int lo_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_6[_lp*2 + 0], res_6[_lp*2+1 + 0]));
                    lo_6[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (112)), "r"(lo_6[(0) + 0]), "r"(lo_6[(0) + 1]), "r"(lo_6[(0) + 2]), "r"(lo_6[(0) + 3]), "r"(lo_6[(0) + 4]), "r"(lo_6[(0) + 5]), "r"(lo_6[(0) + 6]), "r"(lo_6[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float ov_2[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_2[0]), "=f"(ov_2[1]), "=f"(ov_2[2]), "=f"(ov_2[3]), "=f"(ov_2[4]), "=f"(ov_2[5]), "=f"(ov_2[6]), "=f"(ov_2[7]), "=f"(ov_2[8]), "=f"(ov_2[9]), "=f"(ov_2[10]), "=f"(ov_2[11]), "=f"(ov_2[12]), "=f"(ov_2[13]), "=f"(ov_2[14]), "=f"(ov_2[15]), "=f"(ov_2[16]), "=f"(ov_2[17]), "=f"(ov_2[18]), "=f"(ov_2[19]), "=f"(ov_2[20]), "=f"(ov_2[21]), "=f"(ov_2[22]), "=f"(ov_2[23]), "=f"(ov_2[24]), "=f"(ov_2[25]), "=f"(ov_2[26]), "=f"(ov_2[27]), "=f"(ov_2[28]), "=f"(ov_2[29]), "=f"(ov_2[30]), "=f"(ov_2[31]), "=f"(ov_2[32]), "=f"(ov_2[33]), "=f"(ov_2[34]), "=f"(ov_2[35]), "=f"(ov_2[36]), "=f"(ov_2[37]), "=f"(ov_2[38]), "=f"(ov_2[39]), "=f"(ov_2[40]), "=f"(ov_2[41]), "=f"(ov_2[42]), "=f"(ov_2[43]), "=f"(ov_2[44]), "=f"(ov_2[45]), "=f"(ov_2[46]), "=f"(ov_2[47]), "=f"(ov_2[48]), "=f"(ov_2[49]), "=f"(ov_2[50]), "=f"(ov_2[51]), "=f"(ov_2[52]), "=f"(ov_2[53]), "=f"(ov_2[54]), "=f"(ov_2[55]), "=f"(ov_2[56]), "=f"(ov_2[57]), "=f"(ov_2[58]), "=f"(ov_2[59]), "=f"(ov_2[60]), "=f"(ov_2[61]), "=f"(ov_2[62]), "=f"(ov_2[63])
                        : "r"(tmrow + 128));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_2[0] = 0.0f;
                    ov_2[1] = 0.0f;
                    ov_2[2] = 0.0f;
                    ov_2[3] = 0.0f;
                    ov_2[4] = 0.0f;
                    ov_2[5] = 0.0f;
                    ov_2[6] = 0.0f;
                    ov_2[7] = 0.0f;
                    ov_2[8] = 0.0f;
                    ov_2[9] = 0.0f;
                    ov_2[10] = 0.0f;
                    ov_2[11] = 0.0f;
                    ov_2[12] = 0.0f;
                    ov_2[13] = 0.0f;
                    ov_2[14] = 0.0f;
                    ov_2[15] = 0.0f;
                    ov_2[16] = 0.0f;
                    ov_2[17] = 0.0f;
                    ov_2[18] = 0.0f;
                    ov_2[19] = 0.0f;
                    ov_2[20] = 0.0f;
                    ov_2[21] = 0.0f;
                    ov_2[22] = 0.0f;
                    ov_2[23] = 0.0f;
                    ov_2[24] = 0.0f;
                    ov_2[25] = 0.0f;
                    ov_2[26] = 0.0f;
                    ov_2[27] = 0.0f;
                    ov_2[28] = 0.0f;
                    ov_2[29] = 0.0f;
                    ov_2[30] = 0.0f;
                    ov_2[31] = 0.0f;
                    ov_2[32] = 0.0f;
                    ov_2[33] = 0.0f;
                    ov_2[34] = 0.0f;
                    ov_2[35] = 0.0f;
                    ov_2[36] = 0.0f;
                    ov_2[37] = 0.0f;
                    ov_2[38] = 0.0f;
                    ov_2[39] = 0.0f;
                    ov_2[40] = 0.0f;
                    ov_2[41] = 0.0f;
                    ov_2[42] = 0.0f;
                    ov_2[43] = 0.0f;
                    ov_2[44] = 0.0f;
                    ov_2[45] = 0.0f;
                    ov_2[46] = 0.0f;
                    ov_2[47] = 0.0f;
                    ov_2[48] = 0.0f;
                    ov_2[49] = 0.0f;
                    ov_2[50] = 0.0f;
                    ov_2[51] = 0.0f;
                    ov_2[52] = 0.0f;
                    ov_2[53] = 0.0f;
                    ov_2[54] = 0.0f;
                    ov_2[55] = 0.0f;
                    ov_2[56] = 0.0f;
                    ov_2[57] = 0.0f;
                    ov_2[58] = 0.0f;
                    ov_2[59] = 0.0f;
                    ov_2[60] = 0.0f;
                    ov_2[61] = 0.0f;
                    ov_2[62] = 0.0f;
                    ov_2[63] = 0.0f;
                }
                const float2 _scale2_14 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_2)[_ls], _scale2_14);
                unsigned int hi_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 0], ov_2[_lp*2+1 + 0]));
                    hi_7[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (128)), "r"(hi_7[(0) + 0]), "r"(hi_7[(0) + 1]), "r"(hi_7[(0) + 2]), "r"(hi_7[(0) + 3]), "r"(hi_7[(0) + 4]), "r"(hi_7[(0) + 5]), "r"(hi_7[(0) + 6]), "r"(hi_7[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_7[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_7[_pair]) << 16);
                    (&unpacked_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_7[_pair]) & 0xffff0000u);
                }
                float res_7[16];
                res_7[0] = ov_2[0] - unpacked_7[0];
                res_7[1] = ov_2[1] - unpacked_7[1];
                res_7[2] = ov_2[2] - unpacked_7[2];
                res_7[3] = ov_2[3] - unpacked_7[3];
                res_7[4] = ov_2[4] - unpacked_7[4];
                res_7[5] = ov_2[5] - unpacked_7[5];
                res_7[6] = ov_2[6] - unpacked_7[6];
                res_7[7] = ov_2[7] - unpacked_7[7];
                res_7[8] = ov_2[8] - unpacked_7[8];
                res_7[9] = ov_2[9] - unpacked_7[9];
                res_7[10] = ov_2[10] - unpacked_7[10];
                res_7[11] = ov_2[11] - unpacked_7[11];
                res_7[12] = ov_2[12] - unpacked_7[12];
                res_7[13] = ov_2[13] - unpacked_7[13];
                res_7[14] = ov_2[14] - unpacked_7[14];
                res_7[15] = ov_2[15] - unpacked_7[15];
                unsigned int lo_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_7[_lp*2 + 0], res_7[_lp*2+1 + 0]));
                    lo_7[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (128)), "r"(lo_7[(0) + 0]), "r"(lo_7[(0) + 1]), "r"(lo_7[(0) + 2]), "r"(lo_7[(0) + 3]), "r"(lo_7[(0) + 4]), "r"(lo_7[(0) + 5]), "r"(lo_7[(0) + 6]), "r"(lo_7[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 16], ov_2[_lp*2+1 + 16]));
                    hi_8[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (144)), "r"(hi_8[(0) + 0]), "r"(hi_8[(0) + 1]), "r"(hi_8[(0) + 2]), "r"(hi_8[(0) + 3]), "r"(hi_8[(0) + 4]), "r"(hi_8[(0) + 5]), "r"(hi_8[(0) + 6]), "r"(hi_8[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_8[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_8[_pair]) << 16);
                    (&unpacked_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_8[_pair]) & 0xffff0000u);
                }
                float res_8[16];
                res_8[0] = ov_2[16] - unpacked_8[0];
                res_8[1] = ov_2[17] - unpacked_8[1];
                res_8[2] = ov_2[18] - unpacked_8[2];
                res_8[3] = ov_2[19] - unpacked_8[3];
                res_8[4] = ov_2[20] - unpacked_8[4];
                res_8[5] = ov_2[21] - unpacked_8[5];
                res_8[6] = ov_2[22] - unpacked_8[6];
                res_8[7] = ov_2[23] - unpacked_8[7];
                res_8[8] = ov_2[24] - unpacked_8[8];
                res_8[9] = ov_2[25] - unpacked_8[9];
                res_8[10] = ov_2[26] - unpacked_8[10];
                res_8[11] = ov_2[27] - unpacked_8[11];
                res_8[12] = ov_2[28] - unpacked_8[12];
                res_8[13] = ov_2[29] - unpacked_8[13];
                res_8[14] = ov_2[30] - unpacked_8[14];
                res_8[15] = ov_2[31] - unpacked_8[15];
                unsigned int lo_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_8[_lp*2 + 0], res_8[_lp*2+1 + 0]));
                    lo_8[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (144)), "r"(lo_8[(0) + 0]), "r"(lo_8[(0) + 1]), "r"(lo_8[(0) + 2]), "r"(lo_8[(0) + 3]), "r"(lo_8[(0) + 4]), "r"(lo_8[(0) + 5]), "r"(lo_8[(0) + 6]), "r"(lo_8[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_9[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 32], ov_2[_lp*2+1 + 32]));
                    hi_9[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (160)), "r"(hi_9[(0) + 0]), "r"(hi_9[(0) + 1]), "r"(hi_9[(0) + 2]), "r"(hi_9[(0) + 3]), "r"(hi_9[(0) + 4]), "r"(hi_9[(0) + 5]), "r"(hi_9[(0) + 6]), "r"(hi_9[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_9[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_9[_pair]) << 16);
                    (&unpacked_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_9[_pair]) & 0xffff0000u);
                }
                float res_9[16];
                res_9[0] = ov_2[32] - unpacked_9[0];
                res_9[1] = ov_2[33] - unpacked_9[1];
                res_9[2] = ov_2[34] - unpacked_9[2];
                res_9[3] = ov_2[35] - unpacked_9[3];
                res_9[4] = ov_2[36] - unpacked_9[4];
                res_9[5] = ov_2[37] - unpacked_9[5];
                res_9[6] = ov_2[38] - unpacked_9[6];
                res_9[7] = ov_2[39] - unpacked_9[7];
                res_9[8] = ov_2[40] - unpacked_9[8];
                res_9[9] = ov_2[41] - unpacked_9[9];
                res_9[10] = ov_2[42] - unpacked_9[10];
                res_9[11] = ov_2[43] - unpacked_9[11];
                res_9[12] = ov_2[44] - unpacked_9[12];
                res_9[13] = ov_2[45] - unpacked_9[13];
                res_9[14] = ov_2[46] - unpacked_9[14];
                res_9[15] = ov_2[47] - unpacked_9[15];
                unsigned int lo_9[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_9[_lp*2 + 0], res_9[_lp*2+1 + 0]));
                    lo_9[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (160)), "r"(lo_9[(0) + 0]), "r"(lo_9[(0) + 1]), "r"(lo_9[(0) + 2]), "r"(lo_9[(0) + 3]), "r"(lo_9[(0) + 4]), "r"(lo_9[(0) + 5]), "r"(lo_9[(0) + 6]), "r"(lo_9[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_10[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 48], ov_2[_lp*2+1 + 48]));
                    hi_10[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (176)), "r"(hi_10[(0) + 0]), "r"(hi_10[(0) + 1]), "r"(hi_10[(0) + 2]), "r"(hi_10[(0) + 3]), "r"(hi_10[(0) + 4]), "r"(hi_10[(0) + 5]), "r"(hi_10[(0) + 6]), "r"(hi_10[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_10[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_10[_pair]) << 16);
                    (&unpacked_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_10[_pair]) & 0xffff0000u);
                }
                float res_10[16];
                res_10[0] = ov_2[48] - unpacked_10[0];
                res_10[1] = ov_2[49] - unpacked_10[1];
                res_10[2] = ov_2[50] - unpacked_10[2];
                res_10[3] = ov_2[51] - unpacked_10[3];
                res_10[4] = ov_2[52] - unpacked_10[4];
                res_10[5] = ov_2[53] - unpacked_10[5];
                res_10[6] = ov_2[54] - unpacked_10[6];
                res_10[7] = ov_2[55] - unpacked_10[7];
                res_10[8] = ov_2[56] - unpacked_10[8];
                res_10[9] = ov_2[57] - unpacked_10[9];
                res_10[10] = ov_2[58] - unpacked_10[10];
                res_10[11] = ov_2[59] - unpacked_10[11];
                res_10[12] = ov_2[60] - unpacked_10[12];
                res_10[13] = ov_2[61] - unpacked_10[13];
                res_10[14] = ov_2[62] - unpacked_10[14];
                res_10[15] = ov_2[63] - unpacked_10[15];
                unsigned int lo_10[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_10[_lp*2 + 0], res_10[_lp*2+1 + 0]));
                    lo_10[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (176)), "r"(lo_10[(0) + 0]), "r"(lo_10[(0) + 1]), "r"(lo_10[(0) + 2]), "r"(lo_10[(0) + 3]), "r"(lo_10[(0) + 4]), "r"(lo_10[(0) + 5]), "r"(lo_10[(0) + 6]), "r"(lo_10[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float ov_3[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_3[0]), "=f"(ov_3[1]), "=f"(ov_3[2]), "=f"(ov_3[3]), "=f"(ov_3[4]), "=f"(ov_3[5]), "=f"(ov_3[6]), "=f"(ov_3[7]), "=f"(ov_3[8]), "=f"(ov_3[9]), "=f"(ov_3[10]), "=f"(ov_3[11]), "=f"(ov_3[12]), "=f"(ov_3[13]), "=f"(ov_3[14]), "=f"(ov_3[15]), "=f"(ov_3[16]), "=f"(ov_3[17]), "=f"(ov_3[18]), "=f"(ov_3[19]), "=f"(ov_3[20]), "=f"(ov_3[21]), "=f"(ov_3[22]), "=f"(ov_3[23]), "=f"(ov_3[24]), "=f"(ov_3[25]), "=f"(ov_3[26]), "=f"(ov_3[27]), "=f"(ov_3[28]), "=f"(ov_3[29]), "=f"(ov_3[30]), "=f"(ov_3[31]), "=f"(ov_3[32]), "=f"(ov_3[33]), "=f"(ov_3[34]), "=f"(ov_3[35]), "=f"(ov_3[36]), "=f"(ov_3[37]), "=f"(ov_3[38]), "=f"(ov_3[39]), "=f"(ov_3[40]), "=f"(ov_3[41]), "=f"(ov_3[42]), "=f"(ov_3[43]), "=f"(ov_3[44]), "=f"(ov_3[45]), "=f"(ov_3[46]), "=f"(ov_3[47]), "=f"(ov_3[48]), "=f"(ov_3[49]), "=f"(ov_3[50]), "=f"(ov_3[51]), "=f"(ov_3[52]), "=f"(ov_3[53]), "=f"(ov_3[54]), "=f"(ov_3[55]), "=f"(ov_3[56]), "=f"(ov_3[57]), "=f"(ov_3[58]), "=f"(ov_3[59]), "=f"(ov_3[60]), "=f"(ov_3[61]), "=f"(ov_3[62]), "=f"(ov_3[63])
                        : "r"(tmrow + 192));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_3[0] = 0.0f;
                    ov_3[1] = 0.0f;
                    ov_3[2] = 0.0f;
                    ov_3[3] = 0.0f;
                    ov_3[4] = 0.0f;
                    ov_3[5] = 0.0f;
                    ov_3[6] = 0.0f;
                    ov_3[7] = 0.0f;
                    ov_3[8] = 0.0f;
                    ov_3[9] = 0.0f;
                    ov_3[10] = 0.0f;
                    ov_3[11] = 0.0f;
                    ov_3[12] = 0.0f;
                    ov_3[13] = 0.0f;
                    ov_3[14] = 0.0f;
                    ov_3[15] = 0.0f;
                    ov_3[16] = 0.0f;
                    ov_3[17] = 0.0f;
                    ov_3[18] = 0.0f;
                    ov_3[19] = 0.0f;
                    ov_3[20] = 0.0f;
                    ov_3[21] = 0.0f;
                    ov_3[22] = 0.0f;
                    ov_3[23] = 0.0f;
                    ov_3[24] = 0.0f;
                    ov_3[25] = 0.0f;
                    ov_3[26] = 0.0f;
                    ov_3[27] = 0.0f;
                    ov_3[28] = 0.0f;
                    ov_3[29] = 0.0f;
                    ov_3[30] = 0.0f;
                    ov_3[31] = 0.0f;
                    ov_3[32] = 0.0f;
                    ov_3[33] = 0.0f;
                    ov_3[34] = 0.0f;
                    ov_3[35] = 0.0f;
                    ov_3[36] = 0.0f;
                    ov_3[37] = 0.0f;
                    ov_3[38] = 0.0f;
                    ov_3[39] = 0.0f;
                    ov_3[40] = 0.0f;
                    ov_3[41] = 0.0f;
                    ov_3[42] = 0.0f;
                    ov_3[43] = 0.0f;
                    ov_3[44] = 0.0f;
                    ov_3[45] = 0.0f;
                    ov_3[46] = 0.0f;
                    ov_3[47] = 0.0f;
                    ov_3[48] = 0.0f;
                    ov_3[49] = 0.0f;
                    ov_3[50] = 0.0f;
                    ov_3[51] = 0.0f;
                    ov_3[52] = 0.0f;
                    ov_3[53] = 0.0f;
                    ov_3[54] = 0.0f;
                    ov_3[55] = 0.0f;
                    ov_3[56] = 0.0f;
                    ov_3[57] = 0.0f;
                    ov_3[58] = 0.0f;
                    ov_3[59] = 0.0f;
                    ov_3[60] = 0.0f;
                    ov_3[61] = 0.0f;
                    ov_3[62] = 0.0f;
                    ov_3[63] = 0.0f;
                }
                const float2 _scale2_15 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_3)[_ls], _scale2_15);
                unsigned int hi_11[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 0], ov_3[_lp*2+1 + 0]));
                    hi_11[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (192)), "r"(hi_11[(0) + 0]), "r"(hi_11[(0) + 1]), "r"(hi_11[(0) + 2]), "r"(hi_11[(0) + 3]), "r"(hi_11[(0) + 4]), "r"(hi_11[(0) + 5]), "r"(hi_11[(0) + 6]), "r"(hi_11[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_11[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_11[_pair]) << 16);
                    (&unpacked_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_11[_pair]) & 0xffff0000u);
                }
                float res_11[16];
                res_11[0] = ov_3[0] - unpacked_11[0];
                res_11[1] = ov_3[1] - unpacked_11[1];
                res_11[2] = ov_3[2] - unpacked_11[2];
                res_11[3] = ov_3[3] - unpacked_11[3];
                res_11[4] = ov_3[4] - unpacked_11[4];
                res_11[5] = ov_3[5] - unpacked_11[5];
                res_11[6] = ov_3[6] - unpacked_11[6];
                res_11[7] = ov_3[7] - unpacked_11[7];
                res_11[8] = ov_3[8] - unpacked_11[8];
                res_11[9] = ov_3[9] - unpacked_11[9];
                res_11[10] = ov_3[10] - unpacked_11[10];
                res_11[11] = ov_3[11] - unpacked_11[11];
                res_11[12] = ov_3[12] - unpacked_11[12];
                res_11[13] = ov_3[13] - unpacked_11[13];
                res_11[14] = ov_3[14] - unpacked_11[14];
                res_11[15] = ov_3[15] - unpacked_11[15];
                unsigned int lo_11[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_11[_lp*2 + 0], res_11[_lp*2+1 + 0]));
                    lo_11[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (192)), "r"(lo_11[(0) + 0]), "r"(lo_11[(0) + 1]), "r"(lo_11[(0) + 2]), "r"(lo_11[(0) + 3]), "r"(lo_11[(0) + 4]), "r"(lo_11[(0) + 5]), "r"(lo_11[(0) + 6]), "r"(lo_11[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_12[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 16], ov_3[_lp*2+1 + 16]));
                    hi_12[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (208)), "r"(hi_12[(0) + 0]), "r"(hi_12[(0) + 1]), "r"(hi_12[(0) + 2]), "r"(hi_12[(0) + 3]), "r"(hi_12[(0) + 4]), "r"(hi_12[(0) + 5]), "r"(hi_12[(0) + 6]), "r"(hi_12[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_12[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_12[_pair]) << 16);
                    (&unpacked_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_12[_pair]) & 0xffff0000u);
                }
                float res_12[16];
                res_12[0] = ov_3[16] - unpacked_12[0];
                res_12[1] = ov_3[17] - unpacked_12[1];
                res_12[2] = ov_3[18] - unpacked_12[2];
                res_12[3] = ov_3[19] - unpacked_12[3];
                res_12[4] = ov_3[20] - unpacked_12[4];
                res_12[5] = ov_3[21] - unpacked_12[5];
                res_12[6] = ov_3[22] - unpacked_12[6];
                res_12[7] = ov_3[23] - unpacked_12[7];
                res_12[8] = ov_3[24] - unpacked_12[8];
                res_12[9] = ov_3[25] - unpacked_12[9];
                res_12[10] = ov_3[26] - unpacked_12[10];
                res_12[11] = ov_3[27] - unpacked_12[11];
                res_12[12] = ov_3[28] - unpacked_12[12];
                res_12[13] = ov_3[29] - unpacked_12[13];
                res_12[14] = ov_3[30] - unpacked_12[14];
                res_12[15] = ov_3[31] - unpacked_12[15];
                unsigned int lo_12[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_12[_lp*2 + 0], res_12[_lp*2+1 + 0]));
                    lo_12[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (208)), "r"(lo_12[(0) + 0]), "r"(lo_12[(0) + 1]), "r"(lo_12[(0) + 2]), "r"(lo_12[(0) + 3]), "r"(lo_12[(0) + 4]), "r"(lo_12[(0) + 5]), "r"(lo_12[(0) + 6]), "r"(lo_12[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_13[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 32], ov_3[_lp*2+1 + 32]));
                    hi_13[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (224)), "r"(hi_13[(0) + 0]), "r"(hi_13[(0) + 1]), "r"(hi_13[(0) + 2]), "r"(hi_13[(0) + 3]), "r"(hi_13[(0) + 4]), "r"(hi_13[(0) + 5]), "r"(hi_13[(0) + 6]), "r"(hi_13[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_13[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_13[_pair]) << 16);
                    (&unpacked_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_13[_pair]) & 0xffff0000u);
                }
                float res_13[16];
                res_13[0] = ov_3[32] - unpacked_13[0];
                res_13[1] = ov_3[33] - unpacked_13[1];
                res_13[2] = ov_3[34] - unpacked_13[2];
                res_13[3] = ov_3[35] - unpacked_13[3];
                res_13[4] = ov_3[36] - unpacked_13[4];
                res_13[5] = ov_3[37] - unpacked_13[5];
                res_13[6] = ov_3[38] - unpacked_13[6];
                res_13[7] = ov_3[39] - unpacked_13[7];
                res_13[8] = ov_3[40] - unpacked_13[8];
                res_13[9] = ov_3[41] - unpacked_13[9];
                res_13[10] = ov_3[42] - unpacked_13[10];
                res_13[11] = ov_3[43] - unpacked_13[11];
                res_13[12] = ov_3[44] - unpacked_13[12];
                res_13[13] = ov_3[45] - unpacked_13[13];
                res_13[14] = ov_3[46] - unpacked_13[14];
                res_13[15] = ov_3[47] - unpacked_13[15];
                unsigned int lo_13[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_13[_lp*2 + 0], res_13[_lp*2+1 + 0]));
                    lo_13[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (224)), "r"(lo_13[(0) + 0]), "r"(lo_13[(0) + 1]), "r"(lo_13[(0) + 2]), "r"(lo_13[(0) + 3]), "r"(lo_13[(0) + 4]), "r"(lo_13[(0) + 5]), "r"(lo_13[(0) + 6]), "r"(lo_13[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                unsigned int hi_14[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 48], ov_3[_lp*2+1 + 48]));
                    hi_14[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((out + row_out) + (240)), "r"(hi_14[(0) + 0]), "r"(hi_14[(0) + 1]), "r"(hi_14[(0) + 2]), "r"(hi_14[(0) + 3]), "r"(hi_14[(0) + 4]), "r"(hi_14[(0) + 5]), "r"(hi_14[(0) + 6]), "r"(hi_14[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                float unpacked_14[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpacked_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hi_14[_pair]) << 16);
                    (&unpacked_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hi_14[_pair]) & 0xffff0000u);
                }
                float res_14[16];
                res_14[0] = ov_3[48] - unpacked_14[0];
                res_14[1] = ov_3[49] - unpacked_14[1];
                res_14[2] = ov_3[50] - unpacked_14[2];
                res_14[3] = ov_3[51] - unpacked_14[3];
                res_14[4] = ov_3[52] - unpacked_14[4];
                res_14[5] = ov_3[53] - unpacked_14[5];
                res_14[6] = ov_3[54] - unpacked_14[6];
                res_14[7] = ov_3[55] - unpacked_14[7];
                res_14[8] = ov_3[56] - unpacked_14[8];
                res_14[9] = ov_3[57] - unpacked_14[9];
                res_14[10] = ov_3[58] - unpacked_14[10];
                res_14[11] = ov_3[59] - unpacked_14[11];
                res_14[12] = ov_3[60] - unpacked_14[12];
                res_14[13] = ov_3[61] - unpacked_14[13];
                res_14[14] = ov_3[62] - unpacked_14[14];
                res_14[15] = ov_3[63] - unpacked_14[15];
                unsigned int lo_14[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_14[_lp*2 + 0], res_14[_lp*2+1 + 0]));
                    lo_14[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"((o_lo + row_out) + (240)), "r"(lo_14[(0) + 0]), "r"(lo_14[(0) + 1]), "r"(lo_14[(0) + 2]), "r"(lo_14[(0) + 3]), "r"(lo_14[(0) + 4]), "r"(lo_14[(0) + 5]), "r"(lo_14[(0) + 6]), "r"(lo_14[(0) + 7]), "r"((unsigned int)(live != 0)) : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_free_addr), "r"(0) : "memory");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
        }
    }
    // ---- Role: producer_k ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // producer_k_main
            int cid_1 = blockIdx.x / 2;
            int nclu_1 = (int)gridDim.x / 2;
            int ntok_1 = (num_queries - cid_1 + nclu_1 - 1) / nclu_1;
            int base_1 = 0;
            int warp_0 = warp - 4;
            int aligned = ((idx_stride | indices_offset) & 3) == 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int it_1 = 0; it_1 < ntok_1; it_1++) {
                    int query_1 = cid_1 + it_1 * nclu_1;
                    int _stage_len_ready_1 = (int)((unsigned int)it_1 % 2);
                    int _parity_len_ready_1 = (int)((unsigned int)it_1 / 2 % 2);
                    mbarrier_wait(len_ready_addr + (_stage_len_ready_1) * 8, _parity_len_ready_1);
                    int _max_0_1 = ((len_cells[it_1 % 2]) > (0) ? (len_cells[it_1 % 2]) : (0));
                    int active_1 = ((_max_0_1) < (topk) ? (_max_0_1) : (topk));
                    mbarrier_arrive(len_free_addr + ((int)((unsigned int)it_1 % 2)) * 8);
                    int tiles_1 = (((active_1 + 127) / 128) > (1) ? ((active_1 + 127) / 128) : (1));
                    long long row_base = (long long)indices_offset + (long long)query_1 * (long long)idx_stride;
                    #pragma unroll 1
                    for (int n_1 = 0; n_1 < tiles_1; n_1++) {
                        int idx[16];
                        int whole = active_1 >= n_1 * 128 + 128 && aligned != 0;
                        if (whole != 0) {
                            int _vec_load_0[4];
                            {
                                const int4* _ivptr_1 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(n_1 * 128 + cta_rank * 64 + warp_0 * 4)) + 0);
                                int4 _ivld_1;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_1.x), "=r"(_ivld_1.y), "=r"(_ivld_1.z), "=r"(_ivld_1.w) : "l"((const void*)(_ivptr_1)) : "memory");
                                _vec_load_0[0 + 0] = _ivld_1.x;
                                _vec_load_0[0 + 1] = _ivld_1.y;
                                _vec_load_0[0 + 2] = _ivld_1.z;
                                _vec_load_0[0 + 3] = _ivld_1.w;
                            }
                            idx[0] = _vec_load_0[0];
                            idx[1] = _vec_load_0[1];
                            idx[2] = _vec_load_0[2];
                            idx[3] = _vec_load_0[3];
                            int _vec_load_1[4];
                            {
                                const int4* _ivptr_2 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 16)) + 0);
                                int4 _ivld_2;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_2.x), "=r"(_ivld_2.y), "=r"(_ivld_2.z), "=r"(_ivld_2.w) : "l"((const void*)(_ivptr_2)) : "memory");
                                _vec_load_1[0 + 0] = _ivld_2.x;
                                _vec_load_1[0 + 1] = _ivld_2.y;
                                _vec_load_1[0 + 2] = _ivld_2.z;
                                _vec_load_1[0 + 3] = _ivld_2.w;
                            }
                            idx[4] = _vec_load_1[0];
                            idx[5] = _vec_load_1[1];
                            idx[6] = _vec_load_1[2];
                            idx[7] = _vec_load_1[3];
                            int _vec_load_2[4];
                            {
                                const int4* _ivptr_3 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 32)) + 0);
                                int4 _ivld_3;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_3.x), "=r"(_ivld_3.y), "=r"(_ivld_3.z), "=r"(_ivld_3.w) : "l"((const void*)(_ivptr_3)) : "memory");
                                _vec_load_2[0 + 0] = _ivld_3.x;
                                _vec_load_2[0 + 1] = _ivld_3.y;
                                _vec_load_2[0 + 2] = _ivld_3.z;
                                _vec_load_2[0 + 3] = _ivld_3.w;
                            }
                            idx[8] = _vec_load_2[0];
                            idx[9] = _vec_load_2[1];
                            idx[10] = _vec_load_2[2];
                            idx[11] = _vec_load_2[3];
                            int _vec_load_3[4];
                            {
                                const int4* _ivptr_4 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 48)) + 0);
                                int4 _ivld_4;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_4.x), "=r"(_ivld_4.y), "=r"(_ivld_4.z), "=r"(_ivld_4.w) : "l"((const void*)(_ivptr_4)) : "memory");
                                _vec_load_3[0 + 0] = _ivld_4.x;
                                _vec_load_3[0 + 1] = _ivld_4.y;
                                _vec_load_3[0 + 2] = _ivld_4.z;
                                _vec_load_3[0 + 3] = _ivld_4.w;
                            }
                            idx[12] = _vec_load_3[0];
                            idx[13] = _vec_load_3[1];
                            idx[14] = _vec_load_3[2];
                            idx[15] = _vec_load_3[3];
                        } else {
                            int position_0 = n_1 * 128 + cta_rank * 64 + warp_0 * 4;
                            int _min_5 = ((position_0) < (topk - 1) ? (position_0) : (topk - 1));
                            int value_0 = indices[row_base + (long long)_min_5];
                            if (position_0 >= active_1) {
                                value_0 = -1;
                            }
                            idx[0] = value_0;
                            int position_1 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 1;
                            int _min_6 = ((position_1) < (topk - 1) ? (position_1) : (topk - 1));
                            int value_1 = indices[row_base + (long long)_min_6];
                            if (position_1 >= active_1) {
                                value_1 = -1;
                            }
                            idx[1] = value_1;
                            int position_2 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 2;
                            int _min_7 = ((position_2) < (topk - 1) ? (position_2) : (topk - 1));
                            int value_2 = indices[row_base + (long long)_min_7];
                            if (position_2 >= active_1) {
                                value_2 = -1;
                            }
                            idx[2] = value_2;
                            int position_3 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 3;
                            int _min_8 = ((position_3) < (topk - 1) ? (position_3) : (topk - 1));
                            int value_3 = indices[row_base + (long long)_min_8];
                            if (position_3 >= active_1) {
                                value_3 = -1;
                            }
                            idx[3] = value_3;
                            int position_4 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 16;
                            int _min_9 = ((position_4) < (topk - 1) ? (position_4) : (topk - 1));
                            int value_5 = indices[row_base + (long long)_min_9];
                            if (position_4 >= active_1) {
                                value_5 = -1;
                            }
                            idx[4] = value_5;
                            int position_5 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 17;
                            int _min_10 = ((position_5) < (topk - 1) ? (position_5) : (topk - 1));
                            int value_6 = indices[row_base + (long long)_min_10];
                            if (position_5 >= active_1) {
                                value_6 = -1;
                            }
                            idx[5] = value_6;
                            int position_6 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 18;
                            int _min_11 = ((position_6) < (topk - 1) ? (position_6) : (topk - 1));
                            int value_7 = indices[row_base + (long long)_min_11];
                            if (position_6 >= active_1) {
                                value_7 = -1;
                            }
                            idx[6] = value_7;
                            int position_7 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 19;
                            int _min_12 = ((position_7) < (topk - 1) ? (position_7) : (topk - 1));
                            int value_8 = indices[row_base + (long long)_min_12];
                            if (position_7 >= active_1) {
                                value_8 = -1;
                            }
                            idx[7] = value_8;
                            int position_8 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 32;
                            int _min_13 = ((position_8) < (topk - 1) ? (position_8) : (topk - 1));
                            int value_9 = indices[row_base + (long long)_min_13];
                            if (position_8 >= active_1) {
                                value_9 = -1;
                            }
                            idx[8] = value_9;
                            int position_9 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 33;
                            int _min_14 = ((position_9) < (topk - 1) ? (position_9) : (topk - 1));
                            int value_10 = indices[row_base + (long long)_min_14];
                            if (position_9 >= active_1) {
                                value_10 = -1;
                            }
                            idx[9] = value_10;
                            int position_10 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 34;
                            int _min_15 = ((position_10) < (topk - 1) ? (position_10) : (topk - 1));
                            int value_11 = indices[row_base + (long long)_min_15];
                            if (position_10 >= active_1) {
                                value_11 = -1;
                            }
                            idx[10] = value_11;
                            int position_11 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 35;
                            int _min_16 = ((position_11) < (topk - 1) ? (position_11) : (topk - 1));
                            int value_12 = indices[row_base + (long long)_min_16];
                            if (position_11 >= active_1) {
                                value_12 = -1;
                            }
                            idx[11] = value_12;
                            int position_12 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 48;
                            int _min_17 = ((position_12) < (topk - 1) ? (position_12) : (topk - 1));
                            int value_13 = indices[row_base + (long long)_min_17];
                            if (position_12 >= active_1) {
                                value_13 = -1;
                            }
                            idx[12] = value_13;
                            int position_13 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 49;
                            int _min_18 = ((position_13) < (topk - 1) ? (position_13) : (topk - 1));
                            int value_14 = indices[row_base + (long long)_min_18];
                            if (position_13 >= active_1) {
                                value_14 = -1;
                            }
                            idx[13] = value_14;
                            int position_14 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 50;
                            int _min_19 = ((position_14) < (topk - 1) ? (position_14) : (topk - 1));
                            int value_15 = indices[row_base + (long long)_min_19];
                            if (position_14 >= active_1) {
                                value_15 = -1;
                            }
                            idx[14] = value_15;
                            int position_15 = n_1 * 128 + cta_rank * 64 + warp_0 * 4 + 51;
                            int _min_20 = ((position_15) < (topk - 1) ? (position_15) : (topk - 1));
                            int value_16 = indices[row_base + (long long)_min_20];
                            if (position_15 >= active_1) {
                                value_16 = -1;
                            }
                            idx[15] = value_16;
                        }
                        if (base_1 + n_1 > 0) {
                            int _stage_qk_part = (int)((unsigned int)(base_1 + n_1 - 1) % 2);
                            int _parity_qk_part = (int)((unsigned int)(base_1 + n_1 - 1) / 2 % 2);
                            mbarrier_wait(qk_part_addr + (_stage_qk_part) * 8, _parity_qk_part);
                        }
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048), "l"((&kv_latent)), "r"(0), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096), "l"((&kv_latent)), "r"(0), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144), "l"((&kv_latent)), "r"(0), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_a_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        if (base_1 + n_1 > 0) {
                            int _stage_qk_done_1 = (int)((unsigned int)(base_1 + n_1 - 1) % 2);
                            int _parity_qk_done_1 = (int)((unsigned int)(base_1 + n_1 - 1) / 2 % 2);
                            mbarrier_wait(qk_done_addr + (_stage_qk_done_1) * 8, _parity_qk_done_1);
                        }
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 65536), "l"((&k_rope)), "r"(0), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 2048 + 65536), "l"((&k_rope)), "r"(0), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 4096 + 65536), "l"((&k_rope)), "r"(0), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_slot_addr + (unsigned int)(warp_0 * 512) + 6144 + 65536), "l"((&k_rope)), "r"(0), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(((k_b_addr + ((int)((unsigned int)(base_1 + n_1) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                    base_1 += tiles_1;
                }
                int _stage_qk_part_1 = (int)((unsigned int)(base_1 + 1) % 2);
                int _parity_qk_part_1 = (int)((unsigned int)(base_1 + 1) / 2 % 2) ^ 1;
                mbarrier_wait(qk_part_addr + (_stage_qk_part_1) * 8, _parity_qk_part_1);
            }
        }
    }
    // ---- Role: producer_v ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // producer_v_main
            int cid_2 = blockIdx.x / 2;
            int nclu_2 = (int)gridDim.x / 2;
            int ntok_2 = (num_queries - cid_2 + nclu_2 - 1) / nclu_2;
            int base_2 = 0;
            int warp_0_1 = warp - 8;
            int aligned_1 = ((idx_stride | indices_offset) & 3) == 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int it_2 = 0; it_2 < ntok_2; it_2++) {
                    int query_2 = cid_2 + it_2 * nclu_2;
                    int _stage_len_ready_2 = (int)((unsigned int)it_2 % 2);
                    int _parity_len_ready_2 = (int)((unsigned int)it_2 / 2 % 2);
                    mbarrier_wait(len_ready_addr + (_stage_len_ready_2) * 8, _parity_len_ready_2);
                    int _max_0_2 = ((len_cells[it_2 % 2]) > (0) ? (len_cells[it_2 % 2]) : (0));
                    int active_2 = ((_max_0_2) < (topk) ? (_max_0_2) : (topk));
                    mbarrier_arrive(len_free_addr + ((int)((unsigned int)it_2 % 2)) * 8);
                    int tiles_2 = (((active_2 + 127) / 128) > (1) ? ((active_2 + 127) / 128) : (1));
                    long long row_base_1 = (long long)indices_offset + (long long)query_2 * (long long)idx_stride;
                    #pragma unroll 1
                    for (int n_2 = 0; n_2 < tiles_2; n_2++) {
                        int idx_0[16];
                        int whole_0 = active_2 >= n_2 * 128 + 64 && aligned_1 != 0;
                        if (whole_0 != 0) {
                            int _vec_load_5[4];
                            {
                                const int4* _ivptr_1 = reinterpret_cast<const int4*>(indices + (row_base_1 + (long long)(n_2 * 128 + warp_0_1 * 4)) + 0);
                                int4 _ivld_1;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_1.x), "=r"(_ivld_1.y), "=r"(_ivld_1.z), "=r"(_ivld_1.w) : "l"((const void*)(_ivptr_1)) : "memory");
                                _vec_load_5[0 + 0] = _ivld_1.x;
                                _vec_load_5[0 + 1] = _ivld_1.y;
                                _vec_load_5[0 + 2] = _ivld_1.z;
                                _vec_load_5[0 + 3] = _ivld_1.w;
                            }
                            idx_0[0] = _vec_load_5[0];
                            idx_0[1] = _vec_load_5[1];
                            idx_0[2] = _vec_load_5[2];
                            idx_0[3] = _vec_load_5[3];
                            int _vec_load_7[4];
                            {
                                const int4* _ivptr_2 = reinterpret_cast<const int4*>(indices + (row_base_1 + (long long)(n_2 * 128 + warp_0_1 * 4 + 16)) + 0);
                                int4 _ivld_2;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_2.x), "=r"(_ivld_2.y), "=r"(_ivld_2.z), "=r"(_ivld_2.w) : "l"((const void*)(_ivptr_2)) : "memory");
                                _vec_load_7[0 + 0] = _ivld_2.x;
                                _vec_load_7[0 + 1] = _ivld_2.y;
                                _vec_load_7[0 + 2] = _ivld_2.z;
                                _vec_load_7[0 + 3] = _ivld_2.w;
                            }
                            idx_0[4] = _vec_load_7[0];
                            idx_0[5] = _vec_load_7[1];
                            idx_0[6] = _vec_load_7[2];
                            idx_0[7] = _vec_load_7[3];
                            int _vec_load_8[4];
                            {
                                const int4* _ivptr_3 = reinterpret_cast<const int4*>(indices + (row_base_1 + (long long)(n_2 * 128 + warp_0_1 * 4 + 32)) + 0);
                                int4 _ivld_3;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_3.x), "=r"(_ivld_3.y), "=r"(_ivld_3.z), "=r"(_ivld_3.w) : "l"((const void*)(_ivptr_3)) : "memory");
                                _vec_load_8[0 + 0] = _ivld_3.x;
                                _vec_load_8[0 + 1] = _ivld_3.y;
                                _vec_load_8[0 + 2] = _ivld_3.z;
                                _vec_load_8[0 + 3] = _ivld_3.w;
                            }
                            idx_0[8] = _vec_load_8[0];
                            idx_0[9] = _vec_load_8[1];
                            idx_0[10] = _vec_load_8[2];
                            idx_0[11] = _vec_load_8[3];
                            int _vec_load_9[4];
                            {
                                const int4* _ivptr_4 = reinterpret_cast<const int4*>(indices + (row_base_1 + (long long)(n_2 * 128 + warp_0_1 * 4 + 48)) + 0);
                                int4 _ivld_4;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_4.x), "=r"(_ivld_4.y), "=r"(_ivld_4.z), "=r"(_ivld_4.w) : "l"((const void*)(_ivptr_4)) : "memory");
                                _vec_load_9[0 + 0] = _ivld_4.x;
                                _vec_load_9[0 + 1] = _ivld_4.y;
                                _vec_load_9[0 + 2] = _ivld_4.z;
                                _vec_load_9[0 + 3] = _ivld_4.w;
                            }
                            idx_0[12] = _vec_load_9[0];
                            idx_0[13] = _vec_load_9[1];
                            idx_0[14] = _vec_load_9[2];
                            idx_0[15] = _vec_load_9[3];
                        } else {
                            int position_16 = n_2 * 128 + warp_0_1 * 4;
                            int _min_5_1 = ((position_16) < (topk - 1) ? (position_16) : (topk - 1));
                            int value_17 = indices[row_base_1 + (long long)_min_5_1];
                            if (position_16 >= active_2) {
                                value_17 = -1;
                            }
                            idx_0[0] = value_17;
                            int position_17 = n_2 * 128 + warp_0_1 * 4 + 1;
                            int _min_6_1 = ((position_17) < (topk - 1) ? (position_17) : (topk - 1));
                            int value_18 = indices[row_base_1 + (long long)_min_6_1];
                            if (position_17 >= active_2) {
                                value_18 = -1;
                            }
                            idx_0[1] = value_18;
                            int position_18 = n_2 * 128 + warp_0_1 * 4 + 2;
                            int _min_7_1 = ((position_18) < (topk - 1) ? (position_18) : (topk - 1));
                            int value_19 = indices[row_base_1 + (long long)_min_7_1];
                            if (position_18 >= active_2) {
                                value_19 = -1;
                            }
                            idx_0[2] = value_19;
                            int position_19 = n_2 * 128 + warp_0_1 * 4 + 3;
                            int _min_8_1 = ((position_19) < (topk - 1) ? (position_19) : (topk - 1));
                            int value_20 = indices[row_base_1 + (long long)_min_8_1];
                            if (position_19 >= active_2) {
                                value_20 = -1;
                            }
                            idx_0[3] = value_20;
                            int position_20 = n_2 * 128 + warp_0_1 * 4 + 16;
                            int _min_9_1 = ((position_20) < (topk - 1) ? (position_20) : (topk - 1));
                            int value_21 = indices[row_base_1 + (long long)_min_9_1];
                            if (position_20 >= active_2) {
                                value_21 = -1;
                            }
                            idx_0[4] = value_21;
                            int position_21 = n_2 * 128 + warp_0_1 * 4 + 17;
                            int _min_10_1 = ((position_21) < (topk - 1) ? (position_21) : (topk - 1));
                            int value_22 = indices[row_base_1 + (long long)_min_10_1];
                            if (position_21 >= active_2) {
                                value_22 = -1;
                            }
                            idx_0[5] = value_22;
                            int position_22 = n_2 * 128 + warp_0_1 * 4 + 18;
                            int _min_11_1 = ((position_22) < (topk - 1) ? (position_22) : (topk - 1));
                            int value_23 = indices[row_base_1 + (long long)_min_11_1];
                            if (position_22 >= active_2) {
                                value_23 = -1;
                            }
                            idx_0[6] = value_23;
                            int position_23 = n_2 * 128 + warp_0_1 * 4 + 19;
                            int _min_12_1 = ((position_23) < (topk - 1) ? (position_23) : (topk - 1));
                            int value_24 = indices[row_base_1 + (long long)_min_12_1];
                            if (position_23 >= active_2) {
                                value_24 = -1;
                            }
                            idx_0[7] = value_24;
                            int position_24 = n_2 * 128 + warp_0_1 * 4 + 32;
                            int _min_13_1 = ((position_24) < (topk - 1) ? (position_24) : (topk - 1));
                            int value_25 = indices[row_base_1 + (long long)_min_13_1];
                            if (position_24 >= active_2) {
                                value_25 = -1;
                            }
                            idx_0[8] = value_25;
                            int position_25 = n_2 * 128 + warp_0_1 * 4 + 33;
                            int _min_14_1 = ((position_25) < (topk - 1) ? (position_25) : (topk - 1));
                            int value_26 = indices[row_base_1 + (long long)_min_14_1];
                            if (position_25 >= active_2) {
                                value_26 = -1;
                            }
                            idx_0[9] = value_26;
                            int position_26 = n_2 * 128 + warp_0_1 * 4 + 34;
                            int _min_15_1 = ((position_26) < (topk - 1) ? (position_26) : (topk - 1));
                            int value_27 = indices[row_base_1 + (long long)_min_15_1];
                            if (position_26 >= active_2) {
                                value_27 = -1;
                            }
                            idx_0[10] = value_27;
                            int position_27 = n_2 * 128 + warp_0_1 * 4 + 35;
                            int _min_16_1 = ((position_27) < (topk - 1) ? (position_27) : (topk - 1));
                            int value_28 = indices[row_base_1 + (long long)_min_16_1];
                            if (position_27 >= active_2) {
                                value_28 = -1;
                            }
                            idx_0[11] = value_28;
                            int position_28 = n_2 * 128 + warp_0_1 * 4 + 48;
                            int _min_17_1 = ((position_28) < (topk - 1) ? (position_28) : (topk - 1));
                            int value_29 = indices[row_base_1 + (long long)_min_17_1];
                            if (position_28 >= active_2) {
                                value_29 = -1;
                            }
                            idx_0[12] = value_29;
                            int position_29 = n_2 * 128 + warp_0_1 * 4 + 49;
                            int _min_18_1 = ((position_29) < (topk - 1) ? (position_29) : (topk - 1));
                            int value_30 = indices[row_base_1 + (long long)_min_18_1];
                            if (position_29 >= active_2) {
                                value_30 = -1;
                            }
                            idx_0[13] = value_30;
                            int position_30 = n_2 * 128 + warp_0_1 * 4 + 50;
                            int _min_19_1 = ((position_30) < (topk - 1) ? (position_30) : (topk - 1));
                            int value_31 = indices[row_base_1 + (long long)_min_19_1];
                            if (position_30 >= active_2) {
                                value_31 = -1;
                            }
                            idx_0[14] = value_31;
                            int position_31 = n_2 * 128 + warp_0_1 * 4 + 51;
                            int _min_20_1 = ((position_31) < (topk - 1) ? (position_31) : (topk - 1));
                            int value_32 = indices[row_base_1 + (long long)_min_20_1];
                            if (position_31 >= active_2) {
                                value_32 = -1;
                            }
                            idx_0[15] = value_32;
                        }
                        if (base_2 + n_2 > 0) {
                            int _stage_pv_a = (int)((unsigned int)(base_2 + n_2 - 1) % 2);
                            int _parity_pv_a = (int)((unsigned int)(base_2 + n_2 - 1) / 2 % 2);
                            mbarrier_wait(pv_a_addr + (_stage_pv_a) * 8, _parity_pv_a);
                        }
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512)), "l"((&kv_latent)), "r"(cta_rank * 256), "r"(idx_0[0]), "r"(idx_0[1]), "r"(idx_0[2]), "r"(idx_0[3]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 16384), "l"((&kv_latent)), "r"(cta_rank * 256 + 64), "r"(idx_0[0]), "r"(idx_0[1]), "r"(idx_0[2]), "r"(idx_0[3]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 32768), "l"((&kv_latent)), "r"(cta_rank * 256 + 128), "r"(idx_0[0]), "r"(idx_0[1]), "r"(idx_0[2]), "r"(idx_0[3]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 49152), "l"((&kv_latent)), "r"(cta_rank * 256 + 192), "r"(idx_0[0]), "r"(idx_0[1]), "r"(idx_0[2]), "r"(idx_0[3]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 2048), "l"((&kv_latent)), "r"(cta_rank * 256), "r"(idx_0[4]), "r"(idx_0[5]), "r"(idx_0[6]), "r"(idx_0[7]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 2048 + 16384), "l"((&kv_latent)), "r"(cta_rank * 256 + 64), "r"(idx_0[4]), "r"(idx_0[5]), "r"(idx_0[6]), "r"(idx_0[7]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 2048 + 32768), "l"((&kv_latent)), "r"(cta_rank * 256 + 128), "r"(idx_0[4]), "r"(idx_0[5]), "r"(idx_0[6]), "r"(idx_0[7]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 2048 + 49152), "l"((&kv_latent)), "r"(cta_rank * 256 + 192), "r"(idx_0[4]), "r"(idx_0[5]), "r"(idx_0[6]), "r"(idx_0[7]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 4096), "l"((&kv_latent)), "r"(cta_rank * 256), "r"(idx_0[8]), "r"(idx_0[9]), "r"(idx_0[10]), "r"(idx_0[11]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 4096 + 16384), "l"((&kv_latent)), "r"(cta_rank * 256 + 64), "r"(idx_0[8]), "r"(idx_0[9]), "r"(idx_0[10]), "r"(idx_0[11]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 4096 + 32768), "l"((&kv_latent)), "r"(cta_rank * 256 + 128), "r"(idx_0[8]), "r"(idx_0[9]), "r"(idx_0[10]), "r"(idx_0[11]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 4096 + 49152), "l"((&kv_latent)), "r"(cta_rank * 256 + 192), "r"(idx_0[8]), "r"(idx_0[9]), "r"(idx_0[10]), "r"(idx_0[11]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 6144), "l"((&kv_latent)), "r"(cta_rank * 256), "r"(idx_0[12]), "r"(idx_0[13]), "r"(idx_0[14]), "r"(idx_0[15]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 6144 + 16384), "l"((&kv_latent)), "r"(cta_rank * 256 + 64), "r"(idx_0[12]), "r"(idx_0[13]), "r"(idx_0[14]), "r"(idx_0[15]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 6144 + 32768), "l"((&kv_latent)), "r"(cta_rank * 256 + 128), "r"(idx_0[12]), "r"(idx_0[13]), "r"(idx_0[14]), "r"(idx_0[15]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + (unsigned int)(warp_0_1 * 512) + 6144 + 49152), "l"((&kv_latent)), "r"(cta_rank * 256 + 192), "r"(idx_0[12]), "r"(idx_0[13]), "r"(idx_0[14]), "r"(idx_0[15]), "r"(((v_a_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        int idx_1[16];
                        int whole_1 = active_2 >= n_2 * 128 + 64 + 64 && aligned_1 != 0;
                        if (whole_1 != 0) {
                            int _vec_load_10[4];
                            {
                                const int4* _ivptr_21 = reinterpret_cast<const int4*>(indices + (row_base_1 + (long long)(n_2 * 128 + 64 + warp_0_1 * 4)) + 0);
                                int4 _ivld_21;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_21.x), "=r"(_ivld_21.y), "=r"(_ivld_21.z), "=r"(_ivld_21.w) : "l"((const void*)(_ivptr_21)) : "memory");
                                _vec_load_10[0 + 0] = _ivld_21.x;
                                _vec_load_10[0 + 1] = _ivld_21.y;
                                _vec_load_10[0 + 2] = _ivld_21.z;
                                _vec_load_10[0 + 3] = _ivld_21.w;
                            }
                            idx_1[0] = _vec_load_10[0];
                            idx_1[1] = _vec_load_10[1];
                            idx_1[2] = _vec_load_10[2];
                            idx_1[3] = _vec_load_10[3];
                            int _vec_load_11[4];
                            {
                                const int4* _ivptr_22 = reinterpret_cast<const int4*>(indices + (row_base_1 + (long long)(n_2 * 128 + 64 + warp_0_1 * 4 + 16)) + 0);
                                int4 _ivld_22;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_22.x), "=r"(_ivld_22.y), "=r"(_ivld_22.z), "=r"(_ivld_22.w) : "l"((const void*)(_ivptr_22)) : "memory");
                                _vec_load_11[0 + 0] = _ivld_22.x;
                                _vec_load_11[0 + 1] = _ivld_22.y;
                                _vec_load_11[0 + 2] = _ivld_22.z;
                                _vec_load_11[0 + 3] = _ivld_22.w;
                            }
                            idx_1[4] = _vec_load_11[0];
                            idx_1[5] = _vec_load_11[1];
                            idx_1[6] = _vec_load_11[2];
                            idx_1[7] = _vec_load_11[3];
                            int _vec_load_12[4];
                            {
                                const int4* _ivptr_23 = reinterpret_cast<const int4*>(indices + (row_base_1 + (long long)(n_2 * 128 + 64 + warp_0_1 * 4 + 32)) + 0);
                                int4 _ivld_23;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_23.x), "=r"(_ivld_23.y), "=r"(_ivld_23.z), "=r"(_ivld_23.w) : "l"((const void*)(_ivptr_23)) : "memory");
                                _vec_load_12[0 + 0] = _ivld_23.x;
                                _vec_load_12[0 + 1] = _ivld_23.y;
                                _vec_load_12[0 + 2] = _ivld_23.z;
                                _vec_load_12[0 + 3] = _ivld_23.w;
                            }
                            idx_1[8] = _vec_load_12[0];
                            idx_1[9] = _vec_load_12[1];
                            idx_1[10] = _vec_load_12[2];
                            idx_1[11] = _vec_load_12[3];
                            int _vec_load_13[4];
                            {
                                const int4* _ivptr_24 = reinterpret_cast<const int4*>(indices + (row_base_1 + (long long)(n_2 * 128 + 64 + warp_0_1 * 4 + 48)) + 0);
                                int4 _ivld_24;
                                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_ivld_24.x), "=r"(_ivld_24.y), "=r"(_ivld_24.z), "=r"(_ivld_24.w) : "l"((const void*)(_ivptr_24)) : "memory");
                                _vec_load_13[0 + 0] = _ivld_24.x;
                                _vec_load_13[0 + 1] = _ivld_24.y;
                                _vec_load_13[0 + 2] = _ivld_24.z;
                                _vec_load_13[0 + 3] = _ivld_24.w;
                            }
                            idx_1[12] = _vec_load_13[0];
                            idx_1[13] = _vec_load_13[1];
                            idx_1[14] = _vec_load_13[2];
                            idx_1[15] = _vec_load_13[3];
                        } else {
                            int position_32 = n_2 * 128 + 64 + warp_0_1 * 4;
                            int _min_25 = ((position_32) < (topk - 1) ? (position_32) : (topk - 1));
                            int value_33 = indices[row_base_1 + (long long)_min_25];
                            if (position_32 >= active_2) {
                                value_33 = -1;
                            }
                            idx_1[0] = value_33;
                            int position_33 = n_2 * 128 + 64 + warp_0_1 * 4 + 1;
                            int _min_26 = ((position_33) < (topk - 1) ? (position_33) : (topk - 1));
                            int value_34 = indices[row_base_1 + (long long)_min_26];
                            if (position_33 >= active_2) {
                                value_34 = -1;
                            }
                            idx_1[1] = value_34;
                            int position_34 = n_2 * 128 + 64 + warp_0_1 * 4 + 2;
                            int _min_27 = ((position_34) < (topk - 1) ? (position_34) : (topk - 1));
                            int value_35 = indices[row_base_1 + (long long)_min_27];
                            if (position_34 >= active_2) {
                                value_35 = -1;
                            }
                            idx_1[2] = value_35;
                            int position_35 = n_2 * 128 + 64 + warp_0_1 * 4 + 3;
                            int _min_28 = ((position_35) < (topk - 1) ? (position_35) : (topk - 1));
                            int value_36 = indices[row_base_1 + (long long)_min_28];
                            if (position_35 >= active_2) {
                                value_36 = -1;
                            }
                            idx_1[3] = value_36;
                            int position_36 = n_2 * 128 + 64 + warp_0_1 * 4 + 16;
                            int _min_29 = ((position_36) < (topk - 1) ? (position_36) : (topk - 1));
                            int value_37 = indices[row_base_1 + (long long)_min_29];
                            if (position_36 >= active_2) {
                                value_37 = -1;
                            }
                            idx_1[4] = value_37;
                            int position_37 = n_2 * 128 + 64 + warp_0_1 * 4 + 17;
                            int _min_30 = ((position_37) < (topk - 1) ? (position_37) : (topk - 1));
                            int value_38 = indices[row_base_1 + (long long)_min_30];
                            if (position_37 >= active_2) {
                                value_38 = -1;
                            }
                            idx_1[5] = value_38;
                            int position_38 = n_2 * 128 + 64 + warp_0_1 * 4 + 18;
                            int _min_31 = ((position_38) < (topk - 1) ? (position_38) : (topk - 1));
                            int value_39 = indices[row_base_1 + (long long)_min_31];
                            if (position_38 >= active_2) {
                                value_39 = -1;
                            }
                            idx_1[6] = value_39;
                            int position_39 = n_2 * 128 + 64 + warp_0_1 * 4 + 19;
                            int _min_32 = ((position_39) < (topk - 1) ? (position_39) : (topk - 1));
                            int value_40 = indices[row_base_1 + (long long)_min_32];
                            if (position_39 >= active_2) {
                                value_40 = -1;
                            }
                            idx_1[7] = value_40;
                            int position_40 = n_2 * 128 + 64 + warp_0_1 * 4 + 32;
                            int _min_33 = ((position_40) < (topk - 1) ? (position_40) : (topk - 1));
                            int value_41 = indices[row_base_1 + (long long)_min_33];
                            if (position_40 >= active_2) {
                                value_41 = -1;
                            }
                            idx_1[8] = value_41;
                            int position_41 = n_2 * 128 + 64 + warp_0_1 * 4 + 33;
                            int _min_34 = ((position_41) < (topk - 1) ? (position_41) : (topk - 1));
                            int value_42 = indices[row_base_1 + (long long)_min_34];
                            if (position_41 >= active_2) {
                                value_42 = -1;
                            }
                            idx_1[9] = value_42;
                            int position_42 = n_2 * 128 + 64 + warp_0_1 * 4 + 34;
                            int _min_35 = ((position_42) < (topk - 1) ? (position_42) : (topk - 1));
                            int value_43 = indices[row_base_1 + (long long)_min_35];
                            if (position_42 >= active_2) {
                                value_43 = -1;
                            }
                            idx_1[10] = value_43;
                            int position_43 = n_2 * 128 + 64 + warp_0_1 * 4 + 35;
                            int _min_36 = ((position_43) < (topk - 1) ? (position_43) : (topk - 1));
                            int value_44 = indices[row_base_1 + (long long)_min_36];
                            if (position_43 >= active_2) {
                                value_44 = -1;
                            }
                            idx_1[11] = value_44;
                            int position_44 = n_2 * 128 + 64 + warp_0_1 * 4 + 48;
                            int _min_37 = ((position_44) < (topk - 1) ? (position_44) : (topk - 1));
                            int value_45 = indices[row_base_1 + (long long)_min_37];
                            if (position_44 >= active_2) {
                                value_45 = -1;
                            }
                            idx_1[12] = value_45;
                            int position_45 = n_2 * 128 + 64 + warp_0_1 * 4 + 49;
                            int _min_38 = ((position_45) < (topk - 1) ? (position_45) : (topk - 1));
                            int value_46 = indices[row_base_1 + (long long)_min_38];
                            if (position_45 >= active_2) {
                                value_46 = -1;
                            }
                            idx_1[13] = value_46;
                            int position_46 = n_2 * 128 + 64 + warp_0_1 * 4 + 50;
                            int _min_39 = ((position_46) < (topk - 1) ? (position_46) : (topk - 1));
                            int value_47 = indices[row_base_1 + (long long)_min_39];
                            if (position_46 >= active_2) {
                                value_47 = -1;
                            }
                            idx_1[14] = value_47;
                            int position_47 = n_2 * 128 + 64 + warp_0_1 * 4 + 51;
                            int _min_40 = ((position_47) < (topk - 1) ? (position_47) : (topk - 1));
                            int value_48 = indices[row_base_1 + (long long)_min_40];
                            if (position_47 >= active_2) {
                                value_48 = -1;
                            }
                            idx_1[15] = value_48;
                        }
                        if (base_2 + n_2 > 0) {
                            int _stage_pv_b_3 = (int)((unsigned int)(base_2 + n_2 - 1) % 2);
                            int _parity_pv_b_3 = (int)((unsigned int)(base_2 + n_2 - 1) / 2 % 2);
                            mbarrier_wait(pv_b_addr + (_stage_pv_b_3) * 8, _parity_pv_b_3);
                        }
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512)), "l"((&kv_latent)), "r"(cta_rank * 256), "r"(idx_1[0]), "r"(idx_1[1]), "r"(idx_1[2]), "r"(idx_1[3]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 16384), "l"((&kv_latent)), "r"(cta_rank * 256 + 64), "r"(idx_1[0]), "r"(idx_1[1]), "r"(idx_1[2]), "r"(idx_1[3]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 32768), "l"((&kv_latent)), "r"(cta_rank * 256 + 128), "r"(idx_1[0]), "r"(idx_1[1]), "r"(idx_1[2]), "r"(idx_1[3]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 49152), "l"((&kv_latent)), "r"(cta_rank * 256 + 192), "r"(idx_1[0]), "r"(idx_1[1]), "r"(idx_1[2]), "r"(idx_1[3]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 2048), "l"((&kv_latent)), "r"(cta_rank * 256), "r"(idx_1[4]), "r"(idx_1[5]), "r"(idx_1[6]), "r"(idx_1[7]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 2048 + 16384), "l"((&kv_latent)), "r"(cta_rank * 256 + 64), "r"(idx_1[4]), "r"(idx_1[5]), "r"(idx_1[6]), "r"(idx_1[7]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 2048 + 32768), "l"((&kv_latent)), "r"(cta_rank * 256 + 128), "r"(idx_1[4]), "r"(idx_1[5]), "r"(idx_1[6]), "r"(idx_1[7]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 2048 + 49152), "l"((&kv_latent)), "r"(cta_rank * 256 + 192), "r"(idx_1[4]), "r"(idx_1[5]), "r"(idx_1[6]), "r"(idx_1[7]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 4096), "l"((&kv_latent)), "r"(cta_rank * 256), "r"(idx_1[8]), "r"(idx_1[9]), "r"(idx_1[10]), "r"(idx_1[11]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 4096 + 16384), "l"((&kv_latent)), "r"(cta_rank * 256 + 64), "r"(idx_1[8]), "r"(idx_1[9]), "r"(idx_1[10]), "r"(idx_1[11]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 4096 + 32768), "l"((&kv_latent)), "r"(cta_rank * 256 + 128), "r"(idx_1[8]), "r"(idx_1[9]), "r"(idx_1[10]), "r"(idx_1[11]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 4096 + 49152), "l"((&kv_latent)), "r"(cta_rank * 256 + 192), "r"(idx_1[8]), "r"(idx_1[9]), "r"(idx_1[10]), "r"(idx_1[11]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 6144), "l"((&kv_latent)), "r"(cta_rank * 256), "r"(idx_1[12]), "r"(idx_1[13]), "r"(idx_1[14]), "r"(idx_1[15]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 6144 + 16384), "l"((&kv_latent)), "r"(cta_rank * 256 + 64), "r"(idx_1[12]), "r"(idx_1[13]), "r"(idx_1[14]), "r"(idx_1[15]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 6144 + 32768), "l"((&kv_latent)), "r"(cta_rank * 256 + 128), "r"(idx_1[12]), "r"(idx_1[13]), "r"(idx_1[14]), "r"(idx_1[15]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(v_buf_addr + 8192 + (unsigned int)(warp_0_1 * 512) + 6144 + 49152), "l"((&kv_latent)), "r"(cta_rank * 256 + 192), "r"(idx_1[12]), "r"(idx_1[13]), "r"(idx_1[14]), "r"(idx_1[15]), "r"(((v_b_addr + ((int)((unsigned int)(base_2 + n_2) % 2)) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                    base_2 += tiles_2;
                }
                int _stage_pv_a_1 = (int)((unsigned int)(base_2 + 1) % 2);
                int _parity_pv_a_1 = (int)((unsigned int)(base_2 + 1) / 2 % 2) ^ 1;
                mbarrier_wait(pv_a_addr + (_stage_pv_a_1) * 8, _parity_pv_a_1);
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int cid_3 = blockIdx.x / 2;
            int nclu_3 = (int)gridDim.x / 2;
            int ntok_3 = (num_queries - cid_3 + nclu_3 - 1) / nclu_3;
            int base_3 = 0;
            int tokens = 0;
            if (cta_rank == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int it_3 = 0; it_3 < ntok_3; it_3++) {
                        int _stage_len_ready_3 = (int)((unsigned int)it_3 % 2);
                        int _parity_len_ready_3 = (int)((unsigned int)it_3 / 2 % 2);
                        mbarrier_wait(len_ready_addr + (_stage_len_ready_3) * 8, _parity_len_ready_3);
                        int _max_0_3 = ((len_cells[it_3 % 2]) > (0) ? (len_cells[it_3 % 2]) : (0));
                        int active_3 = ((_max_0_3) < (topk) ? (_max_0_3) : (topk));
                        mbarrier_arrive(len_free_addr + ((int)((unsigned int)it_3 % 2)) * 8);
                        int tiles_3 = (((active_3 + 127) / 128) > (1) ? ((active_3 + 127) / 128) : (1));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((q_full_bar_addr) & 0xFEFFFFFF), "r"((uint32_t)(147456)) : "memory");
                        int _parity_q_full_bar = (int)((unsigned int)it_3 % 2);
                        mbarrier_wait(q_full_bar_addr, _parity_q_full_bar);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(q_full_addr)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(q_full_addr)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 0 + 0)), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(q_full_addr + 16)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(q_full_addr + 16)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 0 + 4)), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(q_full_addr + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(q_full_addr + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 0 + 8)), "l"(_tcgen05_cp_desc_3)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(q_full_addr + 48)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(q_full_addr + 48)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 0 + 12)), "l"(_tcgen05_cp_desc_4)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(q_full_addr + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(q_full_addr + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 0 + 16)), "l"(_tcgen05_cp_desc_5)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(q_full_addr + 80)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(q_full_addr + 80)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 0 + 20)), "l"(_tcgen05_cp_desc_6)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(q_full_addr + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(q_full_addr + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 0 + 24)), "l"(_tcgen05_cp_desc_7)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(q_full_addr + 112)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(q_full_addr + 112)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 0 + 28)), "l"(_tcgen05_cp_desc_8)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(q_full_addr + 8192)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(q_full_addr + 8192)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 32 + 0)), "l"(_tcgen05_cp_desc_9)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(q_full_addr + 8192 + 16)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(q_full_addr + 8192 + 16)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 32 + 4)), "l"(_tcgen05_cp_desc_10)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(q_full_addr + 8192 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(q_full_addr + 8192 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 32 + 8)), "l"(_tcgen05_cp_desc_11)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(q_full_addr + 8192 + 48)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(q_full_addr + 8192 + 48)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 32 + 12)), "l"(_tcgen05_cp_desc_12)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(q_full_addr + 8192 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(q_full_addr + 8192 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 32 + 16)), "l"(_tcgen05_cp_desc_13)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(q_full_addr + 8192 + 80)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(q_full_addr + 8192 + 80)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 32 + 20)), "l"(_tcgen05_cp_desc_14)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(q_full_addr + 8192 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(q_full_addr + 8192 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 32 + 24)), "l"(_tcgen05_cp_desc_15)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(q_full_addr + 8192 + 112)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(q_full_addr + 8192 + 112)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 32 + 28)), "l"(_tcgen05_cp_desc_16)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(q_full_addr + 16384)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(q_full_addr + 16384)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 64 + 0)), "l"(_tcgen05_cp_desc_17)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(q_full_addr + 16384 + 16)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(q_full_addr + 16384 + 16)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 64 + 4)), "l"(_tcgen05_cp_desc_18)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(q_full_addr + 16384 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(q_full_addr + 16384 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 64 + 8)), "l"(_tcgen05_cp_desc_19)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_20 = ((((uint64_t)(q_full_addr + 16384 + 48)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_20 = ((((uint64_t)(q_full_addr + 16384 + 48)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 64 + 12)), "l"(_tcgen05_cp_desc_20)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_21 = ((((uint64_t)(q_full_addr + 16384 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_21 = ((((uint64_t)(q_full_addr + 16384 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 64 + 16)), "l"(_tcgen05_cp_desc_21)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_22 = ((((uint64_t)(q_full_addr + 16384 + 80)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_22 = ((((uint64_t)(q_full_addr + 16384 + 80)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 64 + 20)), "l"(_tcgen05_cp_desc_22)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_23 = ((((uint64_t)(q_full_addr + 16384 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_23 = ((((uint64_t)(q_full_addr + 16384 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 64 + 24)), "l"(_tcgen05_cp_desc_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_24 = ((((uint64_t)(q_full_addr + 16384 + 112)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_24 = ((((uint64_t)(q_full_addr + 16384 + 112)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 64 + 28)), "l"(_tcgen05_cp_desc_24)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_25 = ((((uint64_t)(q_full_addr + 24576)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_25 = ((((uint64_t)(q_full_addr + 24576)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 96 + 0)), "l"(_tcgen05_cp_desc_25)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_26 = ((((uint64_t)(q_full_addr + 24576 + 16)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_26 = ((((uint64_t)(q_full_addr + 24576 + 16)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 96 + 4)), "l"(_tcgen05_cp_desc_26)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_27 = ((((uint64_t)(q_full_addr + 24576 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_27 = ((((uint64_t)(q_full_addr + 24576 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 96 + 8)), "l"(_tcgen05_cp_desc_27)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_28 = ((((uint64_t)(q_full_addr + 24576 + 48)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_28 = ((((uint64_t)(q_full_addr + 24576 + 48)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 96 + 12)), "l"(_tcgen05_cp_desc_28)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_29 = ((((uint64_t)(q_full_addr + 24576 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_29 = ((((uint64_t)(q_full_addr + 24576 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 96 + 16)), "l"(_tcgen05_cp_desc_29)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_30 = ((((uint64_t)(q_full_addr + 24576 + 80)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_30 = ((((uint64_t)(q_full_addr + 24576 + 80)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 96 + 20)), "l"(_tcgen05_cp_desc_30)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_31 = ((((uint64_t)(q_full_addr + 24576 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_31 = ((((uint64_t)(q_full_addr + 24576 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 96 + 24)), "l"(_tcgen05_cp_desc_31)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
#if __CUDA_ARCH__ == 1070
                            uint64_t _tcgen05_cp_desc_32 = ((((uint64_t)(q_full_addr + 24576 + 112)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                            uint64_t _tcgen05_cp_desc_32 = ((((uint64_t)(q_full_addr + 24576 + 112)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                            asm volatile(
                                "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;"
                                :: "r"((uint32_t)(tmem_tmem + 384 + 96 + 28)), "l"(_tcgen05_cp_desc_32)
                                : "memory");
                        }
                        tcgen05_commit_cg2_multicast(q_copied_addr, (uint16_t)(3));
                        #pragma unroll 1
                        for (int n_3 = 0; n_3 < tiles_3 + 1; n_3++) {
                            if (tiles_3 > n_3) {
                                int _stage_s_free = (int)((unsigned int)(base_3 + n_3) % 2);
                                int _parity_s_free = (int)((unsigned int)(base_3 + n_3) / 2 % 2) ^ 1;
                                mbarrier_wait(s_free_addr + (_stage_s_free) * 8, _parity_s_free);
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((k_a_addr + ((int)((unsigned int)(base_3 + n_3) % 2)) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                                int _stage_k_a = (int)((unsigned int)(base_3 + n_3) % 2);
                                int _parity_k_a = (int)((unsigned int)(base_3 + n_3) / 2 % 2);
                                mbarrier_wait(k_a_addr + (_stage_k_a) * 8, _parity_k_a);
                                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_33 = ((k_slot_addr) >> 4) & 0x7FFF;
#else
                                int _smem_desc_lo_33 = ((k_slot_addr) >> 4) & 0x3FFF;
#endif
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (256 + (base_3 + n_3) % 2 * 64))), "r"(_smem_desc_lo_33), "r"(tmem_tmem + 384), "r"(0) : "memory");
                                tcgen05_commit_cg2_multicast(qk_part_addr + ((int)((unsigned int)(base_3 + n_3) % 2)) * 8, (uint16_t)(3));
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((k_b_addr + ((int)((unsigned int)(base_3 + n_3) % 2)) * 8) & 0xFEFFFFFF), "r"((uint32_t)(81920)) : "memory");
                                int _stage_k_b = (int)((unsigned int)(base_3 + n_3) % 2);
                                int _parity_k_b = (int)((unsigned int)(base_3 + n_3) / 2 % 2);
                                mbarrier_wait(k_b_addr + (_stage_k_b) * 8, _parity_k_b);
                                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_34 = ((q_smem_addr) >> 4) & 0x7FFF;
                                int _smem_desc_lo_35 = ((k_slot_addr + 32768) >> 4) & 0x7FFF;
#else
                                int _smem_desc_lo_34 = ((q_smem_addr) >> 4) & 0x3FFF;
                                int _smem_desc_lo_35 = ((k_slot_addr + 32768) >> 4) & 0x3FFF;
#endif
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_34), "r"(_smem_desc_lo_35), "r"((tmem_tmem + (256 + (base_3 + n_3) % 2 * 64))), "r"(1) : "memory");
                                tcgen05_commit_cg2_multicast(qk_done_addr + ((int)((unsigned int)(base_3 + n_3) % 2)) * 8, (uint16_t)(3));
                            }
                            if (n_3 > 0) {
                                if (n_3 == 1) {
                                    int _parity_o_free = (int)((unsigned int)tokens % 2) ^ 1;
                                    mbarrier_wait(o_free_addr, _parity_o_free);
                                }
                                int _stage_so_ready = (int)((unsigned int)(base_3 + n_3 - 1) % 2);
                                int _parity_so_ready = (int)((unsigned int)(base_3 + n_3 - 1) / 2 % 2);
                                mbarrier_wait(so_ready_addr + (_stage_so_ready) * 8, _parity_so_ready);
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((v_a_addr + ((int)((unsigned int)(base_3 + n_3 - 1) % 2)) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                                int _stage_v_a = (int)((unsigned int)(base_3 + n_3 - 1) % 2);
                                int _parity_v_a = (int)((unsigned int)(base_3 + n_3 - 1) / 2 % 2);
                                mbarrier_wait(v_a_addr + (_stage_v_a) * 8, _parity_v_a);
                                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_36 = ((((prob_addr) >> 4) & 0x7FFF) | 0x400000) + ((base_3 + n_3 - 1) % 2) * 1024;
                                int _smem_desc_lo_37 = (((v_buf_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                                int _smem_desc_lo_36 = ((((prob_addr) >> 4) & 0x3FFF) | 0x400000) + ((base_3 + n_3 - 1) % 2) * 1024;
                                int _smem_desc_lo_37 = (((v_buf_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x00004008;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138478736;\n\t"
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
                    :: "r"(_smem_desc_lo_36), "r"(_smem_desc_lo_37), "r"(tmem_tmem), "r"(((n_3 == 1) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_38 = ((((prob_addr) >> 4) & 0x7FFF) | 0x400000) + ((base_3 + n_3 - 1) % 2) * 1024;
                                int _smem_desc_lo_39 = (((v_buf_addr + 32768) >> 4) & 0x7FFF) | 0x4000000;
#else
                                int _smem_desc_lo_38 = ((((prob_addr) >> 4) & 0x3FFF) | 0x400000) + ((base_3 + n_3 - 1) % 2) * 1024;
                                int _smem_desc_lo_39 = (((v_buf_addr + 32768) >> 4) & 0x3FFF) | 0x4000000;
#endif
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x00004008;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138478736;\n\t"
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
                    :: "r"(_smem_desc_lo_38), "r"(_smem_desc_lo_39), "r"((tmem_tmem + (128))), "r"(((n_3 == 1) ? 0 : 1)) : "memory");
                                tcgen05_commit_cg2_multicast(pv_a_addr + ((int)((unsigned int)(base_3 + n_3 - 1) % 2)) * 8, (uint16_t)(3));
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((v_b_addr + ((int)((unsigned int)(base_3 + n_3 - 1) % 2)) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                                int _stage_v_b = (int)((unsigned int)(base_3 + n_3 - 1) % 2);
                                int _parity_v_b = (int)((unsigned int)(base_3 + n_3 - 1) / 2 % 2);
                                mbarrier_wait(v_b_addr + (_stage_v_b) * 8, _parity_v_b);
                                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_40 = ((((prob_addr) >> 4) & 0x7FFF) | 0x400000) + ((base_3 + n_3 - 1) % 2) * 1024;
                                int _smem_desc_lo_41 = (((v_buf_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                                int _smem_desc_lo_40 = ((((prob_addr) >> 4) & 0x3FFF) | 0x400000) + ((base_3 + n_3 - 1) % 2) * 1024;
                                int _smem_desc_lo_41 = (((v_buf_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x00004008;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138478736;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    :: "r"(_smem_desc_lo_40), "r"(_smem_desc_lo_41), "r"(tmem_tmem), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_42 = ((((prob_addr) >> 4) & 0x7FFF) | 0x400000) + ((base_3 + n_3 - 1) % 2) * 1024;
                                int _smem_desc_lo_43 = (((v_buf_addr + 32768) >> 4) & 0x7FFF) | 0x4000000;
#else
                                int _smem_desc_lo_42 = ((((prob_addr) >> 4) & 0x3FFF) | 0x400000) + ((base_3 + n_3 - 1) % 2) * 1024;
                                int _smem_desc_lo_43 = (((v_buf_addr + 32768) >> 4) & 0x3FFF) | 0x4000000;
#endif
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x00004008;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138478736;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    :: "r"(_smem_desc_lo_42), "r"(_smem_desc_lo_43), "r"((tmem_tmem + (128))), "r"(1) : "memory");
                                tcgen05_commit_cg2_multicast(pv_b_addr + ((int)((unsigned int)(base_3 + n_3 - 1) % 2)) * 8, (uint16_t)(3));
                            }
                        }
                        base_3 += tiles_3;
                        tokens += 1;
                    }
                    int _stage_s_free_1 = (int)((unsigned int)base_3 % 2);
                    int _parity_s_free_1 = (int)((unsigned int)base_3 / 2 % 2) ^ 1;
                    mbarrier_wait(s_free_addr + (_stage_s_free_1) * 8, _parity_s_free_1);
                    int _stage_s_free_2 = (int)((unsigned int)(base_3 + 1) % 2);
                    int _parity_s_free_2 = (int)((unsigned int)(base_3 + 1) / 2 % 2) ^ 1;
                    mbarrier_wait(s_free_addr + (_stage_s_free_2) * 8, _parity_s_free_2);
                    int _parity_o_free_1 = (int)((unsigned int)tokens % 2) ^ 1;
                    mbarrier_wait(o_free_addr, _parity_o_free_1);
                }
            } else {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int it_4 = 0; it_4 < ntok_3; it_4++) {
                        int _stage_len_ready_4 = (int)((unsigned int)it_4 % 2);
                        int _parity_len_ready_4 = (int)((unsigned int)it_4 / 2 % 2);
                        mbarrier_wait(len_ready_addr + (_stage_len_ready_4) * 8, _parity_len_ready_4);
                        int _max_44 = ((len_cells[it_4 % 2]) > (0) ? (len_cells[it_4 % 2]) : (0));
                        int active_4 = ((_max_44) < (topk) ? (_max_44) : (topk));
                        mbarrier_arrive(len_free_addr + ((int)((unsigned int)it_4 % 2)) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: metadata ----
    if (warp == 13) {
        { // metadata_main
            if (lane < 16) {
                int cid_4 = blockIdx.x / 2;
                int nclu_4 = (int)gridDim.x / 2;
                int ntok_4 = (num_queries - cid_4 + nclu_4 - 1) / nclu_4;
                int base_4 = 0;
                int aligned_2 = ((idx_stride | indices_offset) & 7) == 0;
                #pragma unroll 1
                for (int it_5 = 0; it_5 < ntok_4; it_5++) {
                    int query_3 = cid_4 + it_5 * nclu_4;
                    int _stage_len_ready_5 = (int)((unsigned int)it_5 % 2);
                    int _parity_len_ready_5 = (int)((unsigned int)it_5 / 2 % 2);
                    mbarrier_wait(len_ready_addr + (_stage_len_ready_5) * 8, _parity_len_ready_5);
                    int _max_0_4 = ((len_cells[it_5 % 2]) > (0) ? (len_cells[it_5 % 2]) : (0));
                    int active_5 = ((_max_0_4) < (topk) ? (_max_0_4) : (topk));
                    mbarrier_arrive(len_free_addr + ((int)((unsigned int)it_5 % 2)) * 8);
                    int tiles_4 = (((active_5 + 127) / 128) > (1) ? ((active_5 + 127) / 128) : (1));
                    long long row_base_2 = (long long)indices_offset + (long long)query_3 * (long long)idx_stride;
                    #pragma unroll 1
                    for (int n_4 = 0; n_4 < tiles_4; n_4++) {
                        unsigned int mask = 0;
                        int position = n_4 * 128 + lane * 8;
                        if (active_5 >= n_4 * 128 + 128 && aligned_2 != 0) {
                            int _vec_load_4[8];
                            {
                                uint32_t _iv_1_0;
                                uint32_t _iv_1_1;
                                uint32_t _iv_1_2;
                                uint32_t _iv_1_3;
                                uint32_t _iv_1_4;
                                uint32_t _iv_1_5;
                                uint32_t _iv_1_6;
                                uint32_t _iv_1_7;
                                asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(indices + (row_base_2 + (long long)position) + (0))) : "memory");
                                _vec_load_4[0 + 0] = (int32_t)_iv_1_0;
                                _vec_load_4[0 + 1] = (int32_t)_iv_1_1;
                                _vec_load_4[0 + 2] = (int32_t)_iv_1_2;
                                _vec_load_4[0 + 3] = (int32_t)_iv_1_3;
                                _vec_load_4[0 + 4] = (int32_t)_iv_1_4;
                                _vec_load_4[0 + 5] = (int32_t)_iv_1_5;
                                _vec_load_4[0 + 6] = (int32_t)_iv_1_6;
                                _vec_load_4[0 + 7] = (int32_t)_iv_1_7;
                            }
                            if (_vec_load_4[0] >= 0 && _vec_load_4[0] < num_kv) {
                                mask = mask | 1;
                            }
                            if (_vec_load_4[1] >= 0 && _vec_load_4[1] < num_kv) {
                                mask = mask | 2;
                            }
                            if (_vec_load_4[2] >= 0 && _vec_load_4[2] < num_kv) {
                                mask = mask | 4;
                            }
                            if (_vec_load_4[3] >= 0 && _vec_load_4[3] < num_kv) {
                                mask = mask | 8;
                            }
                            if (_vec_load_4[4] >= 0 && _vec_load_4[4] < num_kv) {
                                mask = mask | 16;
                            }
                            if (_vec_load_4[5] >= 0 && _vec_load_4[5] < num_kv) {
                                mask = mask | 32;
                            }
                            if (_vec_load_4[6] >= 0 && _vec_load_4[6] < num_kv) {
                                mask = mask | 64;
                            }
                            if (_vec_load_4[7] >= 0 && _vec_load_4[7] < num_kv) {
                                mask = mask | 128;
                            }
                        } else {
                            int _min_2 = ((position) < (topk - 1) ? (position) : (topk - 1));
                            int value_4_0 = indices[row_base_2 + (long long)_min_2];
                            if (value_4_0 >= 0 && value_4_0 < num_kv && active_5 > position) {
                                mask = mask | 1;
                            }
                            int _min_3 = ((position + 1) < (topk - 1) ? (position + 1) : (topk - 1));
                            int value_4_1 = indices[row_base_2 + (long long)_min_3];
                            if (value_4_1 >= 0 && value_4_1 < num_kv && active_5 > position + 1) {
                                mask = mask | 2;
                            }
                            int _min_4 = ((position + 2) < (topk - 1) ? (position + 2) : (topk - 1));
                            int value_4_2 = indices[row_base_2 + (long long)_min_4];
                            if (value_4_2 >= 0 && value_4_2 < num_kv && active_5 > position + 2) {
                                mask = mask | 4;
                            }
                            int _min_5_2 = ((position + 3) < (topk - 1) ? (position + 3) : (topk - 1));
                            int value_4_3 = indices[row_base_2 + (long long)_min_5_2];
                            if (value_4_3 >= 0 && value_4_3 < num_kv && active_5 > position + 3) {
                                mask = mask | 8;
                            }
                            int _min_6_2 = ((position + 4) < (topk - 1) ? (position + 4) : (topk - 1));
                            int value_4_4 = indices[row_base_2 + (long long)_min_6_2];
                            if (value_4_4 >= 0 && value_4_4 < num_kv && active_5 > position + 4) {
                                mask = mask | 16;
                            }
                            int _min_7_2 = ((position + 5) < (topk - 1) ? (position + 5) : (topk - 1));
                            int value_4_5 = indices[row_base_2 + (long long)_min_7_2];
                            if (value_4_5 >= 0 && value_4_5 < num_kv && active_5 > position + 5) {
                                mask = mask | 32;
                            }
                            int _min_8_2 = ((position + 6) < (topk - 1) ? (position + 6) : (topk - 1));
                            int value_4_6 = indices[row_base_2 + (long long)_min_8_2];
                            if (value_4_6 >= 0 && value_4_6 < num_kv && active_5 > position + 6) {
                                mask = mask | 64;
                            }
                            int _min_9_2 = ((position + 7) < (topk - 1) ? (position + 7) : (topk - 1));
                            int value_4_7 = indices[row_base_2 + (long long)_min_9_2];
                            if (value_4_7 >= 0 && value_4_7 < num_kv && active_5 > position + 7) {
                                mask = mask | 128;
                            }
                        }
                        int _stage_valid_free = (int)((unsigned int)(base_4 + n_4) % 2);
                        int _parity_valid_free = (int)((unsigned int)(base_4 + n_4) / 2 % 2) ^ 1;
                        mbarrier_wait(valid_free_addr + (_stage_valid_free) * 8, _parity_valid_free);
                        validity[(base_4 + n_4) % 2 * 16 + lane] = mask;
                        mbarrier_arrive(valid_ready_addr + ((int)((unsigned int)(base_4 + n_4) % 2)) * 8);
                    }
                    base_4 += tiles_4;
                }
            }
        }
    }
    // ---- Role: scanner ----
    if (warp == 14) {
        { // scanner_main
            int cid_5 = blockIdx.x / 2;
            int nclu_5 = (int)gridDim.x / 2;
            int ntok_5 = (num_queries - cid_5 + nclu_5 - 1) / nclu_5;
            int scan_aligned = ((idx_stride | indices_offset) & 7) == 0;
            #pragma unroll 1
            for (int it_6 = 0; it_6 < ntok_5; it_6++) {
                int query_4 = cid_5 + it_6 * nclu_5;
                int part = topk;
                if (derive_length != 0) {
                    long long row_base_3 = (long long)indices_offset + (long long)query_4 * (long long)idx_stride;
                    int last = 0;
                    int n_full = topk / 256;
                    if (scan_aligned == 0) {
                        n_full = 0;
                    }
                    #pragma unroll 4
                    for (int g = 0; g < n_full; g++) {
                        int spos = g * 256 + lane * 8;
                        int _vec_load_6[8];
                        {
                            uint32_t _iv_0_0;
                            uint32_t _iv_0_1;
                            uint32_t _iv_0_2;
                            uint32_t _iv_0_3;
                            uint32_t _iv_0_4;
                            uint32_t _iv_0_5;
                            uint32_t _iv_0_6;
                            uint32_t _iv_0_7;
                            asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(indices + (row_base_3 + (long long)spos) + (0))) : "memory");
                            _vec_load_6[0 + 0] = (int32_t)_iv_0_0;
                            _vec_load_6[0 + 1] = (int32_t)_iv_0_1;
                            _vec_load_6[0 + 2] = (int32_t)_iv_0_2;
                            _vec_load_6[0 + 3] = (int32_t)_iv_0_3;
                            _vec_load_6[0 + 4] = (int32_t)_iv_0_4;
                            _vec_load_6[0 + 5] = (int32_t)_iv_0_5;
                            _vec_load_6[0 + 6] = (int32_t)_iv_0_6;
                            _vec_load_6[0 + 7] = (int32_t)_iv_0_7;
                        }
                        if (_vec_load_6[0] >= 0 && _vec_load_6[0] < num_kv) {
                            last = spos + 1;
                        }
                        if (_vec_load_6[1] >= 0 && _vec_load_6[1] < num_kv) {
                            last = spos + 1 + 1;
                        }
                        if (_vec_load_6[2] >= 0 && _vec_load_6[2] < num_kv) {
                            last = spos + 2 + 1;
                        }
                        if (_vec_load_6[3] >= 0 && _vec_load_6[3] < num_kv) {
                            last = spos + 3 + 1;
                        }
                        if (_vec_load_6[4] >= 0 && _vec_load_6[4] < num_kv) {
                            last = spos + 4 + 1;
                        }
                        if (_vec_load_6[5] >= 0 && _vec_load_6[5] < num_kv) {
                            last = spos + 5 + 1;
                        }
                        if (_vec_load_6[6] >= 0 && _vec_load_6[6] < num_kv) {
                            last = spos + 6 + 1;
                        }
                        if (_vec_load_6[7] >= 0 && _vec_load_6[7] < num_kv) {
                            last = spos + 7 + 1;
                        }
                    }
                    int tail0 = n_full * 256 + lane;
                    int n_tail = (topk - n_full * 256 - lane + 31) / 32;
                    #pragma unroll 1
                    for (int t = 0; t < n_tail; t++) {
                        int pos = tail0 + t * 32;
                        int value = indices[row_base_3 + (long long)pos];
                        if (value >= 0 && value < num_kv) {
                            last = pos + 1;
                        }
                    }
                    int peer_0 = __shfl_xor_sync(0xFFFFFFFF, last, 16);
                    int _max_1 = ((last) > (peer_0) ? (last) : (peer_0));
                    last = _max_1;
                    int peer_1 = __shfl_xor_sync(0xFFFFFFFF, last, 8);
                    int _max_2 = ((last) > (peer_1) ? (last) : (peer_1));
                    last = _max_2;
                    int peer_2 = __shfl_xor_sync(0xFFFFFFFF, last, 4);
                    int _max_3 = ((last) > (peer_2) ? (last) : (peer_2));
                    last = _max_3;
                    int peer_3 = __shfl_xor_sync(0xFFFFFFFF, last, 2);
                    int _max_4 = ((last) > (peer_3) ? (last) : (peer_3));
                    last = _max_4;
                    int peer_4 = __shfl_xor_sync(0xFFFFFFFF, last, 1);
                    int _max_5 = ((last) > (peer_4) ? (last) : (peer_4));
                    last = _max_5;
                    part = last;
                } else if (has_topk_length != 0) {
                    part = topk_length[query_4];
                }
                int _stage_len_free = (int)((unsigned int)it_6 % 2);
                int _parity_len_free = (int)((unsigned int)it_6 / 2 % 2) ^ 1;
                mbarrier_wait(len_free_addr + (_stage_len_free) * 8, _parity_len_free);
                if (lane == 0) {
                    len_cells[it_6 % 2] = part;
                    if (derive_length != 0 && cta_rank == 0) {
                        topk_length[(long long)query_4] = part;
                    }
                    mbarrier_arrive(len_ready_addr + ((int)((unsigned int)it_6 % 2)) * 8);
                }
            }
        }
    }
    // ---- Role: inactive ----
    if (warp == 15) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
