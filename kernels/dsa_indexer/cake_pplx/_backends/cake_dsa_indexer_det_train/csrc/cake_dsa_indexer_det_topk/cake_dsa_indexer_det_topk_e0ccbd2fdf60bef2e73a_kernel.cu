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



__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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



__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_indexer_det_topk_e0ccbd2fdf60bef2e73a(const __grid_constant__ CUtensorMap Q, __nv_bfloat16* __restrict__ Kraw, float* __restrict__ W, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, long long* __restrict__ q_offsets, int* __restrict__ CandBlocks, int* __restrict__ Indices, float* __restrict__ Scores, int* __restrict__ Lists, long long* __restrict__ Cand, unsigned int* __restrict__ TicketWord, int num_segments, int top_k, int has_offsets, int cand_width, int cand_cap, int check_period, int grid_ctas, float softmax_scale, const __grid_constant__ CUtensorMap K8, long long Kraw_stride_m2)
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
    #define plan_tick_addr (mbar_base + 0)
    #define plan_full_addr (mbar_base + 16)
    #define plan_empty_addr (mbar_base + 32)
    #define q_full_addr (mbar_base + 48)
    #define q_empty_addr (mbar_base + 64)
    #define k_full_addr (mbar_base + 80)
    #define k_empty_addr (mbar_base + 112)
    #define t_full_addr (mbar_base + 144)
    #define t_empty_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_k_addr = smem + 66560;
    int* s_hist = reinterpret_cast<int*>(smem_raw + 197632);
    const int s_hist_addr = smem + 197632;
    float* s_w = reinterpret_cast<float*>(smem_raw + 201728);
    const int s_w_addr = smem + 201728;
    int* s_hdr = reinterpret_cast<int*>(smem_raw + 202752);
    const int s_hdr_addr = smem + 202752;
    int* s_cnt = reinterpret_cast<int*>(smem_raw + 202880);
    const int s_cnt_addr = smem + 202880;
    unsigned long long* s_tau = reinterpret_cast<unsigned long long*>(smem_raw + 202912);
    const int s_tau_addr = smem + 202912;
    int* s_pub = reinterpret_cast<int*>(smem_raw + 202944);
    const int s_pub_addr = smem + 202944;

    // Kernel setup ops
    if (warp == 10) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K8))) : "memory"); }
    if (warp == 9) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }

    // Mbarrier init (26 barriers at smem_raw[0..208))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'plan_pipe' ---
            // plan_tick: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // plan_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // plan_empty: 2 barriers, init_count=9
            mbarrier_init(smem + 32, 9);
            mbarrier_init(smem + 40, 9);
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 4 barriers, init_count=6
            mbarrier_init(smem + 80, 6);
            mbarrier_init(smem + 88, 6);
            mbarrier_init(smem + 96, 6);
            mbarrier_init(smem + 104, 6);
            // k_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 't_pipe' ---
            // t_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // t_empty: 4 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            mbarrier_init(smem + 192, 256);
            mbarrier_init(smem + 200, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 8) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;

    // Kernel post-init ops
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }
    long long cta = (long long)bid;

    // ---- Role: math ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // math_main
            int lane_c = lane;
            int wg = warp / 4;
            int quarter = warp % 4;
            int pr = quarter / 2;
            int pt = (quarter & 1) * 32 + lane_c;
            int rowid = quarter * 32 + lane_c;
            unsigned int lane_base = (unsigned int)(quarter * 32) << 16;
            unsigned int lt_mask = (1 << (unsigned int)lane_c) - 1;
            int myq = 2 * wg + pr;
            int hbase = myq * 256;
            unsigned int pc = 0;
            unsigned int ts = 0;
            int part = 0;
            #pragma unroll 1
            for (int s0 = 0; s0 < num_segments; s0 += 32) {
                int s_c = s0 + lane;
                if (s_c < num_segments) {
                    part += (cu_seqlens_q[s_c + 1] - cu_seqlens_q[s_c] + 3) / 4;
                }
            }
            int n_total;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(n_total) : "r"(part));
            float w[64];
            int nn[2];
            int mm[2];
            unsigned long long tau[2];
            long long cbq[2];
            if (pt == 0) {
                s_cnt[myq] = 0;
                s_tau[myq] = 0ull;
            }
            asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
            unsigned int _phase_plan_tick = 0;
            unsigned int _phase_t_full = 0;
            #pragma unroll 1
            for (int it_c = 0; it_c < n_total + 1; it_c++) {
                mbarrier_wait(plan_tick_addr + (pc) * 8, _phase_plan_tick);
                int hb_c = (int)pc * 16;
                if (s_hdr[hb_c] == 0) {
                    break;
                }
                mbarrier_wait(plan_full_addr + (pc) * 8, _phase_plan_tick);
                int row0_c = s_hdr[hb_c + 1];
                int nq_c = s_hdr[hb_c + 2];
                int steps_c = s_hdr[hb_c + 12];
                long long lb_c = (cta * 2ll + (long long)pc) * 8192ll;
                nn[0] = s_hdr[hb_c + 4 + 2 * wg];
                mm[0] = s_hdr[hb_c + 8 + 2 * wg];
                tau[0] = 0ull;
                cbq[0] = (cta * 4ll + (long long)(2 * wg)) * (long long)cand_cap;
                nn[1] = s_hdr[hb_c + 4 + 2 * wg + 1];
                mm[1] = s_hdr[hb_c + 8 + 2 * wg + 1];
                tau[1] = 0ull;
                cbq[1] = (cta * 4ll + (long long)(2 * wg + 1)) * (long long)cand_cap;
                w[0] = s_w[(int)pc * 128 + wg * 64];
                w[1] = s_w[(int)pc * 128 + wg * 64 + 1];
                w[2] = s_w[(int)pc * 128 + wg * 64 + 2];
                w[3] = s_w[(int)pc * 128 + wg * 64 + 3];
                w[4] = s_w[(int)pc * 128 + wg * 64 + 4];
                w[5] = s_w[(int)pc * 128 + wg * 64 + 5];
                w[6] = s_w[(int)pc * 128 + wg * 64 + 6];
                w[7] = s_w[(int)pc * 128 + wg * 64 + 7];
                w[8] = s_w[(int)pc * 128 + wg * 64 + 8];
                w[9] = s_w[(int)pc * 128 + wg * 64 + 9];
                w[10] = s_w[(int)pc * 128 + wg * 64 + 10];
                w[11] = s_w[(int)pc * 128 + wg * 64 + 11];
                w[12] = s_w[(int)pc * 128 + wg * 64 + 12];
                w[13] = s_w[(int)pc * 128 + wg * 64 + 13];
                w[14] = s_w[(int)pc * 128 + wg * 64 + 14];
                w[15] = s_w[(int)pc * 128 + wg * 64 + 15];
                w[16] = s_w[(int)pc * 128 + wg * 64 + 16];
                w[17] = s_w[(int)pc * 128 + wg * 64 + 17];
                w[18] = s_w[(int)pc * 128 + wg * 64 + 18];
                w[19] = s_w[(int)pc * 128 + wg * 64 + 19];
                w[20] = s_w[(int)pc * 128 + wg * 64 + 20];
                w[21] = s_w[(int)pc * 128 + wg * 64 + 21];
                w[22] = s_w[(int)pc * 128 + wg * 64 + 22];
                w[23] = s_w[(int)pc * 128 + wg * 64 + 23];
                w[24] = s_w[(int)pc * 128 + wg * 64 + 24];
                w[25] = s_w[(int)pc * 128 + wg * 64 + 25];
                w[26] = s_w[(int)pc * 128 + wg * 64 + 26];
                w[27] = s_w[(int)pc * 128 + wg * 64 + 27];
                w[28] = s_w[(int)pc * 128 + wg * 64 + 28];
                w[29] = s_w[(int)pc * 128 + wg * 64 + 29];
                w[30] = s_w[(int)pc * 128 + wg * 64 + 30];
                w[31] = s_w[(int)pc * 128 + wg * 64 + 31];
                w[32] = s_w[(int)pc * 128 + wg * 64 + 32];
                w[33] = s_w[(int)pc * 128 + wg * 64 + 33];
                w[34] = s_w[(int)pc * 128 + wg * 64 + 34];
                w[35] = s_w[(int)pc * 128 + wg * 64 + 35];
                w[36] = s_w[(int)pc * 128 + wg * 64 + 36];
                w[37] = s_w[(int)pc * 128 + wg * 64 + 37];
                w[38] = s_w[(int)pc * 128 + wg * 64 + 38];
                w[39] = s_w[(int)pc * 128 + wg * 64 + 39];
                w[40] = s_w[(int)pc * 128 + wg * 64 + 40];
                w[41] = s_w[(int)pc * 128 + wg * 64 + 41];
                w[42] = s_w[(int)pc * 128 + wg * 64 + 42];
                w[43] = s_w[(int)pc * 128 + wg * 64 + 43];
                w[44] = s_w[(int)pc * 128 + wg * 64 + 44];
                w[45] = s_w[(int)pc * 128 + wg * 64 + 45];
                w[46] = s_w[(int)pc * 128 + wg * 64 + 46];
                w[47] = s_w[(int)pc * 128 + wg * 64 + 47];
                w[48] = s_w[(int)pc * 128 + wg * 64 + 48];
                w[49] = s_w[(int)pc * 128 + wg * 64 + 49];
                w[50] = s_w[(int)pc * 128 + wg * 64 + 50];
                w[51] = s_w[(int)pc * 128 + wg * 64 + 51];
                w[52] = s_w[(int)pc * 128 + wg * 64 + 52];
                w[53] = s_w[(int)pc * 128 + wg * 64 + 53];
                w[54] = s_w[(int)pc * 128 + wg * 64 + 54];
                w[55] = s_w[(int)pc * 128 + wg * 64 + 55];
                w[56] = s_w[(int)pc * 128 + wg * 64 + 56];
                w[57] = s_w[(int)pc * 128 + wg * 64 + 57];
                w[58] = s_w[(int)pc * 128 + wg * 64 + 58];
                w[59] = s_w[(int)pc * 128 + wg * 64 + 59];
                w[60] = s_w[(int)pc * 128 + wg * 64 + 60];
                w[61] = s_w[(int)pc * 128 + wg * 64 + 61];
                w[62] = s_w[(int)pc * 128 + wg * 64 + 62];
                w[63] = s_w[(int)pc * 128 + wg * 64 + 63];
                int blk_next[2];
                blk_next[0] = -1;
                if (rowid >> 3 < mm[0]) {
                    blk_next[0] = Lists[lb_c + (long long)(2 * wg * 2048 + (rowid >> 3))];
                }
                blk_next[1] = -1;
                if (rowid >> 3 < mm[1]) {
                    blk_next[1] = Lists[lb_c + (long long)((2 * wg + 1) * 2048 + (rowid >> 3))];
                }
                #pragma unroll 1
                for (int i = 0; i < steps_c; i++) {
                    int slot_c = i * 16 + (rowid >> 3);
                    int blk[2];
                    blk[0] = blk_next[0];
                    blk_next[0] = -1;
                    if (slot_c + 16 < mm[0]) {
                        blk_next[0] = Lists[lb_c + (long long)(2 * wg * 2048 + slot_c + 16)];
                    }
                    blk[1] = blk_next[1];
                    blk_next[1] = -1;
                    if (slot_c + 16 < mm[1]) {
                        blk_next[1] = Lists[lb_c + (long long)((2 * wg + 1) * 2048 + slot_c + 16)];
                    }
                    mbarrier_wait(t_full_addr + (ts) * 8, _phase_t_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (i * 16 < mm[0]) {
                        float vals_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(vals_0[0]), "=f"(vals_0[1]), "=f"(vals_0[2]), "=f"(vals_0[3]), "=f"(vals_0[4]), "=f"(vals_0[5]), "=f"(vals_0[6]), "=f"(vals_0[7]), "=f"(vals_0[8]), "=f"(vals_0[9]), "=f"(vals_0[10]), "=f"(vals_0[11]), "=f"(vals_0[12]), "=f"(vals_0[13]), "=f"(vals_0[14]), "=f"(vals_0[15]), "=f"(vals_0[16]), "=f"(vals_0[17]), "=f"(vals_0[18]), "=f"(vals_0[19]), "=f"(vals_0[20]), "=f"(vals_0[21]), "=f"(vals_0[22]), "=f"(vals_0[23]), "=f"(vals_0[24]), "=f"(vals_0[25]), "=f"(vals_0[26]), "=f"(vals_0[27]), "=f"(vals_0[28]), "=f"(vals_0[29]), "=f"(vals_0[30]), "=f"(vals_0[31])
                            : "r"(tmem_acc + ts * 128 + (unsigned int)(2 * wg * 32) + lane_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 sum0 = make_float2(0.0f, 0.0f);
                        float2 sum1 = make_float2(0.0f, 0.0f);
                        #pragma unroll
                        for (int j = 0; j < 32; j += 4) {
                            float2 doubled0 = add_f32x2_noftz(make_float2(vals_0[j], vals_0[j + 1]), make_float2(fabsf(vals_0[j]), fabsf(vals_0[j + 1])));
                            sum0 = fma_f32x2_rn_noftz(doubled0, make_float2(w[j], w[j + 1]), sum0);
                            float2 doubled1 = add_f32x2_noftz(make_float2(vals_0[j + 2], vals_0[j + 3]), make_float2(fabsf(vals_0[j + 2]), fabsf(vals_0[j + 3])));
                            sum1 = fma_f32x2_rn_noftz(doubled1, make_float2(w[j + 2], w[j + 3]), sum1);
                        }
                        float2 total = add_f32x2_noftz(sum0, sum1);
                        float relu_wsum = (total.x + total.y) * 0.5f;
                        unsigned int bits_c_0 = __as_u32(relu_wsum);
                        int emit_0 = 0;
                        int kid_0 = 8 * blk[0] + (rowid & 7);
                        unsigned long long ent_0 = (unsigned long long)bits_c_0 << 32ull | (unsigned long long)(unsigned int)kid_0;
                        if (blk[0] >= 0 && kid_0 < nn[0]) {
                            unsigned int bits_e = (unsigned int)(ent_0 >> 32ull);
                            unsigned int mag_e = bits_e & 2147483647;
                            unsigned int m_e = ((mag_e != 0) ? bits_e : 0);
                            unsigned int k32_e = (((m_e & 2147483648u) != 0) ? ~m_e : m_e | 2147483648u);
                            if (mag_e > 2139095040) {
                                k32_e = 0;
                            }
                            unsigned long long key_e = (unsigned long long)k32_e << 32ull | ent_0 & 4294967295ull;
                            if (key_e >= tau[0]) {
                                emit_0 = 1;
                            }
                        }
                        unsigned int vote_0 = __ballot_sync(0xFFFFFFFF, emit_0 != 0);
                        if (vote_0 != 0) {
                            int wbase_0 = 0;
                            if (lane_c == 0) {
                                int _atomic_fetch_add_0 = atomicAdd(&s_cnt[2 * wg], (int)(unsigned int)__popc(vote_0));
                                wbase_0 = _atomic_fetch_add_0;
                            }
                            int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, wbase_0, 0);
                            wbase_0 = _shfl_sync_1;
                            if (emit_0 != 0) {
                                int pos_c_0 = wbase_0 + (int)(unsigned int)__popc(vote_0 & lt_mask);
                                if (pos_c_0 < cand_cap) {
                                    Cand[cbq[0] + (long long)pos_c_0] = (long long)ent_0;
                                }
                            }
                        }
                    }
                    if (i * 16 < mm[1]) {
                        float vals_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(vals_1[0]), "=f"(vals_1[1]), "=f"(vals_1[2]), "=f"(vals_1[3]), "=f"(vals_1[4]), "=f"(vals_1[5]), "=f"(vals_1[6]), "=f"(vals_1[7]), "=f"(vals_1[8]), "=f"(vals_1[9]), "=f"(vals_1[10]), "=f"(vals_1[11]), "=f"(vals_1[12]), "=f"(vals_1[13]), "=f"(vals_1[14]), "=f"(vals_1[15]), "=f"(vals_1[16]), "=f"(vals_1[17]), "=f"(vals_1[18]), "=f"(vals_1[19]), "=f"(vals_1[20]), "=f"(vals_1[21]), "=f"(vals_1[22]), "=f"(vals_1[23]), "=f"(vals_1[24]), "=f"(vals_1[25]), "=f"(vals_1[26]), "=f"(vals_1[27]), "=f"(vals_1[28]), "=f"(vals_1[29]), "=f"(vals_1[30]), "=f"(vals_1[31])
                            : "r"(tmem_acc + ts * 128 + (unsigned int)((2 * wg + 1) * 32) + lane_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 sum0_1 = make_float2(0.0f, 0.0f);
                        float2 sum1_1 = make_float2(0.0f, 0.0f);
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 32; j_1 += 4) {
                            float2 doubled0_1 = add_f32x2_noftz(make_float2(vals_1[j_1], vals_1[j_1 + 1]), make_float2(fabsf(vals_1[j_1]), fabsf(vals_1[j_1 + 1])));
                            sum0_1 = fma_f32x2_rn_noftz(doubled0_1, make_float2(w[32 + j_1], w[32 + j_1 + 1]), sum0_1);
                            float2 doubled1_1 = add_f32x2_noftz(make_float2(vals_1[j_1 + 2], vals_1[j_1 + 3]), make_float2(fabsf(vals_1[j_1 + 2]), fabsf(vals_1[j_1 + 3])));
                            sum1_1 = fma_f32x2_rn_noftz(doubled1_1, make_float2(w[32 + j_1 + 2], w[32 + j_1 + 3]), sum1_1);
                        }
                        float2 total_1 = add_f32x2_noftz(sum0_1, sum1_1);
                        float relu_wsum_1 = (total_1.x + total_1.y) * 0.5f;
                        unsigned int bits_c_1 = __as_u32(relu_wsum_1);
                        int emit_1 = 0;
                        int kid_1 = 8 * blk[1] + (rowid & 7);
                        unsigned long long ent_1 = (unsigned long long)bits_c_1 << 32ull | (unsigned long long)(unsigned int)kid_1;
                        if (blk[1] >= 0 && kid_1 < nn[1]) {
                            unsigned int bits_e_1 = (unsigned int)(ent_1 >> 32ull);
                            unsigned int mag_e_1 = bits_e_1 & 2147483647;
                            unsigned int m_e_1 = ((mag_e_1 != 0) ? bits_e_1 : 0);
                            unsigned int k32_e_1 = (((m_e_1 & 2147483648u) != 0) ? ~m_e_1 : m_e_1 | 2147483648u);
                            if (mag_e_1 > 2139095040) {
                                k32_e_1 = 0;
                            }
                            unsigned long long key_e_1 = (unsigned long long)k32_e_1 << 32ull | ent_1 & 4294967295ull;
                            if (key_e_1 >= tau[1]) {
                                emit_1 = 1;
                            }
                        }
                        unsigned int vote_1 = __ballot_sync(0xFFFFFFFF, emit_1 != 0);
                        if (vote_1 != 0) {
                            int wbase_1 = 0;
                            if (lane_c == 0) {
                                int _atomic_fetch_add_2 = atomicAdd(&s_cnt[2 * wg + 1], (int)(unsigned int)__popc(vote_1));
                                wbase_1 = _atomic_fetch_add_2;
                            }
                            int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, wbase_1, 0);
                            wbase_1 = _shfl_sync_3;
                            if (emit_1 != 0) {
                                int pos_c_1 = wbase_1 + (int)(unsigned int)__popc(vote_1 & lt_mask);
                                if (pos_c_1 < cand_cap) {
                                    Cand[cbq[1] + (long long)pos_c_1] = (long long)ent_1;
                                }
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(t_empty_addr + (ts) * 8);
                    ts += 1;
                    if (ts == 4) { ts = 0; _phase_t_full ^= 1; }
                    if ((i + 1) % check_period == 0 && i + 1 < steps_c) {
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                        if (pt == 0) {
                            int _min_4 = ((s_cnt[myq]) < (cand_cap) ? (s_cnt[myq]) : (cand_cap));
                            s_pub[myq] = _min_4;
                        }
                        asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                        int my_m = s_hdr[hb_c + 8 + myq];
                        int rem_items = (my_m + 16 - 1) / 16 - (i + 1);
                        int cnt_c = s_pub[myq];
                        int _min_5 = ((check_period) < (rem_items) ? (check_period) : (rem_items));
                        if (rem_items > 0 && cnt_c + 128 * _min_5 > cand_cap) {
                            int keep_cap = ((top_k + 64) < (cand_cap - 128 * check_period) ? (top_k + 64) : (cand_cap - 128 * check_period));
                            int k_rem = top_k;
                            int kept_above = 0;
                            unsigned long long prefix = 0ull;
                            unsigned long long edge = 0ull;
                            int kept = cnt_c;
                            #pragma unroll 1
                            for (int p = 0; p < 8; p++) {
                                unsigned long long shift = (unsigned long long)(56 - 8 * p);
                                unsigned long long hi_shift = ((p > 0) ? shift + 8ull : 56ull);
                                s_hist[hbase + pt * 4] = 0;
                                s_hist[hbase + pt * 4 + 1] = 0;
                                s_hist[hbase + pt * 4 + 2] = 0;
                                s_hist[hbase + pt * 4 + 3] = 0;
                                asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                                #pragma unroll 1
                                for (int c0 = 0; c0 < cnt_c; c0 += 512) {
                                    unsigned long long keys_h[8];
                                    int e_0 = c0 + pt;
                                    keys_h[0] = 0ull;
                                    if (e_0 < cnt_c) {
                                        unsigned int bits_e_2 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_0] >> 32ull);
                                        unsigned int mag_e_2 = bits_e_2 & 2147483647;
                                        unsigned int m_e_2 = ((mag_e_2 != 0) ? bits_e_2 : 0);
                                        unsigned int k32_e_2 = (((m_e_2 & 2147483648u) != 0) ? ~m_e_2 : m_e_2 | 2147483648u);
                                        if (mag_e_2 > 2139095040) {
                                            k32_e_2 = 0;
                                        }
                                        unsigned long long key_e_2 = (unsigned long long)k32_e_2 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_0] & 4294967295ull;
                                        keys_h[0] = key_e_2;
                                    }
                                    int e_1 = c0 + 64 + pt;
                                    keys_h[1] = 0ull;
                                    if (e_1 < cnt_c) {
                                        unsigned int bits_e_3 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_1] >> 32ull);
                                        unsigned int mag_e_3 = bits_e_3 & 2147483647;
                                        unsigned int m_e_3 = ((mag_e_3 != 0) ? bits_e_3 : 0);
                                        unsigned int k32_e_3 = (((m_e_3 & 2147483648u) != 0) ? ~m_e_3 : m_e_3 | 2147483648u);
                                        if (mag_e_3 > 2139095040) {
                                            k32_e_3 = 0;
                                        }
                                        unsigned long long key_e_3 = (unsigned long long)k32_e_3 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_1] & 4294967295ull;
                                        keys_h[1] = key_e_3;
                                    }
                                    int e_2 = c0 + 128 + pt;
                                    keys_h[2] = 0ull;
                                    if (e_2 < cnt_c) {
                                        unsigned int bits_e_4 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_2] >> 32ull);
                                        unsigned int mag_e_4 = bits_e_4 & 2147483647;
                                        unsigned int m_e_4 = ((mag_e_4 != 0) ? bits_e_4 : 0);
                                        unsigned int k32_e_4 = (((m_e_4 & 2147483648u) != 0) ? ~m_e_4 : m_e_4 | 2147483648u);
                                        if (mag_e_4 > 2139095040) {
                                            k32_e_4 = 0;
                                        }
                                        unsigned long long key_e_4 = (unsigned long long)k32_e_4 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_2] & 4294967295ull;
                                        keys_h[2] = key_e_4;
                                    }
                                    int e_3 = c0 + 192 + pt;
                                    keys_h[3] = 0ull;
                                    if (e_3 < cnt_c) {
                                        unsigned int bits_e_5 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_3] >> 32ull);
                                        unsigned int mag_e_5 = bits_e_5 & 2147483647;
                                        unsigned int m_e_5 = ((mag_e_5 != 0) ? bits_e_5 : 0);
                                        unsigned int k32_e_5 = (((m_e_5 & 2147483648u) != 0) ? ~m_e_5 : m_e_5 | 2147483648u);
                                        if (mag_e_5 > 2139095040) {
                                            k32_e_5 = 0;
                                        }
                                        unsigned long long key_e_5 = (unsigned long long)k32_e_5 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_3] & 4294967295ull;
                                        keys_h[3] = key_e_5;
                                    }
                                    int e_4 = c0 + 256 + pt;
                                    keys_h[4] = 0ull;
                                    if (e_4 < cnt_c) {
                                        unsigned int bits_e_6 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_4] >> 32ull);
                                        unsigned int mag_e_6 = bits_e_6 & 2147483647;
                                        unsigned int m_e_6 = ((mag_e_6 != 0) ? bits_e_6 : 0);
                                        unsigned int k32_e_6 = (((m_e_6 & 2147483648u) != 0) ? ~m_e_6 : m_e_6 | 2147483648u);
                                        if (mag_e_6 > 2139095040) {
                                            k32_e_6 = 0;
                                        }
                                        unsigned long long key_e_6 = (unsigned long long)k32_e_6 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_4] & 4294967295ull;
                                        keys_h[4] = key_e_6;
                                    }
                                    int e_5 = c0 + 320 + pt;
                                    keys_h[5] = 0ull;
                                    if (e_5 < cnt_c) {
                                        unsigned int bits_e_7 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_5] >> 32ull);
                                        unsigned int mag_e_7 = bits_e_7 & 2147483647;
                                        unsigned int m_e_7 = ((mag_e_7 != 0) ? bits_e_7 : 0);
                                        unsigned int k32_e_7 = (((m_e_7 & 2147483648u) != 0) ? ~m_e_7 : m_e_7 | 2147483648u);
                                        if (mag_e_7 > 2139095040) {
                                            k32_e_7 = 0;
                                        }
                                        unsigned long long key_e_7 = (unsigned long long)k32_e_7 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_5] & 4294967295ull;
                                        keys_h[5] = key_e_7;
                                    }
                                    int e_6 = c0 + 384 + pt;
                                    keys_h[6] = 0ull;
                                    if (e_6 < cnt_c) {
                                        unsigned int bits_e_8 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_6] >> 32ull);
                                        unsigned int mag_e_8 = bits_e_8 & 2147483647;
                                        unsigned int m_e_8 = ((mag_e_8 != 0) ? bits_e_8 : 0);
                                        unsigned int k32_e_8 = (((m_e_8 & 2147483648u) != 0) ? ~m_e_8 : m_e_8 | 2147483648u);
                                        if (mag_e_8 > 2139095040) {
                                            k32_e_8 = 0;
                                        }
                                        unsigned long long key_e_8 = (unsigned long long)k32_e_8 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_6] & 4294967295ull;
                                        keys_h[6] = key_e_8;
                                    }
                                    int e_7 = c0 + 448 + pt;
                                    keys_h[7] = 0ull;
                                    if (e_7 < cnt_c) {
                                        unsigned int bits_e_9 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_7] >> 32ull);
                                        unsigned int mag_e_9 = bits_e_9 & 2147483647;
                                        unsigned int m_e_9 = ((mag_e_9 != 0) ? bits_e_9 : 0);
                                        unsigned int k32_e_9 = (((m_e_9 & 2147483648u) != 0) ? ~m_e_9 : m_e_9 | 2147483648u);
                                        if (mag_e_9 > 2139095040) {
                                            k32_e_9 = 0;
                                        }
                                        unsigned long long key_e_9 = (unsigned long long)k32_e_9 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_7] & 4294967295ull;
                                        keys_h[7] = key_e_9;
                                    }
                                    if (c0 + pt < cnt_c) {
                                        if (p == 0 || keys_h[0] >> hi_shift == prefix) {
                                            atomicAdd(&s_hist[hbase + (int)(keys_h[0] >> shift & 255ull)], 1);
                                        }
                                    }
                                    if (c0 + 64 + pt < cnt_c) {
                                        if (p == 0 || keys_h[1] >> hi_shift == prefix) {
                                            atomicAdd(&s_hist[hbase + (int)(keys_h[1] >> shift & 255ull)], 1);
                                        }
                                    }
                                    if (c0 + 128 + pt < cnt_c) {
                                        if (p == 0 || keys_h[2] >> hi_shift == prefix) {
                                            atomicAdd(&s_hist[hbase + (int)(keys_h[2] >> shift & 255ull)], 1);
                                        }
                                    }
                                    if (c0 + 192 + pt < cnt_c) {
                                        if (p == 0 || keys_h[3] >> hi_shift == prefix) {
                                            atomicAdd(&s_hist[hbase + (int)(keys_h[3] >> shift & 255ull)], 1);
                                        }
                                    }
                                    if (c0 + 256 + pt < cnt_c) {
                                        if (p == 0 || keys_h[4] >> hi_shift == prefix) {
                                            atomicAdd(&s_hist[hbase + (int)(keys_h[4] >> shift & 255ull)], 1);
                                        }
                                    }
                                    if (c0 + 320 + pt < cnt_c) {
                                        if (p == 0 || keys_h[5] >> hi_shift == prefix) {
                                            atomicAdd(&s_hist[hbase + (int)(keys_h[5] >> shift & 255ull)], 1);
                                        }
                                    }
                                    if (c0 + 384 + pt < cnt_c) {
                                        if (p == 0 || keys_h[6] >> hi_shift == prefix) {
                                            atomicAdd(&s_hist[hbase + (int)(keys_h[6] >> shift & 255ull)], 1);
                                        }
                                    }
                                    if (c0 + 448 + pt < cnt_c) {
                                        if (p == 0 || keys_h[7] >> hi_shift == prefix) {
                                            atomicAdd(&s_hist[hbase + (int)(keys_h[7] >> shift & 255ull)], 1);
                                        }
                                    }
                                }
                                asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                                int lane_bins[8];
                                int lane_sum = 0;
                                lane_bins[0] = s_hist[hbase + lane_c * 8];
                                lane_sum += lane_bins[0];
                                lane_bins[1] = s_hist[hbase + lane_c * 8 + 1];
                                lane_sum += lane_bins[1];
                                lane_bins[2] = s_hist[hbase + lane_c * 8 + 2];
                                lane_sum += lane_bins[2];
                                lane_bins[3] = s_hist[hbase + lane_c * 8 + 3];
                                lane_sum += lane_bins[3];
                                lane_bins[4] = s_hist[hbase + lane_c * 8 + 4];
                                lane_sum += lane_bins[4];
                                lane_bins[5] = s_hist[hbase + lane_c * 8 + 5];
                                lane_sum += lane_bins[5];
                                lane_bins[6] = s_hist[hbase + lane_c * 8 + 6];
                                lane_sum += lane_bins[6];
                                lane_bins[7] = s_hist[hbase + lane_c * 8 + 7];
                                lane_sum += lane_bins[7];
                                int suffix = lane_sum;
                                int above_part_0 = __shfl_down_sync(0xFFFFFFFF, suffix, 1, 32);
                                if (lane_c + 1 < 32) {
                                    suffix += above_part_0;
                                }
                                int above_part_1 = __shfl_down_sync(0xFFFFFFFF, suffix, 2, 32);
                                if (lane_c + 2 < 32) {
                                    suffix += above_part_1;
                                }
                                int above_part_2 = __shfl_down_sync(0xFFFFFFFF, suffix, 4, 32);
                                if (lane_c + 4 < 32) {
                                    suffix += above_part_2;
                                }
                                int above_part_3 = __shfl_down_sync(0xFFFFFFFF, suffix, 8, 32);
                                if (lane_c + 8 < 32) {
                                    suffix += above_part_3;
                                }
                                int above_part_4 = __shfl_down_sync(0xFFFFFFFF, suffix, 16, 32);
                                if (lane_c + 16 < 32) {
                                    suffix += above_part_4;
                                }
                                int excl = suffix - lane_sum;
                                int is_target = ((excl < k_rem && k_rem <= excl + lane_sum) ? 1 : 0);
                                int d_sel = 0;
                                int above_sel = 0;
                                int count_sel = 0;
                                int found = 0;
                                int cum_above = excl;
                                if (found == 0) {
                                    if (k_rem <= cum_above + lane_bins[7]) {
                                        d_sel = 7;
                                        above_sel = cum_above;
                                        count_sel = lane_bins[7];
                                        found = 1;
                                    }
                                }
                                cum_above += lane_bins[7];
                                if (found == 0) {
                                    if (k_rem <= cum_above + lane_bins[6]) {
                                        d_sel = 6;
                                        above_sel = cum_above;
                                        count_sel = lane_bins[6];
                                        found = 1;
                                    }
                                }
                                cum_above += lane_bins[6];
                                if (found == 0) {
                                    if (k_rem <= cum_above + lane_bins[5]) {
                                        d_sel = 5;
                                        above_sel = cum_above;
                                        count_sel = lane_bins[5];
                                        found = 1;
                                    }
                                }
                                cum_above += lane_bins[5];
                                if (found == 0) {
                                    if (k_rem <= cum_above + lane_bins[4]) {
                                        d_sel = 4;
                                        above_sel = cum_above;
                                        count_sel = lane_bins[4];
                                        found = 1;
                                    }
                                }
                                cum_above += lane_bins[4];
                                if (found == 0) {
                                    if (k_rem <= cum_above + lane_bins[3]) {
                                        d_sel = 3;
                                        above_sel = cum_above;
                                        count_sel = lane_bins[3];
                                        found = 1;
                                    }
                                }
                                cum_above += lane_bins[3];
                                if (found == 0) {
                                    if (k_rem <= cum_above + lane_bins[2]) {
                                        d_sel = 2;
                                        above_sel = cum_above;
                                        count_sel = lane_bins[2];
                                        found = 1;
                                    }
                                }
                                cum_above += lane_bins[2];
                                if (found == 0) {
                                    if (k_rem <= cum_above + lane_bins[1]) {
                                        d_sel = 1;
                                        above_sel = cum_above;
                                        count_sel = lane_bins[1];
                                        found = 1;
                                    }
                                }
                                cum_above += lane_bins[1];
                                if (found == 0) {
                                    if (k_rem <= cum_above + lane_bins[0]) {
                                        d_sel = 0;
                                        above_sel = cum_above;
                                        count_sel = lane_bins[0];
                                        found = 1;
                                    }
                                }
                                cum_above += lane_bins[0];
                                int target_lane;
                                asm volatile("redux.sync.max.s32 %0, %1, 0xFFFFFFFF;" : "=r"(target_lane) : "r"(is_target * lane_c));
                                int digit_sel = __shfl_sync(0xFFFFFFFF, d_sel, target_lane);
                                int above_cnt = __shfl_sync(0xFFFFFFFF, above_sel, target_lane);
                                int bucket_cnt = __shfl_sync(0xFFFFFFFF, count_sel, target_lane);
                                k_rem = k_rem - above_cnt;
                                kept_above += above_cnt;
                                prefix = prefix << 8ull | (unsigned long long)(unsigned int)(target_lane * 8 + digit_sel);
                                edge = prefix << shift;
                                kept = kept_above + bucket_cnt;
                                if (pt == 0) {
                                    s_pub[4 + myq] = kept;
                                }
                                asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                                if (s_pub[4 + myq] <= keep_cap) {
                                    break;
                                }
                            }
                            if (pt == 0) {
                                s_cnt[4 + myq] = 0;
                            }
                            asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                            #pragma unroll 1
                            for (int c0_1 = 0; c0_1 < cnt_c; c0_1 += 256) {
                                unsigned long long ents[4];
                                unsigned int keeps[4];
                                int tot_w = 0;
                                int e_c_0 = c0_1 + pt;
                                ents[0] = 0ull;
                                int kp_0 = 0;
                                if (e_c_0 < cnt_c) {
                                    ents[0] = (unsigned long long)Cand[cbq[pr] + (long long)e_c_0];
                                    unsigned int bits_e_10 = (unsigned int)(ents[0] >> 32ull);
                                    unsigned int mag_e_10 = bits_e_10 & 2147483647;
                                    unsigned int m_e_10 = ((mag_e_10 != 0) ? bits_e_10 : 0);
                                    unsigned int k32_e_10 = (((m_e_10 & 2147483648u) != 0) ? ~m_e_10 : m_e_10 | 2147483648u);
                                    if (mag_e_10 > 2139095040) {
                                        k32_e_10 = 0;
                                    }
                                    unsigned long long key_e_10 = (unsigned long long)k32_e_10 << 32ull | ents[0] & 4294967295ull;
                                    if (key_e_10 >= edge) {
                                        kp_0 = 1;
                                    }
                                }
                                unsigned int _warp_vote_6 = __ballot_sync(0xFFFFFFFF, kp_0 != 0);
                                keeps[0] = _warp_vote_6;
                                tot_w += (int)(unsigned int)__popc(keeps[0]);
                                int e_c_1 = c0_1 + 64 + pt;
                                ents[1] = 0ull;
                                int kp_1 = 0;
                                if (e_c_1 < cnt_c) {
                                    ents[1] = (unsigned long long)Cand[cbq[pr] + (long long)e_c_1];
                                    unsigned int bits_e_11 = (unsigned int)(ents[1] >> 32ull);
                                    unsigned int mag_e_11 = bits_e_11 & 2147483647;
                                    unsigned int m_e_11 = ((mag_e_11 != 0) ? bits_e_11 : 0);
                                    unsigned int k32_e_11 = (((m_e_11 & 2147483648u) != 0) ? ~m_e_11 : m_e_11 | 2147483648u);
                                    if (mag_e_11 > 2139095040) {
                                        k32_e_11 = 0;
                                    }
                                    unsigned long long key_e_11 = (unsigned long long)k32_e_11 << 32ull | ents[1] & 4294967295ull;
                                    if (key_e_11 >= edge) {
                                        kp_1 = 1;
                                    }
                                }
                                unsigned int _warp_vote_7 = __ballot_sync(0xFFFFFFFF, kp_1 != 0);
                                keeps[1] = _warp_vote_7;
                                tot_w += (int)(unsigned int)__popc(keeps[1]);
                                int e_c_2 = c0_1 + 128 + pt;
                                ents[2] = 0ull;
                                int kp_2 = 0;
                                if (e_c_2 < cnt_c) {
                                    ents[2] = (unsigned long long)Cand[cbq[pr] + (long long)e_c_2];
                                    unsigned int bits_e_12 = (unsigned int)(ents[2] >> 32ull);
                                    unsigned int mag_e_12 = bits_e_12 & 2147483647;
                                    unsigned int m_e_12 = ((mag_e_12 != 0) ? bits_e_12 : 0);
                                    unsigned int k32_e_12 = (((m_e_12 & 2147483648u) != 0) ? ~m_e_12 : m_e_12 | 2147483648u);
                                    if (mag_e_12 > 2139095040) {
                                        k32_e_12 = 0;
                                    }
                                    unsigned long long key_e_12 = (unsigned long long)k32_e_12 << 32ull | ents[2] & 4294967295ull;
                                    if (key_e_12 >= edge) {
                                        kp_2 = 1;
                                    }
                                }
                                unsigned int _warp_vote_8 = __ballot_sync(0xFFFFFFFF, kp_2 != 0);
                                keeps[2] = _warp_vote_8;
                                tot_w += (int)(unsigned int)__popc(keeps[2]);
                                int e_c_3 = c0_1 + 192 + pt;
                                ents[3] = 0ull;
                                int kp_3 = 0;
                                if (e_c_3 < cnt_c) {
                                    ents[3] = (unsigned long long)Cand[cbq[pr] + (long long)e_c_3];
                                    unsigned int bits_e_13 = (unsigned int)(ents[3] >> 32ull);
                                    unsigned int mag_e_13 = bits_e_13 & 2147483647;
                                    unsigned int m_e_13 = ((mag_e_13 != 0) ? bits_e_13 : 0);
                                    unsigned int k32_e_13 = (((m_e_13 & 2147483648u) != 0) ? ~m_e_13 : m_e_13 | 2147483648u);
                                    if (mag_e_13 > 2139095040) {
                                        k32_e_13 = 0;
                                    }
                                    unsigned long long key_e_13 = (unsigned long long)k32_e_13 << 32ull | ents[3] & 4294967295ull;
                                    if (key_e_13 >= edge) {
                                        kp_3 = 1;
                                    }
                                }
                                unsigned int _warp_vote_9 = __ballot_sync(0xFFFFFFFF, kp_3 != 0);
                                keeps[3] = _warp_vote_9;
                                tot_w += (int)(unsigned int)__popc(keeps[3]);
                                int obase = 0;
                                if (lane_c == 0) {
                                    int _atomic_fetch_add_10 = atomicAdd(&s_cnt[4 + myq], tot_w);
                                    obase = _atomic_fetch_add_10;
                                }
                                int _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, obase, 0);
                                obase = _shfl_sync_11;
                                asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                                if ((keeps[0] >> (unsigned int)lane_c & 1) != 0) {
                                    Cand[cbq[pr] + (long long)(obase + (int)(unsigned int)__popc(keeps[0] & lt_mask))] = (long long)ents[0];
                                }
                                obase += (int)(unsigned int)__popc(keeps[0]);
                                if ((keeps[1] >> (unsigned int)lane_c & 1) != 0) {
                                    Cand[cbq[pr] + (long long)(obase + (int)(unsigned int)__popc(keeps[1] & lt_mask))] = (long long)ents[1];
                                }
                                obase += (int)(unsigned int)__popc(keeps[1]);
                                if ((keeps[2] >> (unsigned int)lane_c & 1) != 0) {
                                    Cand[cbq[pr] + (long long)(obase + (int)(unsigned int)__popc(keeps[2] & lt_mask))] = (long long)ents[2];
                                }
                                obase += (int)(unsigned int)__popc(keeps[2]);
                                if ((keeps[3] >> (unsigned int)lane_c & 1) != 0) {
                                    Cand[cbq[pr] + (long long)(obase + (int)(unsigned int)__popc(keeps[3] & lt_mask))] = (long long)ents[3];
                                }
                                obase += (int)(unsigned int)__popc(keeps[3]);
                            }
                            asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                            if (pt == 0) {
                                s_cnt[myq] = s_cnt[4 + myq];
                                s_tau[myq] = edge;
                            }
                        }
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                        tau[0] = (unsigned long long)s_tau[2 * wg];
                        tau[1] = (unsigned long long)s_tau[2 * wg + 1];
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                if (myq < nq_c) {
                    long long row_c = (long long)(row0_c + myq) * (long long)top_k;
                    if (pt == 0) {
                        int _min_12 = ((s_cnt[myq]) < (cand_cap) ? (s_cnt[myq]) : (cand_cap));
                        s_pub[myq] = _min_12;
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                    int cnt_f = s_pub[myq];
                    unsigned long long edge_f = 0ull;
                    if (cnt_f > top_k) {
                        int k_rem_1 = top_k;
                        int kept_above_1 = 0;
                        unsigned long long prefix_1 = 0ull;
                        unsigned long long edge_1 = 0ull;
                        int kept_1 = cnt_f;
                        #pragma unroll 1
                        for (int p_1 = 0; p_1 < 8; p_1++) {
                            unsigned long long shift_1 = (unsigned long long)(56 - 8 * p_1);
                            unsigned long long hi_shift_1 = ((p_1 > 0) ? shift_1 + 8ull : 56ull);
                            s_hist[hbase + pt * 4] = 0;
                            s_hist[hbase + pt * 4 + 1] = 0;
                            s_hist[hbase + pt * 4 + 2] = 0;
                            s_hist[hbase + pt * 4 + 3] = 0;
                            asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                            #pragma unroll 1
                            for (int c0_2 = 0; c0_2 < cnt_f; c0_2 += 512) {
                                unsigned long long keys_h_1[8];
                                int e_8 = c0_2 + pt;
                                keys_h_1[0] = 0ull;
                                if (e_8 < cnt_f) {
                                    unsigned int bits_e_14 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_8] >> 32ull);
                                    unsigned int mag_e_14 = bits_e_14 & 2147483647;
                                    unsigned int m_e_14 = ((mag_e_14 != 0) ? bits_e_14 : 0);
                                    unsigned int k32_e_14 = (((m_e_14 & 2147483648u) != 0) ? ~m_e_14 : m_e_14 | 2147483648u);
                                    if (mag_e_14 > 2139095040) {
                                        k32_e_14 = 0;
                                    }
                                    unsigned long long key_e_14 = (unsigned long long)k32_e_14 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_8] & 4294967295ull;
                                    keys_h_1[0] = key_e_14;
                                }
                                int e_9 = c0_2 + 64 + pt;
                                keys_h_1[1] = 0ull;
                                if (e_9 < cnt_f) {
                                    unsigned int bits_e_15 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_9] >> 32ull);
                                    unsigned int mag_e_15 = bits_e_15 & 2147483647;
                                    unsigned int m_e_15 = ((mag_e_15 != 0) ? bits_e_15 : 0);
                                    unsigned int k32_e_15 = (((m_e_15 & 2147483648u) != 0) ? ~m_e_15 : m_e_15 | 2147483648u);
                                    if (mag_e_15 > 2139095040) {
                                        k32_e_15 = 0;
                                    }
                                    unsigned long long key_e_15 = (unsigned long long)k32_e_15 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_9] & 4294967295ull;
                                    keys_h_1[1] = key_e_15;
                                }
                                int e_10 = c0_2 + 128 + pt;
                                keys_h_1[2] = 0ull;
                                if (e_10 < cnt_f) {
                                    unsigned int bits_e_16 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_10] >> 32ull);
                                    unsigned int mag_e_16 = bits_e_16 & 2147483647;
                                    unsigned int m_e_16 = ((mag_e_16 != 0) ? bits_e_16 : 0);
                                    unsigned int k32_e_16 = (((m_e_16 & 2147483648u) != 0) ? ~m_e_16 : m_e_16 | 2147483648u);
                                    if (mag_e_16 > 2139095040) {
                                        k32_e_16 = 0;
                                    }
                                    unsigned long long key_e_16 = (unsigned long long)k32_e_16 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_10] & 4294967295ull;
                                    keys_h_1[2] = key_e_16;
                                }
                                int e_11 = c0_2 + 192 + pt;
                                keys_h_1[3] = 0ull;
                                if (e_11 < cnt_f) {
                                    unsigned int bits_e_17 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_11] >> 32ull);
                                    unsigned int mag_e_17 = bits_e_17 & 2147483647;
                                    unsigned int m_e_17 = ((mag_e_17 != 0) ? bits_e_17 : 0);
                                    unsigned int k32_e_17 = (((m_e_17 & 2147483648u) != 0) ? ~m_e_17 : m_e_17 | 2147483648u);
                                    if (mag_e_17 > 2139095040) {
                                        k32_e_17 = 0;
                                    }
                                    unsigned long long key_e_17 = (unsigned long long)k32_e_17 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_11] & 4294967295ull;
                                    keys_h_1[3] = key_e_17;
                                }
                                int e_12 = c0_2 + 256 + pt;
                                keys_h_1[4] = 0ull;
                                if (e_12 < cnt_f) {
                                    unsigned int bits_e_18 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_12] >> 32ull);
                                    unsigned int mag_e_18 = bits_e_18 & 2147483647;
                                    unsigned int m_e_18 = ((mag_e_18 != 0) ? bits_e_18 : 0);
                                    unsigned int k32_e_18 = (((m_e_18 & 2147483648u) != 0) ? ~m_e_18 : m_e_18 | 2147483648u);
                                    if (mag_e_18 > 2139095040) {
                                        k32_e_18 = 0;
                                    }
                                    unsigned long long key_e_18 = (unsigned long long)k32_e_18 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_12] & 4294967295ull;
                                    keys_h_1[4] = key_e_18;
                                }
                                int e_13 = c0_2 + 320 + pt;
                                keys_h_1[5] = 0ull;
                                if (e_13 < cnt_f) {
                                    unsigned int bits_e_19 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_13] >> 32ull);
                                    unsigned int mag_e_19 = bits_e_19 & 2147483647;
                                    unsigned int m_e_19 = ((mag_e_19 != 0) ? bits_e_19 : 0);
                                    unsigned int k32_e_19 = (((m_e_19 & 2147483648u) != 0) ? ~m_e_19 : m_e_19 | 2147483648u);
                                    if (mag_e_19 > 2139095040) {
                                        k32_e_19 = 0;
                                    }
                                    unsigned long long key_e_19 = (unsigned long long)k32_e_19 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_13] & 4294967295ull;
                                    keys_h_1[5] = key_e_19;
                                }
                                int e_14 = c0_2 + 384 + pt;
                                keys_h_1[6] = 0ull;
                                if (e_14 < cnt_f) {
                                    unsigned int bits_e_20 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_14] >> 32ull);
                                    unsigned int mag_e_20 = bits_e_20 & 2147483647;
                                    unsigned int m_e_20 = ((mag_e_20 != 0) ? bits_e_20 : 0);
                                    unsigned int k32_e_20 = (((m_e_20 & 2147483648u) != 0) ? ~m_e_20 : m_e_20 | 2147483648u);
                                    if (mag_e_20 > 2139095040) {
                                        k32_e_20 = 0;
                                    }
                                    unsigned long long key_e_20 = (unsigned long long)k32_e_20 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_14] & 4294967295ull;
                                    keys_h_1[6] = key_e_20;
                                }
                                int e_15 = c0_2 + 448 + pt;
                                keys_h_1[7] = 0ull;
                                if (e_15 < cnt_f) {
                                    unsigned int bits_e_21 = (unsigned int)((unsigned long long)Cand[cbq[pr] + (long long)e_15] >> 32ull);
                                    unsigned int mag_e_21 = bits_e_21 & 2147483647;
                                    unsigned int m_e_21 = ((mag_e_21 != 0) ? bits_e_21 : 0);
                                    unsigned int k32_e_21 = (((m_e_21 & 2147483648u) != 0) ? ~m_e_21 : m_e_21 | 2147483648u);
                                    if (mag_e_21 > 2139095040) {
                                        k32_e_21 = 0;
                                    }
                                    unsigned long long key_e_21 = (unsigned long long)k32_e_21 << 32ull | (unsigned long long)Cand[cbq[pr] + (long long)e_15] & 4294967295ull;
                                    keys_h_1[7] = key_e_21;
                                }
                                if (c0_2 + pt < cnt_f) {
                                    if (p_1 == 0 || keys_h_1[0] >> hi_shift_1 == prefix_1) {
                                        atomicAdd(&s_hist[hbase + (int)(keys_h_1[0] >> shift_1 & 255ull)], 1);
                                    }
                                }
                                if (c0_2 + 64 + pt < cnt_f) {
                                    if (p_1 == 0 || keys_h_1[1] >> hi_shift_1 == prefix_1) {
                                        atomicAdd(&s_hist[hbase + (int)(keys_h_1[1] >> shift_1 & 255ull)], 1);
                                    }
                                }
                                if (c0_2 + 128 + pt < cnt_f) {
                                    if (p_1 == 0 || keys_h_1[2] >> hi_shift_1 == prefix_1) {
                                        atomicAdd(&s_hist[hbase + (int)(keys_h_1[2] >> shift_1 & 255ull)], 1);
                                    }
                                }
                                if (c0_2 + 192 + pt < cnt_f) {
                                    if (p_1 == 0 || keys_h_1[3] >> hi_shift_1 == prefix_1) {
                                        atomicAdd(&s_hist[hbase + (int)(keys_h_1[3] >> shift_1 & 255ull)], 1);
                                    }
                                }
                                if (c0_2 + 256 + pt < cnt_f) {
                                    if (p_1 == 0 || keys_h_1[4] >> hi_shift_1 == prefix_1) {
                                        atomicAdd(&s_hist[hbase + (int)(keys_h_1[4] >> shift_1 & 255ull)], 1);
                                    }
                                }
                                if (c0_2 + 320 + pt < cnt_f) {
                                    if (p_1 == 0 || keys_h_1[5] >> hi_shift_1 == prefix_1) {
                                        atomicAdd(&s_hist[hbase + (int)(keys_h_1[5] >> shift_1 & 255ull)], 1);
                                    }
                                }
                                if (c0_2 + 384 + pt < cnt_f) {
                                    if (p_1 == 0 || keys_h_1[6] >> hi_shift_1 == prefix_1) {
                                        atomicAdd(&s_hist[hbase + (int)(keys_h_1[6] >> shift_1 & 255ull)], 1);
                                    }
                                }
                                if (c0_2 + 448 + pt < cnt_f) {
                                    if (p_1 == 0 || keys_h_1[7] >> hi_shift_1 == prefix_1) {
                                        atomicAdd(&s_hist[hbase + (int)(keys_h_1[7] >> shift_1 & 255ull)], 1);
                                    }
                                }
                            }
                            asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                            int lane_bins_1[8];
                            int lane_sum_1 = 0;
                            lane_bins_1[0] = s_hist[hbase + lane_c * 8];
                            lane_sum_1 += lane_bins_1[0];
                            lane_bins_1[1] = s_hist[hbase + lane_c * 8 + 1];
                            lane_sum_1 += lane_bins_1[1];
                            lane_bins_1[2] = s_hist[hbase + lane_c * 8 + 2];
                            lane_sum_1 += lane_bins_1[2];
                            lane_bins_1[3] = s_hist[hbase + lane_c * 8 + 3];
                            lane_sum_1 += lane_bins_1[3];
                            lane_bins_1[4] = s_hist[hbase + lane_c * 8 + 4];
                            lane_sum_1 += lane_bins_1[4];
                            lane_bins_1[5] = s_hist[hbase + lane_c * 8 + 5];
                            lane_sum_1 += lane_bins_1[5];
                            lane_bins_1[6] = s_hist[hbase + lane_c * 8 + 6];
                            lane_sum_1 += lane_bins_1[6];
                            lane_bins_1[7] = s_hist[hbase + lane_c * 8 + 7];
                            lane_sum_1 += lane_bins_1[7];
                            int suffix_1 = lane_sum_1;
                            int above_part_5 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 1, 32);
                            if (lane_c + 1 < 32) {
                                suffix_1 += above_part_5;
                            }
                            int above_part_6 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 2, 32);
                            if (lane_c + 2 < 32) {
                                suffix_1 += above_part_6;
                            }
                            int above_part_7 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 4, 32);
                            if (lane_c + 4 < 32) {
                                suffix_1 += above_part_7;
                            }
                            int above_part_8 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 8, 32);
                            if (lane_c + 8 < 32) {
                                suffix_1 += above_part_8;
                            }
                            int above_part_9 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 16, 32);
                            if (lane_c + 16 < 32) {
                                suffix_1 += above_part_9;
                            }
                            int excl_1 = suffix_1 - lane_sum_1;
                            int is_target_1 = ((excl_1 < k_rem_1 && k_rem_1 <= excl_1 + lane_sum_1) ? 1 : 0);
                            int d_sel_1 = 0;
                            int above_sel_1 = 0;
                            int count_sel_1 = 0;
                            int found_1 = 0;
                            int cum_above_1 = excl_1;
                            if (found_1 == 0) {
                                if (k_rem_1 <= cum_above_1 + lane_bins_1[7]) {
                                    d_sel_1 = 7;
                                    above_sel_1 = cum_above_1;
                                    count_sel_1 = lane_bins_1[7];
                                    found_1 = 1;
                                }
                            }
                            cum_above_1 += lane_bins_1[7];
                            if (found_1 == 0) {
                                if (k_rem_1 <= cum_above_1 + lane_bins_1[6]) {
                                    d_sel_1 = 6;
                                    above_sel_1 = cum_above_1;
                                    count_sel_1 = lane_bins_1[6];
                                    found_1 = 1;
                                }
                            }
                            cum_above_1 += lane_bins_1[6];
                            if (found_1 == 0) {
                                if (k_rem_1 <= cum_above_1 + lane_bins_1[5]) {
                                    d_sel_1 = 5;
                                    above_sel_1 = cum_above_1;
                                    count_sel_1 = lane_bins_1[5];
                                    found_1 = 1;
                                }
                            }
                            cum_above_1 += lane_bins_1[5];
                            if (found_1 == 0) {
                                if (k_rem_1 <= cum_above_1 + lane_bins_1[4]) {
                                    d_sel_1 = 4;
                                    above_sel_1 = cum_above_1;
                                    count_sel_1 = lane_bins_1[4];
                                    found_1 = 1;
                                }
                            }
                            cum_above_1 += lane_bins_1[4];
                            if (found_1 == 0) {
                                if (k_rem_1 <= cum_above_1 + lane_bins_1[3]) {
                                    d_sel_1 = 3;
                                    above_sel_1 = cum_above_1;
                                    count_sel_1 = lane_bins_1[3];
                                    found_1 = 1;
                                }
                            }
                            cum_above_1 += lane_bins_1[3];
                            if (found_1 == 0) {
                                if (k_rem_1 <= cum_above_1 + lane_bins_1[2]) {
                                    d_sel_1 = 2;
                                    above_sel_1 = cum_above_1;
                                    count_sel_1 = lane_bins_1[2];
                                    found_1 = 1;
                                }
                            }
                            cum_above_1 += lane_bins_1[2];
                            if (found_1 == 0) {
                                if (k_rem_1 <= cum_above_1 + lane_bins_1[1]) {
                                    d_sel_1 = 1;
                                    above_sel_1 = cum_above_1;
                                    count_sel_1 = lane_bins_1[1];
                                    found_1 = 1;
                                }
                            }
                            cum_above_1 += lane_bins_1[1];
                            if (found_1 == 0) {
                                if (k_rem_1 <= cum_above_1 + lane_bins_1[0]) {
                                    d_sel_1 = 0;
                                    above_sel_1 = cum_above_1;
                                    count_sel_1 = lane_bins_1[0];
                                    found_1 = 1;
                                }
                            }
                            cum_above_1 += lane_bins_1[0];
                            int target_lane_1;
                            asm volatile("redux.sync.max.s32 %0, %1, 0xFFFFFFFF;" : "=r"(target_lane_1) : "r"(is_target_1 * lane_c));
                            int digit_sel_1 = __shfl_sync(0xFFFFFFFF, d_sel_1, target_lane_1);
                            int above_cnt_1 = __shfl_sync(0xFFFFFFFF, above_sel_1, target_lane_1);
                            int bucket_cnt_1 = __shfl_sync(0xFFFFFFFF, count_sel_1, target_lane_1);
                            k_rem_1 = k_rem_1 - above_cnt_1;
                            kept_above_1 += above_cnt_1;
                            prefix_1 = prefix_1 << 8ull | (unsigned long long)(unsigned int)(target_lane_1 * 8 + digit_sel_1);
                            edge_1 = prefix_1 << shift_1;
                            kept_1 = kept_above_1 + bucket_cnt_1;
                            if (pt == 0) {
                                s_pub[4 + myq] = kept_1;
                            }
                            asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                            if (s_pub[4 + myq] <= top_k) {
                                break;
                            }
                        }
                        edge_f = edge_1;
                    }
                    if (pt == 0) {
                        s_cnt[4 + myq] = 0;
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                    #pragma unroll 1
                    for (int c0_3 = 0; c0_3 < cnt_f; c0_3 += 256) {
                        int tot_f = 0;
                        unsigned long long entf[4];
                        unsigned int keepf[4];
                        int e_f_0 = c0_3 + pt;
                        entf[0] = 0ull;
                        int kf_0 = 0;
                        if (e_f_0 < cnt_f) {
                            entf[0] = (unsigned long long)Cand[cbq[pr] + (long long)e_f_0];
                            unsigned int bits_e_22 = (unsigned int)(entf[0] >> 32ull);
                            unsigned int mag_e_22 = bits_e_22 & 2147483647;
                            unsigned int m_e_22 = ((mag_e_22 != 0) ? bits_e_22 : 0);
                            unsigned int k32_e_22 = (((m_e_22 & 2147483648u) != 0) ? ~m_e_22 : m_e_22 | 2147483648u);
                            if (mag_e_22 > 2139095040) {
                                k32_e_22 = 0;
                            }
                            unsigned long long key_e_22 = (unsigned long long)k32_e_22 << 32ull | entf[0] & 4294967295ull;
                            if (key_e_22 >= edge_f) {
                                kf_0 = 1;
                            }
                        }
                        unsigned int _warp_vote_13 = __ballot_sync(0xFFFFFFFF, kf_0 != 0);
                        keepf[0] = _warp_vote_13;
                        tot_f += (int)(unsigned int)__popc(keepf[0]);
                        int e_f_1 = c0_3 + 64 + pt;
                        entf[1] = 0ull;
                        int kf_1 = 0;
                        if (e_f_1 < cnt_f) {
                            entf[1] = (unsigned long long)Cand[cbq[pr] + (long long)e_f_1];
                            unsigned int bits_e_23 = (unsigned int)(entf[1] >> 32ull);
                            unsigned int mag_e_23 = bits_e_23 & 2147483647;
                            unsigned int m_e_23 = ((mag_e_23 != 0) ? bits_e_23 : 0);
                            unsigned int k32_e_23 = (((m_e_23 & 2147483648u) != 0) ? ~m_e_23 : m_e_23 | 2147483648u);
                            if (mag_e_23 > 2139095040) {
                                k32_e_23 = 0;
                            }
                            unsigned long long key_e_23 = (unsigned long long)k32_e_23 << 32ull | entf[1] & 4294967295ull;
                            if (key_e_23 >= edge_f) {
                                kf_1 = 1;
                            }
                        }
                        unsigned int _warp_vote_14 = __ballot_sync(0xFFFFFFFF, kf_1 != 0);
                        keepf[1] = _warp_vote_14;
                        tot_f += (int)(unsigned int)__popc(keepf[1]);
                        int e_f_2 = c0_3 + 128 + pt;
                        entf[2] = 0ull;
                        int kf_2 = 0;
                        if (e_f_2 < cnt_f) {
                            entf[2] = (unsigned long long)Cand[cbq[pr] + (long long)e_f_2];
                            unsigned int bits_e_24 = (unsigned int)(entf[2] >> 32ull);
                            unsigned int mag_e_24 = bits_e_24 & 2147483647;
                            unsigned int m_e_24 = ((mag_e_24 != 0) ? bits_e_24 : 0);
                            unsigned int k32_e_24 = (((m_e_24 & 2147483648u) != 0) ? ~m_e_24 : m_e_24 | 2147483648u);
                            if (mag_e_24 > 2139095040) {
                                k32_e_24 = 0;
                            }
                            unsigned long long key_e_24 = (unsigned long long)k32_e_24 << 32ull | entf[2] & 4294967295ull;
                            if (key_e_24 >= edge_f) {
                                kf_2 = 1;
                            }
                        }
                        unsigned int _warp_vote_15 = __ballot_sync(0xFFFFFFFF, kf_2 != 0);
                        keepf[2] = _warp_vote_15;
                        tot_f += (int)(unsigned int)__popc(keepf[2]);
                        int e_f_3 = c0_3 + 192 + pt;
                        entf[3] = 0ull;
                        int kf_3 = 0;
                        if (e_f_3 < cnt_f) {
                            entf[3] = (unsigned long long)Cand[cbq[pr] + (long long)e_f_3];
                            unsigned int bits_e_25 = (unsigned int)(entf[3] >> 32ull);
                            unsigned int mag_e_25 = bits_e_25 & 2147483647;
                            unsigned int m_e_25 = ((mag_e_25 != 0) ? bits_e_25 : 0);
                            unsigned int k32_e_25 = (((m_e_25 & 2147483648u) != 0) ? ~m_e_25 : m_e_25 | 2147483648u);
                            if (mag_e_25 > 2139095040) {
                                k32_e_25 = 0;
                            }
                            unsigned long long key_e_25 = (unsigned long long)k32_e_25 << 32ull | entf[3] & 4294967295ull;
                            if (key_e_25 >= edge_f) {
                                kf_3 = 1;
                            }
                        }
                        unsigned int _warp_vote_16 = __ballot_sync(0xFFFFFFFF, kf_3 != 0);
                        keepf[3] = _warp_vote_16;
                        tot_f += (int)(unsigned int)__popc(keepf[3]);
                        int fbase = 0;
                        if (lane_c == 0) {
                            int _atomic_fetch_add_17 = atomicAdd(&s_cnt[4 + myq], tot_f);
                            fbase = _atomic_fetch_add_17;
                        }
                        int _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, fbase, 0);
                        fbase = _shfl_sync_18;
                        if ((keepf[0] >> (unsigned int)lane_c & 1) != 0) {
                            int pos_f_0 = fbase + (int)(unsigned int)__popc(keepf[0] & lt_mask);
                            if (pos_f_0 < top_k) {
                                Indices[row_c + (long long)pos_f_0] = (int)(unsigned int)(entf[0] & 4294967295ull);
                                Scores[row_c + (long long)pos_f_0] = __uint_as_float((unsigned int)(entf[0] >> 32ull));
                            }
                        }
                        fbase += (int)(unsigned int)__popc(keepf[0]);
                        if ((keepf[1] >> (unsigned int)lane_c & 1) != 0) {
                            int pos_f_1 = fbase + (int)(unsigned int)__popc(keepf[1] & lt_mask);
                            if (pos_f_1 < top_k) {
                                Indices[row_c + (long long)pos_f_1] = (int)(unsigned int)(entf[1] & 4294967295ull);
                                Scores[row_c + (long long)pos_f_1] = __uint_as_float((unsigned int)(entf[1] >> 32ull));
                            }
                        }
                        fbase += (int)(unsigned int)__popc(keepf[1]);
                        if ((keepf[2] >> (unsigned int)lane_c & 1) != 0) {
                            int pos_f_2 = fbase + (int)(unsigned int)__popc(keepf[2] & lt_mask);
                            if (pos_f_2 < top_k) {
                                Indices[row_c + (long long)pos_f_2] = (int)(unsigned int)(entf[2] & 4294967295ull);
                                Scores[row_c + (long long)pos_f_2] = __uint_as_float((unsigned int)(entf[2] >> 32ull));
                            }
                        }
                        fbase += (int)(unsigned int)__popc(keepf[2]);
                        if ((keepf[3] >> (unsigned int)lane_c & 1) != 0) {
                            int pos_f_3 = fbase + (int)(unsigned int)__popc(keepf[3] & lt_mask);
                            if (pos_f_3 < top_k) {
                                Indices[row_c + (long long)pos_f_3] = (int)(unsigned int)(entf[3] & 4294967295ull);
                                Scores[row_c + (long long)pos_f_3] = __uint_as_float((unsigned int)(entf[3] >> 32ull));
                            }
                        }
                        fbase += (int)(unsigned int)__popc(keepf[3]);
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(3 + myq) : "memory");
                    int written = ((s_cnt[4 + myq]) < (top_k) ? (s_cnt[4 + myq]) : (top_k));
                    #pragma unroll 1
                    for (int j_2 = written + pt; j_2 < top_k; j_2 += 64) {
                        Indices[row_c + (long long)j_2] = -1;
                        Scores[row_c + (long long)j_2] = __uint_as_float(4286578688u);
                    }
                }
                if (pt == 0) {
                    s_cnt[myq] = 0;
                    s_tau[myq] = 0ull;
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                if (quarter == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(plan_empty_addr + (pc) * 8);
                    }
                }
                pc += 1;
                if (pc == 2) { pc = 0; _phase_plan_tick ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            unsigned int pc_1 = 0;
            unsigned int kc = 0;
            unsigned int ts_1 = 0;
            int part_1 = 0;
            #pragma unroll 1
            for (int s0_1 = 0; s0_1 < num_segments; s0_1 += 32) {
                int s_c_1 = s0_1 + lane;
                if (s_c_1 < num_segments) {
                    part_1 += (cu_seqlens_q[s_c_1 + 1] - cu_seqlens_q[s_c_1] + 3) / 4;
                }
            }
            int n_total_1;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(n_total_1) : "r"(part_1));
            unsigned int _phase_plan_tick_1 = 0;
            unsigned int _phase_t_empty = 1;
            unsigned int _phase_k_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int it_m = 0; it_m < n_total_1 + 1; it_m++) {
                    mbarrier_wait(plan_tick_addr + (pc_1) * 8, _phase_plan_tick_1);
                    int hb_m = (int)pc_1 * 16;
                    if (s_hdr[hb_m] == 0) {
                        break;
                    }
                    mbarrier_wait(plan_full_addr + (pc_1) * 8, _phase_plan_tick_1);
                    int steps_m = s_hdr[hb_m + 12];
                    int m_v[4];
                    m_v[0] = s_hdr[hb_m + 8];
                    m_v[1] = s_hdr[hb_m + 8 + 1];
                    m_v[2] = s_hdr[hb_m + 8 + 2];
                    m_v[3] = s_hdr[hb_m + 8 + 3];
                    mbarrier_wait(q_full_addr + (pc_1) * 8, _phase_plan_tick_1);
                    #pragma unroll 1
                    for (int i_1 = 0; i_1 < steps_m; i_1++) {
                        mbarrier_wait(t_empty_addr + (ts_1) * 8, _phase_t_empty);
                        if (i_1 * 16 < m_v[0]) {
                            mbarrier_wait(k_full_addr + (kc) * 8, _phase_k_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
                            int _smem_desc_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (kc) * 2048;
                            int _smem_desc_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (pc_1) * 2048;
#else
                            int _smem_desc_lo_0 = (((smem_k_addr) >> 4) & 0x7FFF) + (kc) * 2048;
                            int _smem_desc_lo_1 = (((smem_q_addr) >> 4) & 0x7FFF) + (pc_1) * 2048;
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
                    "mov.b32 id, 134743184;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc + (ts_1 * 128))), "r"(0) : "memory");
                            tcgen05_commit(k_empty_addr + (kc) * 8);
                            kc += 1;
                            if (kc == 4) { kc = 0; _phase_k_full ^= 1; }
                        }
                        if (i_1 * 16 < m_v[1]) {
                            mbarrier_wait(k_full_addr + (kc) * 8, _phase_k_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
                            int _smem_desc_lo_2 = (((smem_k_addr) >> 4) & 0x3FFF) + (kc) * 2048;
                            int _smem_desc_lo_3 = (((smem_q_addr + 4096) >> 4) & 0x3FFF) + (pc_1) * 2048;
#else
                            int _smem_desc_lo_2 = (((smem_k_addr) >> 4) & 0x7FFF) + (kc) * 2048;
                            int _smem_desc_lo_3 = (((smem_q_addr + 4096) >> 4) & 0x7FFF) + (pc_1) * 2048;
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
                    "mov.b32 id, 134743184;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_acc + (ts_1 * 128 + 32))), "r"(0) : "memory");
                            tcgen05_commit(k_empty_addr + (kc) * 8);
                            kc += 1;
                            if (kc == 4) { kc = 0; _phase_k_full ^= 1; }
                        }
                        if (i_1 * 16 < m_v[2]) {
                            mbarrier_wait(k_full_addr + (kc) * 8, _phase_k_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
                            int _smem_desc_lo_4 = (((smem_k_addr) >> 4) & 0x3FFF) + (kc) * 2048;
                            int _smem_desc_lo_5 = (((smem_q_addr + 8192) >> 4) & 0x3FFF) + (pc_1) * 2048;
#else
                            int _smem_desc_lo_4 = (((smem_k_addr) >> 4) & 0x7FFF) + (kc) * 2048;
                            int _smem_desc_lo_5 = (((smem_q_addr + 8192) >> 4) & 0x7FFF) + (pc_1) * 2048;
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
                    "mov.b32 id, 134743184;\n\t"
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"((tmem_acc + (ts_1 * 128 + 64))), "r"(0) : "memory");
                            tcgen05_commit(k_empty_addr + (kc) * 8);
                            kc += 1;
                            if (kc == 4) { kc = 0; _phase_k_full ^= 1; }
                        }
                        if (i_1 * 16 < m_v[3]) {
                            mbarrier_wait(k_full_addr + (kc) * 8, _phase_k_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
                            int _smem_desc_lo_6 = (((smem_k_addr) >> 4) & 0x3FFF) + (kc) * 2048;
                            int _smem_desc_lo_7 = (((smem_q_addr + 12288) >> 4) & 0x3FFF) + (pc_1) * 2048;
#else
                            int _smem_desc_lo_6 = (((smem_k_addr) >> 4) & 0x7FFF) + (kc) * 2048;
                            int _smem_desc_lo_7 = (((smem_q_addr + 12288) >> 4) & 0x7FFF) + (pc_1) * 2048;
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
                    "mov.b32 id, 134743184;\n\t"
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"((tmem_acc + (ts_1 * 128 + 96))), "r"(0) : "memory");
                            tcgen05_commit(k_empty_addr + (kc) * 8);
                            kc += 1;
                            if (kc == 4) { kc = 0; _phase_k_full ^= 1; }
                        }
                        tcgen05_commit(t_full_addr + (ts_1) * 8);
                        ts_1 += 1;
                        if (ts_1 == 4) { ts_1 = 0; _phase_t_empty ^= 1; }
                    }
                    tcgen05_commit(q_empty_addr + (pc_1) * 8);
                    mbarrier_arrive(plan_empty_addr + (pc_1) * 8);
                    pc_1 += 1;
                    if (pc_1 == 2) { pc_1 = 0; _phase_plan_tick_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: plan ----
    if (warp == 9) {
        { // plan_main
            int lane_p = lane;
            unsigned int lt_p = (1 << (unsigned int)lane_p) - 1;
            unsigned int pc_2 = 0;
            int part_2 = 0;
            #pragma unroll 1
            for (int s0_2 = 0; s0_2 < num_segments; s0_2 += 32) {
                int s_c_2 = s0_2 + lane;
                if (s_c_2 < num_segments) {
                    part_2 += (cu_seqlens_q[s_c_2 + 1] - cu_seqlens_q[s_c_2] + 3) / 4;
                }
            }
            int n_total_2;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(n_total_2) : "r"(part_2));
            unsigned int limit_p = (unsigned int)(n_total_2 + grid_ctas - 1);
            int seg_cur = 0;
            int seg_base = 0;
            unsigned int _phase_plan_empty = 1;
            unsigned int _phase_plan_tick_2 = 0;
            #pragma unroll 1
            for (int it_p = 0; it_p < n_total_2 + 1; it_p++) {
                int tk = 0;
                if (lane_p == 0) {
                    uint32_t _atomic_fetch_inc_0;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_inc_0) : "l"(TicketWord), "r"(static_cast<uint32_t>(limit_p)) : "memory");
                    tk = (int)_atomic_fetch_inc_0;
                }
                int _shfl_sync_1_1 = __shfl_sync(0xFFFFFFFF, tk, 0);
                tk = _shfl_sync_1_1;
                mbarrier_wait(plan_empty_addr + (pc_2) * 8, _phase_plan_empty);
                int hb_p = (int)pc_2 * 16;
                if (lane_p == 0) {
                    s_hdr[hb_p] = ((tk < n_total_2) ? 1 : 0);
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(plan_tick_addr + (pc_2) * 8);
                }
                mbarrier_wait(plan_tick_addr + (pc_2) * 8, _phase_plan_tick_2);
                if (s_hdr[hb_p] == 0) {
                    break;
                }
                #pragma unroll 1
                for (int s_w_ = seg_cur; s_w_ < num_segments; s_w_++) {
                    int nqb_w = (cu_seqlens_q[s_w_ + 1] - cu_seqlens_q[s_w_] + 3) / 4;
                    if (tk < seg_base + nqb_w) {
                        break;
                    }
                    seg_base = seg_base + nqb_w;
                    seg_cur = s_w_ + 1;
                }
                int q0_p = cu_seqlens_q[seg_cur];
                int lq_p = cu_seqlens_q[seg_cur + 1] - q0_p;
                int k0_p = cu_seqlens_k[seg_cur];
                int lk_p = cu_seqlens_k[seg_cur + 1] - k0_p;
                int nqb_p = (lq_p + 3) / 4;
                int u0_p = (nqb_p - 1 - (tk - seg_base)) * 4;
                int nq_p = ((4) < (lq_p - u0_p) ? (4) : (lq_p - u0_p));
                int row0_p = q0_p + u0_p;
                long long off_p = (long long)(lk_p - lq_p);
                if (has_offsets != 0) {
                    off_p = q_offsets[seg_cur];
                }
                mbarrier_wait(q_empty_addr + (pc_2) * 8, _phase_plan_empty);
                if (elect_sync()) {
                    tma_3d_gmem2smem(smem_q_addr + pc_2 * 32768, (&Q), 0, row0_p * 32, 0, q_full_addr + (pc_2) * 8);
                    mbarrier_arrive_expect_tx(q_full_addr + (pc_2) * 8, 32768);
                }
                int steps_p = 0;
                long long lb_p = (cta * 2ll + (long long)pc_2) * 8192ll;
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    int n_q = 0;
                    int m_q = 0;
                    float wv = 0.0f;
                    float w_ld = 0.0f;
                    int nbv = 0;
                    int cw_q = 0;
                    if (q < nq_p) {
                        long long n64 = off_p + (long long)(u0_p + q + 1);
                        if (n64 < 0ll) {
                            n64 = 0ll;
                        }
                        if (n64 > (long long)lk_p) {
                            n64 = (long long)lk_p;
                        }
                        n_q = (int)n64;
                        nbv = (n_q >> 3) + (((n_q & 7) != 0) ? 1 : 0);
                        cw_q = cand_width;
                        w_ld = W[(long long)(row0_p + q) * 32ll + (long long)lane_p];
                    }
                    long long cbase = (long long)(row0_p + q) * (long long)cand_width;
                    long long lq_base = lb_p + (long long)(q * 2048);
                    #pragma unroll 1
                    for (int c0_4 = 0; c0_4 < cand_width; c0_4 += 512) {
                        int ids[16];
                        int c_j_0 = c0_4 + lane_p;
                        ids[0] = -1;
                        if (c_j_0 < cw_q) {
                            ids[0] = CandBlocks[cbase + (long long)c_j_0];
                        }
                        int c_j_1 = c0_4 + 32 + lane_p;
                        ids[1] = -1;
                        if (c_j_1 < cw_q) {
                            ids[1] = CandBlocks[cbase + (long long)c_j_1];
                        }
                        int c_j_2 = c0_4 + 64 + lane_p;
                        ids[2] = -1;
                        if (c_j_2 < cw_q) {
                            ids[2] = CandBlocks[cbase + (long long)c_j_2];
                        }
                        int c_j_3 = c0_4 + 96 + lane_p;
                        ids[3] = -1;
                        if (c_j_3 < cw_q) {
                            ids[3] = CandBlocks[cbase + (long long)c_j_3];
                        }
                        int c_j_4 = c0_4 + 128 + lane_p;
                        ids[4] = -1;
                        if (c_j_4 < cw_q) {
                            ids[4] = CandBlocks[cbase + (long long)c_j_4];
                        }
                        int c_j_5 = c0_4 + 160 + lane_p;
                        ids[5] = -1;
                        if (c_j_5 < cw_q) {
                            ids[5] = CandBlocks[cbase + (long long)c_j_5];
                        }
                        int c_j_6 = c0_4 + 192 + lane_p;
                        ids[6] = -1;
                        if (c_j_6 < cw_q) {
                            ids[6] = CandBlocks[cbase + (long long)c_j_6];
                        }
                        int c_j_7 = c0_4 + 224 + lane_p;
                        ids[7] = -1;
                        if (c_j_7 < cw_q) {
                            ids[7] = CandBlocks[cbase + (long long)c_j_7];
                        }
                        int c_j_8 = c0_4 + 256 + lane_p;
                        ids[8] = -1;
                        if (c_j_8 < cw_q) {
                            ids[8] = CandBlocks[cbase + (long long)c_j_8];
                        }
                        int c_j_9 = c0_4 + 288 + lane_p;
                        ids[9] = -1;
                        if (c_j_9 < cw_q) {
                            ids[9] = CandBlocks[cbase + (long long)c_j_9];
                        }
                        int c_j_10 = c0_4 + 320 + lane_p;
                        ids[10] = -1;
                        if (c_j_10 < cw_q) {
                            ids[10] = CandBlocks[cbase + (long long)c_j_10];
                        }
                        int c_j_11 = c0_4 + 352 + lane_p;
                        ids[11] = -1;
                        if (c_j_11 < cw_q) {
                            ids[11] = CandBlocks[cbase + (long long)c_j_11];
                        }
                        int c_j_12 = c0_4 + 384 + lane_p;
                        ids[12] = -1;
                        if (c_j_12 < cw_q) {
                            ids[12] = CandBlocks[cbase + (long long)c_j_12];
                        }
                        int c_j_13 = c0_4 + 416 + lane_p;
                        ids[13] = -1;
                        if (c_j_13 < cw_q) {
                            ids[13] = CandBlocks[cbase + (long long)c_j_13];
                        }
                        int c_j_14 = c0_4 + 448 + lane_p;
                        ids[14] = -1;
                        if (c_j_14 < cw_q) {
                            ids[14] = CandBlocks[cbase + (long long)c_j_14];
                        }
                        int c_j_15 = c0_4 + 480 + lane_p;
                        ids[15] = -1;
                        if (c_j_15 < cw_q) {
                            ids[15] = CandBlocks[cbase + (long long)c_j_15];
                        }
                        int keep_j_0 = ((ids[0] >= 0 && ids[0] < nbv) ? 1 : 0);
                        unsigned int vote_j_0 = __ballot_sync(0xFFFFFFFF, keep_j_0 != 0);
                        if (keep_j_0 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_0 & lt_p))] = ids[0];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_0);
                        int keep_j_1 = ((ids[1] >= 0 && ids[1] < nbv) ? 1 : 0);
                        unsigned int vote_j_1 = __ballot_sync(0xFFFFFFFF, keep_j_1 != 0);
                        if (keep_j_1 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_1 & lt_p))] = ids[1];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_1);
                        int keep_j_2 = ((ids[2] >= 0 && ids[2] < nbv) ? 1 : 0);
                        unsigned int vote_j_2 = __ballot_sync(0xFFFFFFFF, keep_j_2 != 0);
                        if (keep_j_2 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_2 & lt_p))] = ids[2];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_2);
                        int keep_j_3 = ((ids[3] >= 0 && ids[3] < nbv) ? 1 : 0);
                        unsigned int vote_j_3 = __ballot_sync(0xFFFFFFFF, keep_j_3 != 0);
                        if (keep_j_3 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_3 & lt_p))] = ids[3];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_3);
                        int keep_j_4 = ((ids[4] >= 0 && ids[4] < nbv) ? 1 : 0);
                        unsigned int vote_j_4 = __ballot_sync(0xFFFFFFFF, keep_j_4 != 0);
                        if (keep_j_4 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_4 & lt_p))] = ids[4];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_4);
                        int keep_j_5 = ((ids[5] >= 0 && ids[5] < nbv) ? 1 : 0);
                        unsigned int vote_j_5 = __ballot_sync(0xFFFFFFFF, keep_j_5 != 0);
                        if (keep_j_5 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_5 & lt_p))] = ids[5];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_5);
                        int keep_j_6 = ((ids[6] >= 0 && ids[6] < nbv) ? 1 : 0);
                        unsigned int vote_j_6 = __ballot_sync(0xFFFFFFFF, keep_j_6 != 0);
                        if (keep_j_6 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_6 & lt_p))] = ids[6];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_6);
                        int keep_j_7 = ((ids[7] >= 0 && ids[7] < nbv) ? 1 : 0);
                        unsigned int vote_j_7 = __ballot_sync(0xFFFFFFFF, keep_j_7 != 0);
                        if (keep_j_7 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_7 & lt_p))] = ids[7];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_7);
                        int keep_j_8 = ((ids[8] >= 0 && ids[8] < nbv) ? 1 : 0);
                        unsigned int vote_j_8 = __ballot_sync(0xFFFFFFFF, keep_j_8 != 0);
                        if (keep_j_8 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_8 & lt_p))] = ids[8];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_8);
                        int keep_j_9 = ((ids[9] >= 0 && ids[9] < nbv) ? 1 : 0);
                        unsigned int vote_j_9 = __ballot_sync(0xFFFFFFFF, keep_j_9 != 0);
                        if (keep_j_9 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_9 & lt_p))] = ids[9];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_9);
                        int keep_j_10 = ((ids[10] >= 0 && ids[10] < nbv) ? 1 : 0);
                        unsigned int vote_j_10 = __ballot_sync(0xFFFFFFFF, keep_j_10 != 0);
                        if (keep_j_10 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_10 & lt_p))] = ids[10];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_10);
                        int keep_j_11 = ((ids[11] >= 0 && ids[11] < nbv) ? 1 : 0);
                        unsigned int vote_j_11 = __ballot_sync(0xFFFFFFFF, keep_j_11 != 0);
                        if (keep_j_11 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_11 & lt_p))] = ids[11];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_11);
                        int keep_j_12 = ((ids[12] >= 0 && ids[12] < nbv) ? 1 : 0);
                        unsigned int vote_j_12 = __ballot_sync(0xFFFFFFFF, keep_j_12 != 0);
                        if (keep_j_12 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_12 & lt_p))] = ids[12];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_12);
                        int keep_j_13 = ((ids[13] >= 0 && ids[13] < nbv) ? 1 : 0);
                        unsigned int vote_j_13 = __ballot_sync(0xFFFFFFFF, keep_j_13 != 0);
                        if (keep_j_13 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_13 & lt_p))] = ids[13];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_13);
                        int keep_j_14 = ((ids[14] >= 0 && ids[14] < nbv) ? 1 : 0);
                        unsigned int vote_j_14 = __ballot_sync(0xFFFFFFFF, keep_j_14 != 0);
                        if (keep_j_14 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_14 & lt_p))] = ids[14];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_14);
                        int keep_j_15 = ((ids[15] >= 0 && ids[15] < nbv) ? 1 : 0);
                        unsigned int vote_j_15 = __ballot_sync(0xFFFFFFFF, keep_j_15 != 0);
                        if (keep_j_15 != 0) {
                            Lists[lq_base + (long long)(m_q + (int)(unsigned int)__popc(vote_j_15 & lt_p))] = ids[15];
                        }
                        m_q += (int)(unsigned int)__popc(vote_j_15);
                    }
                    if (q < nq_p) {
                        wv = w_ld * softmax_scale;
                    }
                    s_w[(int)pc_2 * 128 + q * 32 + lane_p] = wv;
                    if (lane_p == 0) {
                        s_hdr[hb_p + 4 + q] = n_q;
                        s_hdr[hb_p + 8 + q] = m_q;
                    }
                    int _max_2 = ((steps_p) > ((m_q + 16 - 1) / 16) ? (steps_p) : ((m_q + 16 - 1) / 16));
                    steps_p = _max_2;
                }
                if (lane_p == 0) {
                    s_hdr[hb_p + 1] = row0_p;
                    s_hdr[hb_p + 2] = nq_p;
                    s_hdr[hb_p + 3] = k0_p;
                    s_hdr[hb_p + 12] = steps_p;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(plan_full_addr + (pc_2) * 8);
                }
                pc_2 += 1;
                if (pc_2 == 2) { pc_2 = 0; _phase_plan_empty ^= 1; _phase_plan_tick_2 ^= 1; }
            }
        }
    }
    // ---- Role: load_k ----
    if (warp >= 10 && warp <= 15) {
        { // load_k_main
            int lane_l = lane;
            int lw = warp - 10;
            unsigned int pc_3 = 0;
            unsigned int kc_1 = 0;
            int part_3 = 0;
            #pragma unroll 1
            for (int s0_3 = 0; s0_3 < num_segments; s0_3 += 32) {
                int s_c_3 = s0_3 + lane;
                if (s_c_3 < num_segments) {
                    part_3 += (cu_seqlens_q[s_c_3 + 1] - cu_seqlens_q[s_c_3] + 3) / 4;
                }
            }
            int n_total_3;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(n_total_3) : "r"(part_3));
            unsigned int _phase_plan_tick_3 = 0;
            unsigned int _phase_k_empty = 1;
            #pragma unroll 1
            for (int it_l = 0; it_l < n_total_3 + 1; it_l++) {
                mbarrier_wait(plan_tick_addr + (pc_3) * 8, _phase_plan_tick_3);
                int hb_l = (int)pc_3 * 16;
                if (s_hdr[hb_l] == 0) {
                    break;
                }
                mbarrier_wait(plan_full_addr + (pc_3) * 8, _phase_plan_tick_3);
                int k0_l = s_hdr[hb_l + 3];
                int steps_l = s_hdr[hb_l + 12];
                long long lb_l = (cta * 2ll + (long long)pc_3) * 8192ll;
                int q_my = lane_l / 3;
                int s_my = lw + 6 * (lane_l - 3 * q_my);
                int m_my = 0;
                if (lane_l < 12) {
                    m_my = s_hdr[hb_l + 8 + q_my];
                }
                int n_u[4];
                int m_u[4];
                n_u[0] = s_hdr[hb_l + 4];
                m_u[0] = s_hdr[hb_l + 8];
                n_u[1] = s_hdr[hb_l + 4 + 1];
                m_u[1] = s_hdr[hb_l + 8 + 1];
                n_u[2] = s_hdr[hb_l + 4 + 2];
                m_u[2] = s_hdr[hb_l + 8 + 2];
                n_u[3] = s_hdr[hb_l + 4 + 3];
                m_u[3] = s_hdr[hb_l + 8 + 3];
                int b_next = 0;
                if (lane_l < 12 && s_my < 16 && s_my < m_my) {
                    b_next = Lists[lb_l + (long long)(q_my * 2048 + s_my)];
                }
                #pragma unroll 1
                for (int i_2 = 0; i_2 < steps_l; i_2++) {
                    int b_my = b_next;
                    int idx_next = (i_2 + 1) * 16 + s_my;
                    b_next = 0;
                    if (lane_l < 12 && s_my < 16 && idx_next < m_my) {
                        b_next = Lists[lb_l + (long long)(q_my * 2048 + idx_next)];
                    }
                    if (i_2 * 16 < m_u[0]) {
                        mbarrier_wait(k_empty_addr + (kc_1) * 8, _phase_k_empty);
                        int b_r_0[3];
                        int full_r_0[3];
                        int s_l_0 = lw;
                        int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, b_my, 0);
                        b_r_0[0] = _shfl_sync_0;
                        full_r_0[0] = 0;
                        if (s_l_0 < 16 && i_2 * 16 + s_l_0 < m_u[0]) {
                            int rows_l_0 = n_u[0] - 8 * b_r_0[0];
                            if (rows_l_0 >= 8) {
                                full_r_0[0] = 1;
                            } else {
                                unsigned int dst_p_0 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_0 * 1024);
                                #pragma unroll
                                for (int t_0 = 0; t_0 < 4; t_0++) {
                                    int e_l_0 = lane_l + 32 * t_0;
                                    if (e_l_0 < rows_l_0 * 16) {
                                        int row_l_0 = e_l_0 >> 4;
                                        int z_l_0 = e_l_0 >> 3 & 1;
                                        int ch_l_0 = e_l_0 & 7;
                                        long long src_l_0 = (long long)(k0_l + 8 * b_r_0[0] + row_l_0) * Kraw_stride_m2 + (long long)(z_l_0 * 64 + ch_l_0 * 8);
                                        unsigned int v_l_0[4];
                                        {
                                            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(Kraw + src_l_0 + 0);
                                            v_l_0[0 + 0] = _uv4_1.x;
                                            v_l_0[0 + 1] = _uv4_1.y;
                                            v_l_0[0 + 2] = _uv4_1.z;
                                            v_l_0[0 + 3] = _uv4_1.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_0 + (unsigned int)(z_l_0 * 16384 + row_l_0 * 128 + ((ch_l_0 ^ row_l_0) << 4))), "r"(v_l_0[0]), "r"(v_l_0[1]), "r"(v_l_0[2]), "r"(v_l_0[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        int s_l_1 = lw + 6;
                        int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, b_my, 1);
                        b_r_0[1] = _shfl_sync_2;
                        full_r_0[1] = 0;
                        if (s_l_1 < 16 && i_2 * 16 + s_l_1 < m_u[0]) {
                            int rows_l_1 = n_u[0] - 8 * b_r_0[1];
                            if (rows_l_1 >= 8) {
                                full_r_0[1] = 1;
                            } else {
                                unsigned int dst_p_1 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_1 * 1024);
                                #pragma unroll
                                for (int t_1 = 0; t_1 < 4; t_1++) {
                                    int e_l_1 = lane_l + 32 * t_1;
                                    if (e_l_1 < rows_l_1 * 16) {
                                        int row_l_1 = e_l_1 >> 4;
                                        int z_l_1 = e_l_1 >> 3 & 1;
                                        int ch_l_1 = e_l_1 & 7;
                                        long long src_l_1 = (long long)(k0_l + 8 * b_r_0[1] + row_l_1) * Kraw_stride_m2 + (long long)(z_l_1 * 64 + ch_l_1 * 8);
                                        unsigned int v_l_1[4];
                                        {
                                            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(Kraw + src_l_1 + 0);
                                            v_l_1[0 + 0] = _uv4_3.x;
                                            v_l_1[0 + 1] = _uv4_3.y;
                                            v_l_1[0 + 2] = _uv4_3.z;
                                            v_l_1[0 + 3] = _uv4_3.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_1 + (unsigned int)(z_l_1 * 16384 + row_l_1 * 128 + ((ch_l_1 ^ row_l_1) << 4))), "r"(v_l_1[0]), "r"(v_l_1[1]), "r"(v_l_1[2]), "r"(v_l_1[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        int s_l_2 = lw + 12;
                        int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, b_my, 2);
                        b_r_0[2] = _shfl_sync_4;
                        full_r_0[2] = 0;
                        if (s_l_2 < 16 && i_2 * 16 + s_l_2 < m_u[0]) {
                            int rows_l_2 = n_u[0] - 8 * b_r_0[2];
                            if (rows_l_2 >= 8) {
                                full_r_0[2] = 1;
                            } else {
                                unsigned int dst_p_2 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_2 * 1024);
                                #pragma unroll
                                for (int t_2 = 0; t_2 < 4; t_2++) {
                                    int e_l_2 = lane_l + 32 * t_2;
                                    if (e_l_2 < rows_l_2 * 16) {
                                        int row_l_2 = e_l_2 >> 4;
                                        int z_l_2 = e_l_2 >> 3 & 1;
                                        int ch_l_2 = e_l_2 & 7;
                                        long long src_l_2 = (long long)(k0_l + 8 * b_r_0[2] + row_l_2) * Kraw_stride_m2 + (long long)(z_l_2 * 64 + ch_l_2 * 8);
                                        unsigned int v_l_2[4];
                                        {
                                            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(Kraw + src_l_2 + 0);
                                            v_l_2[0 + 0] = _uv4_5.x;
                                            v_l_2[0 + 1] = _uv4_5.y;
                                            v_l_2[0 + 2] = _uv4_5.z;
                                            v_l_2[0 + 3] = _uv4_5.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_2 + (unsigned int)(z_l_2 * 16384 + row_l_2 * 128 + ((ch_l_2 ^ row_l_2) << 4))), "r"(v_l_2[0]), "r"(v_l_2[1]), "r"(v_l_2[2]), "r"(v_l_2[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            if (full_r_0[0] != 0) {
                                unsigned int row_l_3 = (unsigned int)(lw * 1024);
                                unsigned int dst_l_0 = smem_k_addr + kc_1 * 32768 + row_l_3;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_0 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_0[0]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_0 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_0[0]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            if (full_r_0[1] != 0) {
                                unsigned int row_l_4 = (unsigned int)((lw + 6) * 1024);
                                unsigned int dst_l_1 = smem_k_addr + kc_1 * 32768 + row_l_4;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_1 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_0[1]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_1 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_0[1]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            if (full_r_0[2] != 0) {
                                unsigned int row_l_5 = (unsigned int)((lw + 12) * 1024);
                                unsigned int dst_l_2 = smem_k_addr + kc_1 * 32768 + row_l_5;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_2 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_0[2]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_2 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_0[2]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            mbarrier_arrive(k_full_addr + (kc_1) * 8);
                        }
                        kc_1 += 1;
                        if (kc_1 == 4) { kc_1 = 0; _phase_k_empty ^= 1; }
                    }
                    if (i_2 * 16 < m_u[1]) {
                        mbarrier_wait(k_empty_addr + (kc_1) * 8, _phase_k_empty);
                        int b_r_1[3];
                        int full_r_1[3];
                        int s_l_3 = lw;
                        int _shfl_sync_6 = __shfl_sync(0xFFFFFFFF, b_my, 3);
                        b_r_1[0] = _shfl_sync_6;
                        full_r_1[0] = 0;
                        if (s_l_3 < 16 && i_2 * 16 + s_l_3 < m_u[1]) {
                            int rows_l_3 = n_u[1] - 8 * b_r_1[0];
                            if (rows_l_3 >= 8) {
                                full_r_1[0] = 1;
                            } else {
                                unsigned int dst_p_3 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_3 * 1024);
                                #pragma unroll
                                for (int t_3 = 0; t_3 < 4; t_3++) {
                                    int e_l_3 = lane_l + 32 * t_3;
                                    if (e_l_3 < rows_l_3 * 16) {
                                        int row_l_6 = e_l_3 >> 4;
                                        int z_l_3 = e_l_3 >> 3 & 1;
                                        int ch_l_3 = e_l_3 & 7;
                                        long long src_l_3 = (long long)(k0_l + 8 * b_r_1[0] + row_l_6) * Kraw_stride_m2 + (long long)(z_l_3 * 64 + ch_l_3 * 8);
                                        unsigned int v_l_3[4];
                                        {
                                            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(Kraw + src_l_3 + 0);
                                            v_l_3[0 + 0] = _uv4_7.x;
                                            v_l_3[0 + 1] = _uv4_7.y;
                                            v_l_3[0 + 2] = _uv4_7.z;
                                            v_l_3[0 + 3] = _uv4_7.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_3 + (unsigned int)(z_l_3 * 16384 + row_l_6 * 128 + ((ch_l_3 ^ row_l_6) << 4))), "r"(v_l_3[0]), "r"(v_l_3[1]), "r"(v_l_3[2]), "r"(v_l_3[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        int s_l_4 = lw + 6;
                        int _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, b_my, 4);
                        b_r_1[1] = _shfl_sync_8;
                        full_r_1[1] = 0;
                        if (s_l_4 < 16 && i_2 * 16 + s_l_4 < m_u[1]) {
                            int rows_l_4 = n_u[1] - 8 * b_r_1[1];
                            if (rows_l_4 >= 8) {
                                full_r_1[1] = 1;
                            } else {
                                unsigned int dst_p_4 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_4 * 1024);
                                #pragma unroll
                                for (int t_4 = 0; t_4 < 4; t_4++) {
                                    int e_l_4 = lane_l + 32 * t_4;
                                    if (e_l_4 < rows_l_4 * 16) {
                                        int row_l_7 = e_l_4 >> 4;
                                        int z_l_4 = e_l_4 >> 3 & 1;
                                        int ch_l_4 = e_l_4 & 7;
                                        long long src_l_4 = (long long)(k0_l + 8 * b_r_1[1] + row_l_7) * Kraw_stride_m2 + (long long)(z_l_4 * 64 + ch_l_4 * 8);
                                        unsigned int v_l_4[4];
                                        {
                                            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(Kraw + src_l_4 + 0);
                                            v_l_4[0 + 0] = _uv4_9.x;
                                            v_l_4[0 + 1] = _uv4_9.y;
                                            v_l_4[0 + 2] = _uv4_9.z;
                                            v_l_4[0 + 3] = _uv4_9.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_4 + (unsigned int)(z_l_4 * 16384 + row_l_7 * 128 + ((ch_l_4 ^ row_l_7) << 4))), "r"(v_l_4[0]), "r"(v_l_4[1]), "r"(v_l_4[2]), "r"(v_l_4[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        int s_l_5 = lw + 12;
                        int _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, b_my, 5);
                        b_r_1[2] = _shfl_sync_10;
                        full_r_1[2] = 0;
                        if (s_l_5 < 16 && i_2 * 16 + s_l_5 < m_u[1]) {
                            int rows_l_5 = n_u[1] - 8 * b_r_1[2];
                            if (rows_l_5 >= 8) {
                                full_r_1[2] = 1;
                            } else {
                                unsigned int dst_p_5 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_5 * 1024);
                                #pragma unroll
                                for (int t_5 = 0; t_5 < 4; t_5++) {
                                    int e_l_5 = lane_l + 32 * t_5;
                                    if (e_l_5 < rows_l_5 * 16) {
                                        int row_l_8 = e_l_5 >> 4;
                                        int z_l_5 = e_l_5 >> 3 & 1;
                                        int ch_l_5 = e_l_5 & 7;
                                        long long src_l_5 = (long long)(k0_l + 8 * b_r_1[2] + row_l_8) * Kraw_stride_m2 + (long long)(z_l_5 * 64 + ch_l_5 * 8);
                                        unsigned int v_l_5[4];
                                        {
                                            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(Kraw + src_l_5 + 0);
                                            v_l_5[0 + 0] = _uv4_11.x;
                                            v_l_5[0 + 1] = _uv4_11.y;
                                            v_l_5[0 + 2] = _uv4_11.z;
                                            v_l_5[0 + 3] = _uv4_11.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_5 + (unsigned int)(z_l_5 * 16384 + row_l_8 * 128 + ((ch_l_5 ^ row_l_8) << 4))), "r"(v_l_5[0]), "r"(v_l_5[1]), "r"(v_l_5[2]), "r"(v_l_5[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            if (full_r_1[0] != 0) {
                                unsigned int row_l_9 = (unsigned int)(lw * 1024);
                                unsigned int dst_l_3 = smem_k_addr + kc_1 * 32768 + row_l_9;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_3 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_1[0]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_3 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_1[0]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            if (full_r_1[1] != 0) {
                                unsigned int row_l_10 = (unsigned int)((lw + 6) * 1024);
                                unsigned int dst_l_4 = smem_k_addr + kc_1 * 32768 + row_l_10;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_4 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_1[1]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_4 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_1[1]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            if (full_r_1[2] != 0) {
                                unsigned int row_l_11 = (unsigned int)((lw + 12) * 1024);
                                unsigned int dst_l_5 = smem_k_addr + kc_1 * 32768 + row_l_11;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_5 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_1[2]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_5 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_1[2]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            mbarrier_arrive(k_full_addr + (kc_1) * 8);
                        }
                        kc_1 += 1;
                        if (kc_1 == 4) { kc_1 = 0; _phase_k_empty ^= 1; }
                    }
                    if (i_2 * 16 < m_u[2]) {
                        mbarrier_wait(k_empty_addr + (kc_1) * 8, _phase_k_empty);
                        int b_r_2[3];
                        int full_r_2[3];
                        int s_l_6 = lw;
                        int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, b_my, 6);
                        b_r_2[0] = _shfl_sync_12;
                        full_r_2[0] = 0;
                        if (s_l_6 < 16 && i_2 * 16 + s_l_6 < m_u[2]) {
                            int rows_l_6 = n_u[2] - 8 * b_r_2[0];
                            if (rows_l_6 >= 8) {
                                full_r_2[0] = 1;
                            } else {
                                unsigned int dst_p_6 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_6 * 1024);
                                #pragma unroll
                                for (int t_6 = 0; t_6 < 4; t_6++) {
                                    int e_l_6 = lane_l + 32 * t_6;
                                    if (e_l_6 < rows_l_6 * 16) {
                                        int row_l_12 = e_l_6 >> 4;
                                        int z_l_6 = e_l_6 >> 3 & 1;
                                        int ch_l_6 = e_l_6 & 7;
                                        long long src_l_6 = (long long)(k0_l + 8 * b_r_2[0] + row_l_12) * Kraw_stride_m2 + (long long)(z_l_6 * 64 + ch_l_6 * 8);
                                        unsigned int v_l_6[4];
                                        {
                                            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(Kraw + src_l_6 + 0);
                                            v_l_6[0 + 0] = _uv4_13.x;
                                            v_l_6[0 + 1] = _uv4_13.y;
                                            v_l_6[0 + 2] = _uv4_13.z;
                                            v_l_6[0 + 3] = _uv4_13.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_6 + (unsigned int)(z_l_6 * 16384 + row_l_12 * 128 + ((ch_l_6 ^ row_l_12) << 4))), "r"(v_l_6[0]), "r"(v_l_6[1]), "r"(v_l_6[2]), "r"(v_l_6[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        int s_l_7 = lw + 6;
                        int _shfl_sync_14 = __shfl_sync(0xFFFFFFFF, b_my, 7);
                        b_r_2[1] = _shfl_sync_14;
                        full_r_2[1] = 0;
                        if (s_l_7 < 16 && i_2 * 16 + s_l_7 < m_u[2]) {
                            int rows_l_7 = n_u[2] - 8 * b_r_2[1];
                            if (rows_l_7 >= 8) {
                                full_r_2[1] = 1;
                            } else {
                                unsigned int dst_p_7 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_7 * 1024);
                                #pragma unroll
                                for (int t_7 = 0; t_7 < 4; t_7++) {
                                    int e_l_7 = lane_l + 32 * t_7;
                                    if (e_l_7 < rows_l_7 * 16) {
                                        int row_l_13 = e_l_7 >> 4;
                                        int z_l_7 = e_l_7 >> 3 & 1;
                                        int ch_l_7 = e_l_7 & 7;
                                        long long src_l_7 = (long long)(k0_l + 8 * b_r_2[1] + row_l_13) * Kraw_stride_m2 + (long long)(z_l_7 * 64 + ch_l_7 * 8);
                                        unsigned int v_l_7[4];
                                        {
                                            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(Kraw + src_l_7 + 0);
                                            v_l_7[0 + 0] = _uv4_15.x;
                                            v_l_7[0 + 1] = _uv4_15.y;
                                            v_l_7[0 + 2] = _uv4_15.z;
                                            v_l_7[0 + 3] = _uv4_15.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_7 + (unsigned int)(z_l_7 * 16384 + row_l_13 * 128 + ((ch_l_7 ^ row_l_13) << 4))), "r"(v_l_7[0]), "r"(v_l_7[1]), "r"(v_l_7[2]), "r"(v_l_7[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        int s_l_8 = lw + 12;
                        int _shfl_sync_16 = __shfl_sync(0xFFFFFFFF, b_my, 8);
                        b_r_2[2] = _shfl_sync_16;
                        full_r_2[2] = 0;
                        if (s_l_8 < 16 && i_2 * 16 + s_l_8 < m_u[2]) {
                            int rows_l_8 = n_u[2] - 8 * b_r_2[2];
                            if (rows_l_8 >= 8) {
                                full_r_2[2] = 1;
                            } else {
                                unsigned int dst_p_8 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_8 * 1024);
                                #pragma unroll
                                for (int t_8 = 0; t_8 < 4; t_8++) {
                                    int e_l_8 = lane_l + 32 * t_8;
                                    if (e_l_8 < rows_l_8 * 16) {
                                        int row_l_14 = e_l_8 >> 4;
                                        int z_l_8 = e_l_8 >> 3 & 1;
                                        int ch_l_8 = e_l_8 & 7;
                                        long long src_l_8 = (long long)(k0_l + 8 * b_r_2[2] + row_l_14) * Kraw_stride_m2 + (long long)(z_l_8 * 64 + ch_l_8 * 8);
                                        unsigned int v_l_8[4];
                                        {
                                            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(Kraw + src_l_8 + 0);
                                            v_l_8[0 + 0] = _uv4_17.x;
                                            v_l_8[0 + 1] = _uv4_17.y;
                                            v_l_8[0 + 2] = _uv4_17.z;
                                            v_l_8[0 + 3] = _uv4_17.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_8 + (unsigned int)(z_l_8 * 16384 + row_l_14 * 128 + ((ch_l_8 ^ row_l_14) << 4))), "r"(v_l_8[0]), "r"(v_l_8[1]), "r"(v_l_8[2]), "r"(v_l_8[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            if (full_r_2[0] != 0) {
                                unsigned int row_l_15 = (unsigned int)(lw * 1024);
                                unsigned int dst_l_6 = smem_k_addr + kc_1 * 32768 + row_l_15;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_6 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_2[0]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_6 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_2[0]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            if (full_r_2[1] != 0) {
                                unsigned int row_l_16 = (unsigned int)((lw + 6) * 1024);
                                unsigned int dst_l_7 = smem_k_addr + kc_1 * 32768 + row_l_16;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_7 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_2[1]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_7 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_2[1]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            if (full_r_2[2] != 0) {
                                unsigned int row_l_17 = (unsigned int)((lw + 12) * 1024);
                                unsigned int dst_l_8 = smem_k_addr + kc_1 * 32768 + row_l_17;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_8 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_2[2]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_8 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_2[2]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            mbarrier_arrive(k_full_addr + (kc_1) * 8);
                        }
                        kc_1 += 1;
                        if (kc_1 == 4) { kc_1 = 0; _phase_k_empty ^= 1; }
                    }
                    if (i_2 * 16 < m_u[3]) {
                        mbarrier_wait(k_empty_addr + (kc_1) * 8, _phase_k_empty);
                        int b_r_3[3];
                        int full_r_3[3];
                        int s_l_9 = lw;
                        int _shfl_sync_18_1 = __shfl_sync(0xFFFFFFFF, b_my, 9);
                        b_r_3[0] = _shfl_sync_18_1;
                        full_r_3[0] = 0;
                        if (s_l_9 < 16 && i_2 * 16 + s_l_9 < m_u[3]) {
                            int rows_l_9 = n_u[3] - 8 * b_r_3[0];
                            if (rows_l_9 >= 8) {
                                full_r_3[0] = 1;
                            } else {
                                unsigned int dst_p_9 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_9 * 1024);
                                #pragma unroll
                                for (int t_9 = 0; t_9 < 4; t_9++) {
                                    int e_l_9 = lane_l + 32 * t_9;
                                    if (e_l_9 < rows_l_9 * 16) {
                                        int row_l_18 = e_l_9 >> 4;
                                        int z_l_9 = e_l_9 >> 3 & 1;
                                        int ch_l_9 = e_l_9 & 7;
                                        long long src_l_9 = (long long)(k0_l + 8 * b_r_3[0] + row_l_18) * Kraw_stride_m2 + (long long)(z_l_9 * 64 + ch_l_9 * 8);
                                        unsigned int v_l_9[4];
                                        {
                                            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(Kraw + src_l_9 + 0);
                                            v_l_9[0 + 0] = _uv4_19.x;
                                            v_l_9[0 + 1] = _uv4_19.y;
                                            v_l_9[0 + 2] = _uv4_19.z;
                                            v_l_9[0 + 3] = _uv4_19.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_9 + (unsigned int)(z_l_9 * 16384 + row_l_18 * 128 + ((ch_l_9 ^ row_l_18) << 4))), "r"(v_l_9[0]), "r"(v_l_9[1]), "r"(v_l_9[2]), "r"(v_l_9[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        int s_l_10 = lw + 6;
                        int _shfl_sync_20 = __shfl_sync(0xFFFFFFFF, b_my, 10);
                        b_r_3[1] = _shfl_sync_20;
                        full_r_3[1] = 0;
                        if (s_l_10 < 16 && i_2 * 16 + s_l_10 < m_u[3]) {
                            int rows_l_10 = n_u[3] - 8 * b_r_3[1];
                            if (rows_l_10 >= 8) {
                                full_r_3[1] = 1;
                            } else {
                                unsigned int dst_p_10 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_10 * 1024);
                                #pragma unroll
                                for (int t_10 = 0; t_10 < 4; t_10++) {
                                    int e_l_10 = lane_l + 32 * t_10;
                                    if (e_l_10 < rows_l_10 * 16) {
                                        int row_l_19 = e_l_10 >> 4;
                                        int z_l_10 = e_l_10 >> 3 & 1;
                                        int ch_l_10 = e_l_10 & 7;
                                        long long src_l_10 = (long long)(k0_l + 8 * b_r_3[1] + row_l_19) * Kraw_stride_m2 + (long long)(z_l_10 * 64 + ch_l_10 * 8);
                                        unsigned int v_l_10[4];
                                        {
                                            uint4 _uv4_21 = *reinterpret_cast<const uint4*>(Kraw + src_l_10 + 0);
                                            v_l_10[0 + 0] = _uv4_21.x;
                                            v_l_10[0 + 1] = _uv4_21.y;
                                            v_l_10[0 + 2] = _uv4_21.z;
                                            v_l_10[0 + 3] = _uv4_21.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_10 + (unsigned int)(z_l_10 * 16384 + row_l_19 * 128 + ((ch_l_10 ^ row_l_19) << 4))), "r"(v_l_10[0]), "r"(v_l_10[1]), "r"(v_l_10[2]), "r"(v_l_10[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        int s_l_11 = lw + 12;
                        int _shfl_sync_22 = __shfl_sync(0xFFFFFFFF, b_my, 11);
                        b_r_3[2] = _shfl_sync_22;
                        full_r_3[2] = 0;
                        if (s_l_11 < 16 && i_2 * 16 + s_l_11 < m_u[3]) {
                            int rows_l_11 = n_u[3] - 8 * b_r_3[2];
                            if (rows_l_11 >= 8) {
                                full_r_3[2] = 1;
                            } else {
                                unsigned int dst_p_11 = smem_k_addr + kc_1 * 32768 + (unsigned int)(s_l_11 * 1024);
                                #pragma unroll
                                for (int t_11 = 0; t_11 < 4; t_11++) {
                                    int e_l_11 = lane_l + 32 * t_11;
                                    if (e_l_11 < rows_l_11 * 16) {
                                        int row_l_20 = e_l_11 >> 4;
                                        int z_l_11 = e_l_11 >> 3 & 1;
                                        int ch_l_11 = e_l_11 & 7;
                                        long long src_l_11 = (long long)(k0_l + 8 * b_r_3[2] + row_l_20) * Kraw_stride_m2 + (long long)(z_l_11 * 64 + ch_l_11 * 8);
                                        unsigned int v_l_11[4];
                                        {
                                            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(Kraw + src_l_11 + 0);
                                            v_l_11[0 + 0] = _uv4_23.x;
                                            v_l_11[0 + 1] = _uv4_23.y;
                                            v_l_11[0 + 2] = _uv4_23.z;
                                            v_l_11[0 + 3] = _uv4_23.w;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_p_11 + (unsigned int)(z_l_11 * 16384 + row_l_20 * 128 + ((ch_l_11 ^ row_l_20) << 4))), "r"(v_l_11[0]), "r"(v_l_11[1]), "r"(v_l_11[2]), "r"(v_l_11[3]) : "memory");
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            if (full_r_3[0] != 0) {
                                unsigned int row_l_21 = (unsigned int)(lw * 1024);
                                unsigned int dst_l_9 = smem_k_addr + kc_1 * 32768 + row_l_21;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_9 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_3[0]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_9 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_3[0]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            if (full_r_3[1] != 0) {
                                unsigned int row_l_22 = (unsigned int)((lw + 6) * 1024);
                                unsigned int dst_l_10 = smem_k_addr + kc_1 * 32768 + row_l_22;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_10 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_3[1]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_10 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_3[1]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            if (full_r_3[2] != 0) {
                                unsigned int row_l_23 = (unsigned int)((lw + 12) * 1024);
                                unsigned int dst_l_11 = smem_k_addr + kc_1 * 32768 + row_l_23;
                                mbarrier_expect_tx(k_full_addr + (kc_1) * 8, 2048);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_11 + 0), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_3[2]), "r"(0),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(dst_l_11 + 16384), "l"((&K8)), "r"(0), "r"(k0_l + 8 * b_r_3[2]), "r"(1),
                                       "r"(k_full_addr + (kc_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                            mbarrier_arrive(k_full_addr + (kc_1) * 8);
                        }
                        kc_1 += 1;
                        if (kc_1 == 4) { kc_1 = 0; _phase_k_empty ^= 1; }
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(plan_empty_addr + (pc_3) * 8);
                }
                pc_3 += 1;
                if (pc_3 == 2) { pc_3 = 0; _phase_plan_tick_3 ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 8) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
