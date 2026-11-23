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





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(448, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_44f8275bc726f8a46e33_kcp(const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap v_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap h_tma, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ dAqk, float scale, int num_heads, int num_chunks)
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
    #define tiles_full_addr (mbar_base + 0)
    #define ops_ready_addr (mbar_base + 16)
    #define tiles_empty_addr (mbar_base + 32)
    #define h_full_addr (mbar_base + 48)
    #define h_empty_addr (mbar_base + 56)
    #define acc_full_addr (mbar_base + 64)
    #define acc_empty_addr (mbar_base + 72)
    #define t32_full_addr (mbar_base + 80)
    #define t32_empty_addr (mbar_base + 88)
    #define d_full_addr (mbar_base + 96)
    #define d_empty_addr (mbar_base + 112)
    #define work_full_addr (mbar_base + 128)
    #define work_empty_addr (mbar_base + 160)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* akk_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int akk_k_addr = smem + 1024;
    __nv_bfloat16* v_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int v_mn_addr = smem + 9216;
    __nv_bfloat16* w_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int w_k_addr = smem + 25600;
    __nv_bfloat16* do_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int do_s_addr = smem + 41984;
    __nv_bfloat16* h_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int h_p_addr = smem + 115712;
    __nv_bfloat16* h_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int h_mn_addr = smem + 115712;
    float* dot_k = reinterpret_cast<float*>(smem_raw + 148480);
    const int dot_k_addr = smem + 148480;
    float* vnt_k = reinterpret_cast<float*>(smem_raw + 181248);
    const int vnt_k_addr = smem + 181248;
    unsigned int* work_resp = reinterpret_cast<unsigned int*>(smem_raw + 214016);
    const int work_resp_addr = smem + 214016;

    // Mbarrier init (24 barriers at smem_raw[0..192))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // ops_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 16, 128);
            mbarrier_init(smem + 24, 128);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'one' ---
            // h_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // h_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // t32_full: 1 barriers, init_count=256
            mbarrier_init(smem + 80, 256);
            // t32_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'dbufs' ---
            // d_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // d_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            // --- pipeline 'work' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // work_empty: 4 barriers, init_count=386
            mbarrier_init(smem + 160, 386);
            mbarrier_init(smem + 168, 386);
            mbarrier_init(smem + 176, 386);
            mbarrier_init(smem + 184, 386);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_u = taddr;
    const unsigned int tmem_wh = taddr + 128;
    const unsigned int tmem_d = taddr + 256;

    // ---- Role: drain ----
    if (warp <= 3) {
        { // drain_main
            const int wq_d = warp & 3;
            const int t_row_d = wq_d * 16 + (lane & 15);
            const int half_d = lane >> 4;
            const int row_base_d = wq_d * 32 << 16;
            unsigned int work_stage_d = 0;
            int num_items_d = num_chunks * num_heads;
            int it_d = (int)bid;
            unsigned int _phase_work_full = 0;
            unsigned int _phase_acc_full = 0;
            unsigned int _phase_t32_empty = 1;
            #pragma unroll 1
            for (int _item_d = 0; _item_d < num_items_d; _item_d++) {
                mbarrier_wait(work_full_addr + (work_stage_d) * 8, _phase_work_full);
                uint32_t valid = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(valid)
                    : "r"(work_resp_addr + work_stage_d * 16 + 0 * 16)
                    : "memory");
                uint32_t cta = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(cta)
                    : "r"(work_resp_addr + work_stage_d * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_d) * 8);
                work_stage_d += 1;
                if (work_stage_d == 4) { work_stage_d = 0; _phase_work_full ^= 1; }
                int clen_d = chunk_len[it_d / num_heads];
                mbarrier_wait(acc_full_addr, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(t32_empty_addr, _phase_t32_empty);
                #pragma unroll
                for (int blk = 0; blk < 4; blk++) {
                    float u_r[16];
                    float wh_r[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&u_r[0])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[1])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[2])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[3])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[4])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[5])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[6])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[7])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[8])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[9])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[10])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[11])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[12])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[13])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[14])), "=r"(*reinterpret_cast<uint32_t*>(&u_r[15]))
                        : "r"(taddr + (unsigned int)row_base_d + (unsigned int)(blk * 32)));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wh_r[0])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[1])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[2])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[3])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[4])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[5])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[6])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[7])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[8])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[9])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[10])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[11])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[12])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[13])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[14])), "=r"(*reinterpret_cast<uint32_t*>(&wh_r[15]))
                        : "r"(taddr + (unsigned int)row_base_d + (unsigned int)(128 + blk * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float vn[16];
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        float dv = u_r[j] - wh_r[j];
                        if (t_row_d >= clen_d) {
                            dv = 0.0f;
                        }
                        vn[j] = dv;
                    }
                    unsigned int vn_t[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(vn_t[_lp]) : "f"(vn[_lp + 0]));
                    }
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vnt_k_addr + (unsigned int)(blk * 8192 + t_row_d * 128 + ((half_d * 4 + q ^ t_row_d & 7) << 4))), "r"(vn_t[q * 4]), "r"(vn_t[q * 4 + 1]), "r"(vn_t[q * 4 + 2]), "r"(vn_t[q * 4 + 3]) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_empty_addr);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(t32_full_addr);
                _phase_acc_full ^= 1;
                _phase_t32_empty ^= 1;
                if (valid == 0) {
                    break;
                }
                it_d = (int)cta;
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: epi ----
    if (warp >= 4 && warp <= 7) {
        { // epi_main
            const int wq_e = warp & 3;
            const int t_row_e = wq_e * 16 + (lane & 15);
            const int half_e = lane >> 4;
            const int row_base_e = wq_e * 32 << 16;
            unsigned int epi_buf = 0;
            unsigned int work_stage_e = 0;
            int num_items_e = num_chunks * num_heads;
            int it_e = (int)bid;
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_d_full = 0;
            #pragma unroll 1
            for (int _item_e = 0; _item_e < num_items_e; _item_e++) {
                mbarrier_wait(work_full_addr + (work_stage_e) * 8, _phase_work_full_1);
                uint32_t valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(valid_1)
                    : "r"(work_resp_addr + work_stage_e * 16 + 0 * 16)
                    : "memory");
                uint32_t cta_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(cta_1)
                    : "r"(work_resp_addr + work_stage_e * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_e) * 8);
                work_stage_e += 1;
                if (work_stage_e == 4) { work_stage_e = 0; _phase_work_full_1 ^= 1; }
                int chunk_e = it_e / num_heads;
                int head_e = it_e - chunk_e * num_heads;
                int dA_base = ((chunk_e * 64 + t_row_e) * num_heads + head_e) * 64;
                int buf_col = 256 + (int)epi_buf * 64;
                mbarrier_wait(d_full_addr + (epi_buf) * 8, _phase_d_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float acc0[16];
                float acc1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc0[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc0[15]))
                    : "r"(taddr + (unsigned int)row_base_e + (unsigned int)buf_col));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc1[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc1[15]))
                    : "r"(taddr + (unsigned int)row_base_e + (unsigned int)(buf_col + 32)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(d_empty_addr + (epi_buf) * 8);
                epi_buf += 1;
                if (epi_buf == 2) { epi_buf = 0; _phase_d_full ^= 1; }
                int s0 = half_e * 16;
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    float out8[8];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        float val = acc0[half * 8 + j_1] * scale;
                        if (s0 + half * 8 + j_1 > t_row_e) {
                            val = 0.0f;
                        }
                        out8[j_1] = val;
                    }
                    {
                        unsigned _stv8_0_0 = __float_as_uint(out8[0 + 0]);
                        unsigned _stv8_0_1 = __float_as_uint(out8[0 + 1]);
                        unsigned _stv8_0_2 = __float_as_uint(out8[0 + 2]);
                        unsigned _stv8_0_3 = __float_as_uint(out8[0 + 3]);
                        unsigned _stv8_0_4 = __float_as_uint(out8[0 + 4]);
                        unsigned _stv8_0_5 = __float_as_uint(out8[0 + 5]);
                        unsigned _stv8_0_6 = __float_as_uint(out8[0 + 6]);
                        unsigned _stv8_0_7 = __float_as_uint(out8[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dAqk + (dA_base + s0 + half * 8) + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                    }
                }
                int s0_0 = 32 + half_e * 16;
                #pragma unroll
                for (int half_1 = 0; half_1 < 2; half_1++) {
                    float out8_1[8];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        float val_1 = acc1[half_1 * 8 + j_2] * scale;
                        if (s0_0 + half_1 * 8 + j_2 > t_row_e) {
                            val_1 = 0.0f;
                        }
                        out8_1[j_2] = val_1;
                    }
                    {
                        unsigned _stv8_1_0 = __float_as_uint(out8_1[0 + 0]);
                        unsigned _stv8_1_1 = __float_as_uint(out8_1[0 + 1]);
                        unsigned _stv8_1_2 = __float_as_uint(out8_1[0 + 2]);
                        unsigned _stv8_1_3 = __float_as_uint(out8_1[0 + 3]);
                        unsigned _stv8_1_4 = __float_as_uint(out8_1[0 + 4]);
                        unsigned _stv8_1_5 = __float_as_uint(out8_1[0 + 5]);
                        unsigned _stv8_1_6 = __float_as_uint(out8_1[0 + 6]);
                        unsigned _stv8_1_7 = __float_as_uint(out8_1[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dAqk + (dA_base + s0_0 + half_1 * 8) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
                    }
                }
                if (valid_1 == 0) {
                    break;
                }
                it_e = (int)cta_1;
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: cvt ----
    if (warp >= 8 && warp <= 11) {
        { // cvt_main
            const int ct = (warp - 8) * 32 + lane;
            const int cq = ct & 7;
            const int cr = ct >> 3;
            const int swz_a = cq ^ cr & 7;
            const int drow = ct >> 1;
            const int dhalf = ct & 1;
            unsigned int cvt_stage = 0;
            unsigned int work_stage_c = 0;
            int num_items_c = num_chunks * num_heads;
            unsigned int zeros[4];
            zeros[0] = 0;
            zeros[1] = 0;
            zeros[2] = 0;
            zeros[3] = 0;
            float bt_p[8];
            int it_c = (int)bid;
            int chunk_f = it_c / num_heads;
            int head_f = it_c - chunk_f * num_heads;
            int bos_f = chunk_bos[chunk_f];
            int clen_f = chunk_len[chunk_f];
            int clast_f = clen_f - 1;
            if (clast_f < 0) {
                clast_f = 0;
            }
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                int s_f = cq * 8 + e;
                if (s_f > clast_f) {
                    s_f = clast_f;
                }
                long long bidx = (long long)(bos_f + s_f) * (long long)num_heads + (long long)head_f;
                bt_p[e] = (float)beta[bidx];
            }
            int clen_p = clen_f;
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_t32_empty_1 = 1;
            #pragma unroll 1
            for (int _item_c = 0; _item_c < num_items_c; _item_c++) {
                float bt[8];
                #pragma unroll
                for (int e_1 = 0; e_1 < 8; e_1++) {
                    bt[e_1] = bt_p[e_1];
                }
                int clen_c = clen_p;
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_2);
                uint32_t valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(valid_2)
                    : "r"(work_resp_addr + work_stage_c * 16 + 0 * 16)
                    : "memory");
                uint32_t cta_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(cta_2)
                    : "r"(work_resp_addr + work_stage_c * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_2 ^= 1; }
                int it_n = it_c;
                if (valid_2 != 0) {
                    it_n = (int)cta_2;
                }
                int chunk_f_0 = it_n / num_heads;
                int head_f_0 = it_n - chunk_f_0 * num_heads;
                int bos_f_0 = chunk_bos[chunk_f_0];
                int clen_f_0 = chunk_len[chunk_f_0];
                int clast_f_0 = clen_f_0 - 1;
                if (clast_f_0 < 0) {
                    clast_f_0 = 0;
                }
                #pragma unroll
                for (int e_2 = 0; e_2 < 8; e_2++) {
                    int s_f_1 = cq * 8 + e_2;
                    if (s_f_1 > clast_f_0) {
                        s_f_1 = clast_f_0;
                    }
                    long long bidx_1 = (long long)(bos_f_0 + s_f_1) * (long long)num_heads + (long long)head_f_0;
                    bt_p[e_2] = (float)beta[bidx_1];
                }
                clen_p = clen_f_0;
                mbarrier_wait(tiles_full_addr + (cvt_stage) * 8, _phase_tiles_full);
                unsigned int stage_v = v_mn_addr + cvt_stage * 57344;
                int npad = (64 - clen_c) * 16;
                #pragma unroll 1
                for (int idx = ct; idx < npad; idx += 128) {
                    int prow = clen_c + (idx >> 4);
                    int rem = idx & 15;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stage_v + (unsigned int)((rem >> 3) * 8192 + prow * 128 + (rem & 7) * 16)), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                }
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int t_a = cr + 16 * k;
                    int lim = clen_c;
                    if (t_a >= clen_c) {
                        lim = 0;
                    }
                    unsigned int a_addr = akk_k_addr + cvt_stage * 57344 + (unsigned int)(t_a * 128 + (swz_a << 4));
                    unsigned int raw[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3]))
                        : "r"(a_addr));
                    float akk_f[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&akk_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(raw[_pair]) << 16);
                        (&akk_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(raw[_pair]) & 0xffff0000u);
                    }
                    float prod[8];
                    #pragma unroll
                    for (int e_3 = 0; e_3 < 8; e_3++) {
                        float x = akk_f[e_3] * bt[e_3];
                        if (cq * 8 + e_3 >= lim) {
                            x = 0.0f;
                        }
                        prod[e_3] = x;
                    }
                    unsigned int packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod[_lp*2 + 0], prod[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(a_addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(ops_ready_addr + (cvt_stage) * 8);
                mbarrier_wait(t32_empty_addr, _phase_t32_empty_1);
                int drow_live = 0;
                if (drow < clen_c) {
                    drow_live = 1;
                }
                unsigned int dmask = (unsigned int)(-drow_live);
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    unsigned int d_addr = do_s_addr + cvt_stage * 57344 + (unsigned int)(dhalf * 8192 + drow * 128 + ((j_3 ^ drow & 7) << 4));
                    unsigned int draw[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&draw[0])), "=r"(*reinterpret_cast<uint32_t*>(&draw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&draw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&draw[(0) + 3]))
                        : "r"(d_addr));
                    unsigned int dsel[4];
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        dsel[i] = draw[i] & dmask;
                    }
                    float do_f[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&do_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dsel[_pair]) << 16);
                        (&do_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dsel[_pair]) & 0xffff0000u);
                    }
                    int slab_off = (dhalf * 2 + (j_3 >> 2)) * 8192 + drow * 128;
                    int c0 = (j_3 & 3) * 2;
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dot_k_addr + (unsigned int)(slab_off + ((c0 ^ drow & 7) << 4))), "f"(do_f[0]), "f"(do_f[1]), "f"(do_f[2]), "f"(do_f[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dot_k_addr + (unsigned int)(slab_off + ((c0 + 1 ^ drow & 7) << 4))), "f"(do_f[4]), "f"(do_f[5]), "f"(do_f[6]), "f"(do_f[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(t32_full_addr);
                cvt_stage += 1;
                if (cvt_stage == 2) { cvt_stage = 0; _phase_tiles_full ^= 1; }
                _phase_t32_empty_1 ^= 1;
                if (valid_2 == 0) {
                    break;
                }
                it_c = it_n;
            }
        }
    }
    // ---- Role: load ----
    if (warp == 12) {
        { // load_main
            unsigned int ld_stage = 0;
            unsigned int work_stage_l = 0;
            int num_items_l = num_chunks * num_heads;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_h_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            if (elect_sync()) {
                int it_l = (int)bid;
                int nbos = chunk_bos[it_l / num_heads];
                #pragma unroll 1
                for (int _item_l = 0; _item_l < num_items_l; _item_l++) {
                    mbarrier_wait(work_empty_addr + (work_stage_l) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_l) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_resp_addr + work_stage_l * 16 + 0 * 16), "r"(work_full_addr + work_stage_l * 8)
                        : "memory");
                    int chunk_l = it_l / num_heads;
                    int head_l = it_l - chunk_l * num_heads;
                    int row0_l = chunk_l * 64;
                    int bos_l = nbos;
                    int akk_row = row0_l;
                    int do_row = row0_l;
                    mbarrier_wait(tiles_empty_addr + (ld_stage) * 8, _phase_tiles_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(tiles_full_addr + (ld_stage) * 8, 57344);
                    tma_3d_gmem2smem(akk_k_addr + ld_stage * 57344, (&akk_tma), 0, head_l, akk_row, tiles_full_addr + (ld_stage) * 8);
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(v_mn_addr + ld_stage * 57344 + (unsigned int)(seg * 8192), (&v_tma), seg * 64, head_l, bos_l, tiles_full_addr + (ld_stage) * 8);
                        tma_3d_gmem2smem(do_s_addr + ld_stage * 57344 + (unsigned int)(seg * 8192), (&do_tma), seg * 64, head_l, do_row, tiles_full_addr + (ld_stage) * 8);
                        tma_3d_gmem2smem(w_k_addr + ld_stage * 57344 + (unsigned int)(seg * 8192), (&w_tma), seg * 64, head_l, row0_l, tiles_full_addr + (ld_stage) * 8);
                    }
                    ld_stage += 1;
                    if (ld_stage == 2) { ld_stage = 0; _phase_tiles_empty ^= 1; }
                    mbarrier_wait(h_empty_addr, _phase_h_empty);
                    mbarrier_arrive_expect_tx(h_full_addr, 32768);
                    int hrow0 = (chunk_l * num_heads + head_l) * 128;
                    #pragma unroll
                    for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                        tma_2d_gmem2smem(h_p_addr + (unsigned int)(seg_1 * 16384), (&h_tma), seg_1 * 64, hrow0, h_full_addr);
                    }
                    _phase_h_empty ^= 1;
                    mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_3);
                    uint32_t valid_3 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(valid_3)
                        : "r"(work_resp_addr + work_stage_l * 16 + 0 * 16)
                        : "memory");
                    uint32_t cta_3 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(cta_3)
                        : "r"(work_resp_addr + work_stage_l * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                    work_stage_l += 1;
                    if (work_stage_l == 4) { work_stage_l = 0; _phase_work_empty ^= 1; _phase_work_full_3 ^= 1; }
                    if (valid_3 == 0) {
                        break;
                    }
                    it_l = (int)cta_3;
                    nbos = chunk_bos[it_l / num_heads];
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int d_buf = 0;
            unsigned int work_stage_m = 0;
            int num_items_m = num_chunks * num_heads;
            unsigned int _phase_work_full_4 = 0;
            unsigned int _phase_tiles_full_1 = 0;
            unsigned int _phase_h_full = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_t32_full = 0;
            unsigned int _phase_d_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int _item_m = 0; _item_m < num_items_m; _item_m++) {
                    mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_4);
                    uint32_t valid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(valid_4)
                        : "r"(work_resp_addr + work_stage_m * 16 + 0 * 16)
                        : "memory");
                    uint32_t cta_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(cta_4)
                        : "r"(work_resp_addr + work_stage_m * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                    work_stage_m += 1;
                    if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_4 ^= 1; }
                    mbarrier_wait(tiles_full_addr + (mma_stage) * 8, _phase_tiles_full_1);
                    mbarrier_wait(ops_ready_addr + (mma_stage) * 8, _phase_tiles_full_1);
                    mbarrier_wait(h_full_addr, _phase_h_full);
                    mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((akk_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 3584;
                    int _smem_desc_lo_1 = ((((v_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_stage) * 3584;
#else
                    int _smem_desc_lo_0 = (((akk_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 3584;
                    int _smem_desc_lo_1 = ((((v_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 3584;
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_u), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((w_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 3584;
                    int _smem_desc_lo_3 = (((h_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_2 = (((w_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 3584;
                    int _smem_desc_lo_3 = (((h_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_wh), "r"(0) : "memory");
                    tcgen05_commit(h_empty_addr);
                    tcgen05_commit(acc_full_addr);
                    _phase_h_full ^= 1;
                    _phase_acc_empty ^= 1;
                    mbarrier_wait(t32_full_addr, _phase_t32_full);
                    mbarrier_wait(d_empty_addr + (d_buf) * 8, _phase_d_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((dot_k_addr) >> 4) & 0x7FFF) + (0) * 512;
                    int _smem_desc_lo_5 = (((vnt_k_addr) >> 4) & 0x7FFF) + (0) * 512;
#else
                    int _smem_desc_lo_4 = (((dot_k_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _smem_desc_lo_5 = (((vnt_k_addr) >> 4) & 0x3FFF) + (0) * 512;
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
                    "mov.b32 id, 68159760;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"((tmem_d + (d_buf * 64))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = (((dot_k_addr) >> 4) & 0x7FFF) + (1) * 512;
                    int _smem_desc_lo_7 = (((vnt_k_addr) >> 4) & 0x7FFF) + (1) * 512;
#else
                    int _smem_desc_lo_6 = (((dot_k_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _smem_desc_lo_7 = (((vnt_k_addr) >> 4) & 0x3FFF) + (1) * 512;
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
                    "mov.b32 id, 68159760;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"((tmem_d + (d_buf * 64))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = (((dot_k_addr) >> 4) & 0x7FFF) + (2) * 512;
                    int _smem_desc_lo_9 = (((vnt_k_addr) >> 4) & 0x7FFF) + (2) * 512;
#else
                    int _smem_desc_lo_8 = (((dot_k_addr) >> 4) & 0x3FFF) + (2) * 512;
                    int _smem_desc_lo_9 = (((vnt_k_addr) >> 4) & 0x3FFF) + (2) * 512;
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
                    "mov.b32 id, 68159760;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"((tmem_d + (d_buf * 64))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = (((dot_k_addr) >> 4) & 0x7FFF) + (3) * 512;
                    int _smem_desc_lo_11 = (((vnt_k_addr) >> 4) & 0x7FFF) + (3) * 512;
#else
                    int _smem_desc_lo_10 = (((dot_k_addr) >> 4) & 0x3FFF) + (3) * 512;
                    int _smem_desc_lo_11 = (((vnt_k_addr) >> 4) & 0x3FFF) + (3) * 512;
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
                    "mov.b32 id, 68159760;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"((tmem_d + (d_buf * 64))), "r"(1) : "memory");
                    tcgen05_commit(tiles_empty_addr + (mma_stage) * 8);
                    tcgen05_commit(t32_empty_addr);
                    tcgen05_commit(d_full_addr + (d_buf) * 8);
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_tiles_full_1 ^= 1; }
                    _phase_t32_full ^= 1;
                    d_buf += 1;
                    if (d_buf == 2) { d_buf = 0; _phase_d_empty ^= 1; }
                    if (valid_4 == 0) {
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
