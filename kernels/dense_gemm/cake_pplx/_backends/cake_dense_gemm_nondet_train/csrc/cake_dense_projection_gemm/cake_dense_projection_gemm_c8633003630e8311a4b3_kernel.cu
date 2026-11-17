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

#include <math_constants.h>

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

extern "C" {

__global__ __launch_bounds__(256) __cluster_dims__(8,1,1) void
kernel_cake_dense_projection_gemm_c8633003630e8311a4b3(float* __restrict__ X, float* __restrict__ W, float* __restrict__ out, int T, int N, int K, int ldx, int ldw, int ldo)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* smem_1 = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int rank = blockIdx.x % 8;
    int e0 = blockIdx.x / 8 * 16;
    int t0 = blockIdx.y * 32;
    int seg = K / 8;
    int k0 = rank * seg;
    int chunks = seg / 128;
    int j = lane % 16;
    int ge = tid / 16 % 4;
    int gt = tid / 16 / 4;
    float2 acc[16];
    #pragma unroll
    for (int o = 0; o < 16; o++) {
        acc[o] = make_float2(0.0f, 0.0f);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (0 < chunks) {
        unsigned int fill = smem_addr + 0;
        int col0 = k0;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int wv_ = i * 256 + tid;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(fill + (unsigned int)((wv_ / 32 * 144 + wv_ % 32 * 4) * 4)), "l"(W + ((long long)(e0 + wv_ / 32) * (long long)ldw + (long long)(col0 + wv_ % 32 * 4))));
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; i_1++) {
            int xv_ = i_1 * 256 + tid;
            int tok = t0 + xv_ / 32;
            int _min_0 = ((tok) < (T - 1) ? (tok) : (T - 1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(fill + (unsigned int)(((16 + xv_ / 32) * 144 + xv_ % 32 * 4) * 4)), "l"(X + ((long long)_min_0 * (long long)ldx + (long long)(col0 + xv_ % 32 * 4))), "r"((tok < T) ? 16 : 0));
        }
    }
    asm volatile("cp.async.commit_group;");
    for (int c = 0; c < chunks; c++) {
        if (c + 2 - 1 < chunks) {
            unsigned int fill_1 = smem_addr + (unsigned int)((c + 2 - 1) % 2 * 27648);
            int col0_1 = k0 + (c + 2 - 1) * 128;
            #pragma unroll
            for (int i_2 = 0; i_2 < 2; i_2++) {
                int wv__1 = i_2 * 256 + tid;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(fill_1 + (unsigned int)((wv__1 / 32 * 144 + wv__1 % 32 * 4) * 4)), "l"(W + ((long long)(e0 + wv__1 / 32) * (long long)ldw + (long long)(col0_1 + wv__1 % 32 * 4))));
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 4; i_3++) {
                int xv__1 = i_3 * 256 + tid;
                int tok_1 = t0 + xv__1 / 32;
                int _min_1 = ((tok_1) < (T - 1) ? (tok_1) : (T - 1));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill_1 + (unsigned int)(((16 + xv__1 / 32) * 144 + xv__1 % 32 * 4) * 4)), "l"(X + ((long long)_min_1 * (long long)ldx + (long long)(col0_1 + xv__1 % 32 * 4))), "r"((tok_1 < T) ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        unsigned int cur = smem_addr + (unsigned int)(c % 2 * 27648);
        float wv[4];
        float xv[8];
        float yv[1];
        #pragma unroll
        for (int g4 = 0; g4 < 8; g4++) {
            int col_0 = j + 16 * g4;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])) : "r"(cur + (unsigned int)((ge * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[1])) : "r"(cur + (unsigned int)(((ge + 4) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[2])) : "r"(cur + (unsigned int)(((ge + 8) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[3])) : "r"(cur + (unsigned int)(((ge + 12) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])) : "r"(cur + (unsigned int)(((16 + gt) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[1])) : "r"(cur + (unsigned int)(((16 + gt + 4) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[2])) : "r"(cur + (unsigned int)(((16 + gt + 8) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[3])) : "r"(cur + (unsigned int)(((16 + gt + 12) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])) : "r"(cur + (unsigned int)(((16 + gt + 16) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[5])) : "r"(cur + (unsigned int)(((16 + gt + 20) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[6])) : "r"(cur + (unsigned int)(((16 + gt + 24) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[7])) : "r"(cur + (unsigned int)(((16 + gt + 28) * 144 + col_0) * 4)));
            acc[0] = fma_f32x2_rn_noftz(make_float2(xv[0], xv[1]), make_float2(wv[0], wv[0]), acc[0]);
            acc[1] = fma_f32x2_rn_noftz(make_float2(xv[2], xv[3]), make_float2(wv[0], wv[0]), acc[1]);
            acc[2] = fma_f32x2_rn_noftz(make_float2(xv[4], xv[5]), make_float2(wv[0], wv[0]), acc[2]);
            acc[3] = fma_f32x2_rn_noftz(make_float2(xv[6], xv[7]), make_float2(wv[0], wv[0]), acc[3]);
            acc[4] = fma_f32x2_rn_noftz(make_float2(xv[0], xv[1]), make_float2(wv[1], wv[1]), acc[4]);
            acc[5] = fma_f32x2_rn_noftz(make_float2(xv[2], xv[3]), make_float2(wv[1], wv[1]), acc[5]);
            acc[6] = fma_f32x2_rn_noftz(make_float2(xv[4], xv[5]), make_float2(wv[1], wv[1]), acc[6]);
            acc[7] = fma_f32x2_rn_noftz(make_float2(xv[6], xv[7]), make_float2(wv[1], wv[1]), acc[7]);
            acc[8] = fma_f32x2_rn_noftz(make_float2(xv[0], xv[1]), make_float2(wv[2], wv[2]), acc[8]);
            acc[9] = fma_f32x2_rn_noftz(make_float2(xv[2], xv[3]), make_float2(wv[2], wv[2]), acc[9]);
            acc[10] = fma_f32x2_rn_noftz(make_float2(xv[4], xv[5]), make_float2(wv[2], wv[2]), acc[10]);
            acc[11] = fma_f32x2_rn_noftz(make_float2(xv[6], xv[7]), make_float2(wv[2], wv[2]), acc[11]);
            acc[12] = fma_f32x2_rn_noftz(make_float2(xv[0], xv[1]), make_float2(wv[3], wv[3]), acc[12]);
            acc[13] = fma_f32x2_rn_noftz(make_float2(xv[2], xv[3]), make_float2(wv[3], wv[3]), acc[13]);
            acc[14] = fma_f32x2_rn_noftz(make_float2(xv[4], xv[5]), make_float2(wv[3], wv[3]), acc[14]);
            acc[15] = fma_f32x2_rn_noftz(make_float2(xv[6], xv[7]), make_float2(wv[3], wv[3]), acc[15]);
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;");
    #pragma unroll
    for (int o_0 = 0; o_0 < 16; o_0++) {
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, acc[o_0].x, 1);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, acc[o_0].y, 1);
        acc[o_0] = add_f32x2_noftz(acc[o_0], make_float2(_shfl_xor_sync_2, _shfl_xor_sync_3));
    }
    #pragma unroll
    for (int o_1 = 0; o_1 < 16; o_1++) {
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, acc[o_1].x, 2);
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, acc[o_1].y, 2);
        acc[o_1] = add_f32x2_noftz(acc[o_1], make_float2(_shfl_xor_sync_4, _shfl_xor_sync_5));
    }
    #pragma unroll
    for (int o_2 = 0; o_2 < 16; o_2++) {
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, acc[o_2].x, 4);
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, acc[o_2].y, 4);
        acc[o_2] = add_f32x2_noftz(acc[o_2], make_float2(_shfl_xor_sync_6, _shfl_xor_sync_7));
    }
    #pragma unroll
    for (int o_3 = 0; o_3 < 16; o_3++) {
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, acc[o_3].x, 8);
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, acc[o_3].y, 8);
        acc[o_3] = add_f32x2_noftz(acc[o_3], make_float2(_shfl_xor_sync_8, _shfl_xor_sync_9));
    }
    if (j == 0) {
        int red_o_0 = ge * 32 + gt;
        smem_1[red_o_0] = acc[0].x;
    }
    if (j == 1) {
        int red_o_1 = ge * 32 + gt + 4;
        smem_1[red_o_1] = acc[0].y;
    }
    if (j == 2) {
        int red_o_2 = ge * 32 + gt + 8;
        smem_1[red_o_2] = acc[1].x;
    }
    if (j == 3) {
        int red_o_3 = ge * 32 + gt + 12;
        smem_1[red_o_3] = acc[1].y;
    }
    if (j == 4) {
        int red_o_4 = ge * 32 + gt + 16;
        smem_1[red_o_4] = acc[2].x;
    }
    if (j == 5) {
        int red_o_5 = ge * 32 + gt + 20;
        smem_1[red_o_5] = acc[2].y;
    }
    if (j == 6) {
        int red_o_6 = ge * 32 + gt + 24;
        smem_1[red_o_6] = acc[3].x;
    }
    if (j == 7) {
        int red_o_7 = ge * 32 + gt + 28;
        smem_1[red_o_7] = acc[3].y;
    }
    if (j == 8) {
        int red_o_8 = (ge + 4) * 32 + gt;
        smem_1[red_o_8] = acc[4].x;
    }
    if (j == 9) {
        int red_o_9 = (ge + 4) * 32 + gt + 4;
        smem_1[red_o_9] = acc[4].y;
    }
    if (j == 10) {
        int red_o_10 = (ge + 4) * 32 + gt + 8;
        smem_1[red_o_10] = acc[5].x;
    }
    if (j == 11) {
        int red_o_11 = (ge + 4) * 32 + gt + 12;
        smem_1[red_o_11] = acc[5].y;
    }
    if (j == 12) {
        int red_o_12 = (ge + 4) * 32 + gt + 16;
        smem_1[red_o_12] = acc[6].x;
    }
    if (j == 13) {
        int red_o_13 = (ge + 4) * 32 + gt + 20;
        smem_1[red_o_13] = acc[6].y;
    }
    if (j == 14) {
        int red_o_14 = (ge + 4) * 32 + gt + 24;
        smem_1[red_o_14] = acc[7].x;
    }
    if (j == 15) {
        int red_o_15 = (ge + 4) * 32 + gt + 28;
        smem_1[red_o_15] = acc[7].y;
    }
    if (j == 0) {
        int red_o_16 = (ge + 8) * 32 + gt;
        smem_1[red_o_16] = acc[8].x;
    }
    if (j == 1) {
        int red_o_17 = (ge + 8) * 32 + gt + 4;
        smem_1[red_o_17] = acc[8].y;
    }
    if (j == 2) {
        int red_o_18 = (ge + 8) * 32 + gt + 8;
        smem_1[red_o_18] = acc[9].x;
    }
    if (j == 3) {
        int red_o_19 = (ge + 8) * 32 + gt + 12;
        smem_1[red_o_19] = acc[9].y;
    }
    if (j == 4) {
        int red_o_20 = (ge + 8) * 32 + gt + 16;
        smem_1[red_o_20] = acc[10].x;
    }
    if (j == 5) {
        int red_o_21 = (ge + 8) * 32 + gt + 20;
        smem_1[red_o_21] = acc[10].y;
    }
    if (j == 6) {
        int red_o_22 = (ge + 8) * 32 + gt + 24;
        smem_1[red_o_22] = acc[11].x;
    }
    if (j == 7) {
        int red_o_23 = (ge + 8) * 32 + gt + 28;
        smem_1[red_o_23] = acc[11].y;
    }
    if (j == 8) {
        int red_o_24 = (ge + 12) * 32 + gt;
        smem_1[red_o_24] = acc[12].x;
    }
    if (j == 9) {
        int red_o_25 = (ge + 12) * 32 + gt + 4;
        smem_1[red_o_25] = acc[12].y;
    }
    if (j == 10) {
        int red_o_26 = (ge + 12) * 32 + gt + 8;
        smem_1[red_o_26] = acc[13].x;
    }
    if (j == 11) {
        int red_o_27 = (ge + 12) * 32 + gt + 12;
        smem_1[red_o_27] = acc[13].y;
    }
    if (j == 12) {
        int red_o_28 = (ge + 12) * 32 + gt + 16;
        smem_1[red_o_28] = acc[14].x;
    }
    if (j == 13) {
        int red_o_29 = (ge + 12) * 32 + gt + 20;
        smem_1[red_o_29] = acc[14].y;
    }
    if (j == 14) {
        int red_o_30 = (ge + 12) * 32 + gt + 24;
        smem_1[red_o_30] = acc[15].x;
    }
    if (j == 15) {
        int red_o_31 = (ge + 12) * 32 + gt + 28;
        smem_1[red_o_31] = acc[15].y;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    float peer[8];
    float res[1];
    int idx_0 = tid;
    if (idx_0 < 64) {
        int el_0 = rank * 2 + idx_0 % 2;
        int tl_0 = idx_0 / 2;
        unsigned int red_byte_0 = (unsigned int)((el_0 * 32 + tl_0) * 4);
        uint32_t _cluster_mapa_10;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_10) : "r"(smem_addr + red_byte_0), "r"(0));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[0])) : "r"(_cluster_mapa_10));
        uint32_t _cluster_mapa_11;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_11) : "r"(smem_addr + red_byte_0), "r"(1));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[1])) : "r"(_cluster_mapa_11));
        uint32_t _cluster_mapa_12;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_12) : "r"(smem_addr + red_byte_0), "r"(2));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[2])) : "r"(_cluster_mapa_12));
        uint32_t _cluster_mapa_13;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_13) : "r"(smem_addr + red_byte_0), "r"(3));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[3])) : "r"(_cluster_mapa_13));
        uint32_t _cluster_mapa_14;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_14) : "r"(smem_addr + red_byte_0), "r"(4));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[4])) : "r"(_cluster_mapa_14));
        uint32_t _cluster_mapa_15;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_15) : "r"(smem_addr + red_byte_0), "r"(5));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[5])) : "r"(_cluster_mapa_15));
        uint32_t _cluster_mapa_16;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_16) : "r"(smem_addr + red_byte_0), "r"(6));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[6])) : "r"(_cluster_mapa_16));
        uint32_t _cluster_mapa_17;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_17) : "r"(smem_addr + red_byte_0), "r"(7));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[7])) : "r"(_cluster_mapa_17));
        res[0] = peer[0];
        res[0] = res[0] + peer[1];
        res[0] = res[0] + peer[2];
        res[0] = res[0] + peer[3];
        res[0] = res[0] + peer[4];
        res[0] = res[0] + peer[5];
        res[0] = res[0] + peer[6];
        res[0] = res[0] + peer[7];
        if (t0 + tl_0 < T) {
            *(reinterpret_cast<float*>(out + ((long long)(t0 + tl_0) * (long long)ldo + (long long)(e0 + el_0))) + (0)) = res[0];
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

} // extern "C"
