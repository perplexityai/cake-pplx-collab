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

#define LAUNCH_MIN_BLOCKS 5

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





__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


extern "C" __device__ float __nv_expf(float);



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



__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_attn_res_train_e8783a6a545412badc06(__nv_bfloat16* __restrict__ v0, __nv_bfloat16* __restrict__ v1, __nv_bfloat16* __restrict__ v2, __nv_bfloat16* __restrict__ v3, __nv_bfloat16* __restrict__ v4, __nv_bfloat16* __restrict__ v5, __nv_bfloat16* __restrict__ v6, __nv_bfloat16* __restrict__ v7, __nv_bfloat16* __restrict__ v8, float* __restrict__ w, __nv_bfloat16* __restrict__ out, float* __restrict__ p, float* __restrict__ r, float* __restrict__ d, int T, int n, float eps)
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
    #define xbar_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* wtab = reinterpret_cast<float*>(smem_raw + 1024);
    const int wtab_addr = smem + 1024;
    float* recv = reinterpret_cast<float*>(smem_raw + 1472);
    const int recv_addr = smem + 1472;
    unsigned int* stg = reinterpret_cast<unsigned int*>(smem_raw + 1728);
    const int stage_addr = smem + 1728;

    // Mbarrier init (2 barriers at smem_raw[0..16))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xbar: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // === Task calls (declaration order) ===
    int rank = cta_rank;
    int col = rank * 1792 + tid * 8;
    int first = (int)cluster_id;
    int step = (int)num_clusters;
    int jl = ((lane) < (3) ? (lane) : (3));
    int act = ((lane < n) ? 1 : 0);
    float wf[8];
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        {
            unsigned _v4_0_0;
            unsigned _v4_0_1;
            unsigned _v4_0_2;
            unsigned _v4_0_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(w + (col + 4 * half) + (0))) : "memory");
            wf[4 * half + 0] = __uint_as_float(_v4_0_0);
            wf[4 * half + 1] = __uint_as_float(_v4_0_1);
            wf[4 * half + 2] = __uint_as_float(_v4_0_2);
            wf[4 * half + 3] = __uint_as_float(_v4_0_3);
        }
    }
    unsigned int vw[16];
    float part[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        part[i] = 0.0f;
    }
    int xk = 0;
    if (0 < n) {
        int _min_1 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(v0 + ((long long)_min_1 * 7168ll + (long long)col) + 0);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&vw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
    }
    if (1 < n) {
        int _min_3 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(v1 + ((long long)_min_3 * 7168ll + (long long)col) + 0);
            uint4* _vdst_4 = reinterpret_cast<uint4*>(&vw[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_4[_blk] = _vptr_4[_blk];
            }
        }
    }
    if (2 < n) {
        int _min_5 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(v2 + ((long long)_min_5 * 7168ll + (long long)col) + 0);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&vw[8]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_6[_blk] = _vptr_6[_blk];
            }
        }
    }
    if (3 < n) {
        int _min_7 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(v3 + ((long long)_min_7 * 7168ll + (long long)col) + 0);
            uint4* _vdst_8 = reinterpret_cast<uint4*>(&vw[12]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_8[_blk] = _vptr_8[_blk];
            }
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    #pragma unroll 1
    for (int t = first; t < T; t += step) {
        part[0] = 0.0f;
        part[4] = 0.0f;
        if (0 < n) {
            float ts_s_0 = 0.0f;
            #pragma unroll
            for (int m_0 = 0; m_0 < 4; m_0++) {
                float _dot_9;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_9) : "r"(vw[m_0]), "r"(vw[m_0]), "f"(ts_s_0));
                ts_s_0 = _dot_9;
            }
            float2 ts_f_0 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_1 = 0; m_1 < 4; m_1++) {
                ts_f_0 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_1], wf[2 * m_1 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_1])), ts_f_0);
            }
            part[0] = ts_s_0;
            float _add_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(ts_f_0.x), "f"(ts_f_0.y));
            part[4] = _add_10;
        }
        part[1] = 0.0f;
        part[5] = 0.0f;
        if (1 < n) {
            float ts_s_1 = 0.0f;
            #pragma unroll
            for (int m_2 = 0; m_2 < 4; m_2++) {
                float _dot_11;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_11) : "r"(vw[4 + m_2]), "r"(vw[4 + m_2]), "f"(ts_s_1));
                ts_s_1 = _dot_11;
            }
            float2 ts_f_1 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_3 = 0; m_3 < 4; m_3++) {
                ts_f_1 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_3], wf[2 * m_3 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_3])), ts_f_1);
            }
            part[1] = ts_s_1;
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(ts_f_1.x), "f"(ts_f_1.y));
            part[5] = _add_12;
        }
        part[2] = 0.0f;
        part[6] = 0.0f;
        if (2 < n) {
            float ts_s_2 = 0.0f;
            #pragma unroll
            for (int m_4 = 0; m_4 < 4; m_4++) {
                float _dot_13;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_13) : "r"(vw[8 + m_4]), "r"(vw[8 + m_4]), "f"(ts_s_2));
                ts_s_2 = _dot_13;
            }
            float2 ts_f_2 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_5 = 0; m_5 < 4; m_5++) {
                ts_f_2 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_5], wf[2 * m_5 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[8 + m_5])), ts_f_2);
            }
            part[2] = ts_s_2;
            float _add_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(ts_f_2.x), "f"(ts_f_2.y));
            part[6] = _add_14;
        }
        part[3] = 0.0f;
        part[7] = 0.0f;
        if (3 < n) {
            float ts_s_3 = 0.0f;
            #pragma unroll
            for (int m_6 = 0; m_6 < 4; m_6++) {
                float _dot_15;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_15) : "r"(vw[12 + m_6]), "r"(vw[12 + m_6]), "f"(ts_s_3));
                ts_s_3 = _dot_15;
            }
            float2 ts_f_3 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_7 = 0; m_7 < 4; m_7++) {
                ts_f_3 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_7], wf[2 * m_7 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[12 + m_7])), ts_f_3);
            }
            part[3] = ts_s_3;
            float _add_16;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(ts_f_3.x), "f"(ts_f_3.y));
            part[7] = _add_16;
        }
        if (t + step < T) {
            if (0 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * tid)), "l"(v0 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            if (1 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (224 + tid))), "l"(v1 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            if (2 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (448 + tid))), "l"(v2 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            if (3 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (672 + tid))), "l"(v3 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            asm volatile("cp.async.commit_group;");
        }
        int xp_slot = xk % 2;
        int xp_row = xp_slot * 56 + warp * 8;
        #pragma unroll
        for (int i_0 = 0; i_0 < 4; i_0++) {
            int ws_upper_0 = lane & 16;
            float ws_keep_0 = ((ws_upper_0 != 0) ? part[4 + i_0] : part[i_0]);
            float ws_send_0 = ((ws_upper_0 != 0) ? part[i_0] : part[4 + i_0]);
            float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
            part[i_0] = ws_keep_0 + _shfl_xor_sync_17;
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; i_1++) {
            int ws_upper_1 = lane & 8;
            float ws_keep_1 = ((ws_upper_1 != 0) ? part[2 + i_1] : part[i_1]);
            float ws_send_1 = ((ws_upper_1 != 0) ? part[i_1] : part[2 + i_1]);
            float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
            part[i_1] = ws_keep_1 + _shfl_xor_sync_18;
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 1; i_2++) {
            int ws_upper_2 = lane & 4;
            float ws_keep_2 = ((ws_upper_2 != 0) ? part[1 + i_2] : part[i_2]);
            float ws_send_2 = ((ws_upper_2 != 0) ? part[i_2] : part[1 + i_2]);
            float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
            part[i_2] = ws_keep_2 + _shfl_xor_sync_19;
        }
        float ws_own = part[0];
        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 2);
        ws_own = ws_own + _shfl_xor_sync_20;
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
        ws_own = ws_own + _shfl_xor_sync_21;
        float xp_word = ws_own;
        if ((lane & 3) == 0) {
            if (lane >> 2 < 8) {
                wtab[xp_row + (lane >> 2)] = xp_word;
            }
        }
        __syncthreads();
        if (warp == 0) {
            if (lane < 8) {
                int xp_job = lane % 2;
                int xp_dst = lane / 2;
                int xp_base = xp_slot * 56 + 4 * xp_job;
                unsigned int xp_to = recv_addr + (unsigned int)(4 * (xp_slot * 32 + rank * 8));
                float xp_s[4];
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    xp_s[c] = wtab[xp_base + c];
                }
                #pragma unroll
                for (int wi = 1; wi < 7; wi++) {
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 4; c_1++) {
                        xp_s[c_1] = xp_s[c_1] + wtab[xp_base + wi * 8 + c_1];
                    }
                }
                uint32_t _cluster_mapa_22;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_22) : "r"(xp_to + (unsigned int)(16 * xp_job)), "r"(xp_dst));
                uint32_t _mbar_peer_23;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_23) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_22), "r"(__float_as_uint(xp_s[0])), "r"(__float_as_uint(xp_s[1])), "r"(__float_as_uint(xp_s[2])), "r"(__float_as_uint(xp_s[3])), "r"(_mbar_peer_23) : "memory");
            }
        }
        if (tid == 0) {
            mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 128);
        }
        int _stage_xbar = (int)((unsigned int)xk % 2);
        int _parity_xbar = (int)((unsigned int)xk / 2 % 2);
        asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(xbar_addr + (_stage_xbar) * 8), "r"(uint32_t(_parity_xbar)), "r"(uint32_t(10000000)) : "memory");
        asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
        int xt_at = xk % 2 * 32 + jl;
        float xt_t = recv[xt_at];
        #pragma unroll
        for (int rk = 1; rk < 4; rk++) {
            xt_t = xt_t + recv[xt_at + rk * 8];
        }
        float tk_ss = xt_t;
        int xt_at_0 = xk % 2 * 32 + (4 + jl);
        float xt_t_0 = recv[xt_at_0];
        #pragma unroll
        for (int rk_1 = 1; rk_1 < 4; rk_1++) {
            xt_t_0 = xt_t_0 + recv[xt_at_0 + rk_1 * 8];
        }
        float tk_d = xt_t_0;
        xk += 1;
        float _fdiv_25 = __fdiv_rn(tk_ss, 7168.0f);
        float _sqrt_24;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_24) : "f"(_fdiv_25 + eps));
        float tk_r = __frcp_rn(_sqrt_24);
        float tk_s = tk_r * tk_d;
        float tk_m = ((act != 0) ? tk_s : -3.4e+38f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tk_m = max_noftz(tk_m, __shfl_xor_sync(0xFFFFFFFF, tk_m, offset));
        float _exp_26 = __nv_expf(tk_s - tk_m);
        float tk_e = ((act != 0) ? _exp_26 : 0.0f);
        float tk_z = tk_e;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tk_z += __shfl_xor_sync(0xFFFFFFFF, tk_z, offset);
        float tk_p = __fdiv_rn(tk_e, tk_z);
        if (rank == 0) {
            if (warp == 0) {
                if (act != 0) {
                    long long tk_at = (long long)t * (long long)n + (long long)lane;
                    p[tk_at] = tk_p;
                    r[tk_at] = tk_r;
                    d[tk_at] = tk_d;
                }
            }
        }
        float to_o[8];
        #pragma unroll
        for (int i_3 = 0; i_3 < 8; i_3++) {
            to_o[i_3] = 0.0f;
        }
        if (0 < n) {
            float tj_p_0 = __shfl_sync(0xFFFFFFFF, tk_p, 0);
            float2 tj_p2_0 = make_float2(tj_p_0, tj_p_0);
            #pragma unroll
            for (int m_8 = 0; m_8 < 4; m_8++) {
                float2 tj_o_0 = fma_f32x2_rn_noftz(tj_p2_0, cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_8])), make_float2(to_o[2 * m_8], to_o[2 * m_8 + 1]));
                to_o[2 * m_8] = tj_o_0.x;
                to_o[2 * m_8 + 1] = tj_o_0.y;
            }
        }
        if (1 < n) {
            float tj_p_1 = __shfl_sync(0xFFFFFFFF, tk_p, 1);
            float2 tj_p2_1 = make_float2(tj_p_1, tj_p_1);
            #pragma unroll
            for (int m_9 = 0; m_9 < 4; m_9++) {
                float2 tj_o_1 = fma_f32x2_rn_noftz(tj_p2_1, cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_9])), make_float2(to_o[2 * m_9], to_o[2 * m_9 + 1]));
                to_o[2 * m_9] = tj_o_1.x;
                to_o[2 * m_9 + 1] = tj_o_1.y;
            }
        }
        if (2 < n) {
            float tj_p_2 = __shfl_sync(0xFFFFFFFF, tk_p, 2);
            float2 tj_p2_2 = make_float2(tj_p_2, tj_p_2);
            #pragma unroll
            for (int m_10 = 0; m_10 < 4; m_10++) {
                float2 tj_o_2 = fma_f32x2_rn_noftz(tj_p2_2, cake_bf16x2_to_f32x2(__as_bf16x2(vw[8 + m_10])), make_float2(to_o[2 * m_10], to_o[2 * m_10 + 1]));
                to_o[2 * m_10] = tj_o_2.x;
                to_o[2 * m_10 + 1] = tj_o_2.y;
            }
        }
        if (3 < n) {
            float tj_p_3 = __shfl_sync(0xFFFFFFFF, tk_p, 3);
            float2 tj_p2_3 = make_float2(tj_p_3, tj_p_3);
            #pragma unroll
            for (int m_11 = 0; m_11 < 4; m_11++) {
                float2 tj_o_3 = fma_f32x2_rn_noftz(tj_p2_3, cake_bf16x2_to_f32x2(__as_bf16x2(vw[12 + m_11])), make_float2(to_o[2 * m_11], to_o[2 * m_11 + 1]));
                to_o[2 * m_11] = tj_o_3.x;
                to_o[2 * m_11 + 1] = tj_o_3.y;
            }
        }
        unsigned int ss_packed[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_o[_lp*2 + 0], to_o[_lp*2+1 + 0]));
            ss_packed[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + ((long long)t * 7168ll + (long long)col))[0] = reinterpret_cast<int4*>(ss_packed)[0];
        if (t + step < T) {
            asm volatile("cp.async.wait_group 0;");
            if (0 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * tid)));
            }
            if (1 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[4])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (224 + tid))));
            }
            if (2 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[8])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (448 + tid))));
            }
            if (3 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[12])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (672 + tid))));
            }
        }
    }
}

} // extern "C"
