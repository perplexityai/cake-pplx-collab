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

#define LAUNCH_MIN_BLOCKS 4

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
kernel_cake_attn_res_train_31b6129cc550ef82db2f(__nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ v0, __nv_bfloat16* __restrict__ v1, __nv_bfloat16* __restrict__ v2, __nv_bfloat16* __restrict__ v3, __nv_bfloat16* __restrict__ v4, __nv_bfloat16* __restrict__ v5, __nv_bfloat16* __restrict__ v6, __nv_bfloat16* __restrict__ v7, __nv_bfloat16* __restrict__ v8, float* __restrict__ w, float* __restrict__ p, float* __restrict__ r, float* __restrict__ d, __nv_bfloat16* __restrict__ dv0, __nv_bfloat16* __restrict__ dv1, __nv_bfloat16* __restrict__ dv2, __nv_bfloat16* __restrict__ dv3, __nv_bfloat16* __restrict__ dv4, __nv_bfloat16* __restrict__ dv5, __nv_bfloat16* __restrict__ dv6, __nv_bfloat16* __restrict__ dv7, __nv_bfloat16* __restrict__ dv8, float* __restrict__ dw, int T, int n)
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
    float* recv = reinterpret_cast<float*>(smem_raw + 1248);
    const int recv_addr = smem + 1248;
    unsigned int* stg = reinterpret_cast<unsigned int*>(smem_raw + 1376);
    const int stage_addr = smem + 1376;

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
    int jl = ((lane) < (1) ? (lane) : (1));
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
    unsigned int gw[4];
    unsigned int vw[8];
    float st[3];
    float part[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        part[i] = 0.0f;
    }
    float blk[8];
    float tot[8];
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; i_1++) {
        blk[i_1] = 0.0f;
        tot[i_1] = 0.0f;
    }
    int nblk = 0;
    int xk = 0;
    float sn[3];
    int _min_1 = ((first) < (T - 1) ? (first) : (T - 1));
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(dout + ((long long)_min_1 * 7168ll + (long long)col) + 0);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&gw[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_2[_blk] = _vptr_2[_blk];
        }
    }
    if (0 < n) {
        int _min_3 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(v0 + ((long long)_min_3 * 7168ll + (long long)col) + 0);
            uint4* _vdst_4 = reinterpret_cast<uint4*>(&vw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_4[_blk] = _vptr_4[_blk];
            }
        }
    }
    if (1 < n) {
        int _min_5 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(v1 + ((long long)_min_5 * 7168ll + (long long)col) + 0);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&vw[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_6[_blk] = _vptr_6[_blk];
            }
        }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 3; i_2++) {
        st[i_2] = 0.0f;
    }
    if (act != 0) {
        int _min_7 = ((first) < (T - 1) ? (first) : (T - 1));
        long long lt_at = (long long)_min_7 * (long long)n + (long long)jl;
        st[0] = p[lt_at];
        st[1] = r[lt_at];
        st[2] = d[lt_at];
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    #pragma unroll 1
    for (int t = first; t < T; t += step) {
        part[0] = 0.0f;
        if (0 < n) {
            float tp_s_0 = 0.0f;
            #pragma unroll
            for (int m_0 = 0; m_0 < 4; m_0++) {
                float _dot_8;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_8) : "r"(gw[m_0]), "r"(vw[m_0]), "f"(tp_s_0));
                tp_s_0 = _dot_8;
            }
            part[0] = tp_s_0;
        }
        part[1] = 0.0f;
        if (1 < n) {
            float tp_s_1 = 0.0f;
            #pragma unroll
            for (int m_1 = 0; m_1 < 4; m_1++) {
                float _dot_9;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_9) : "r"(gw[m_1]), "r"(vw[4 + m_1]), "f"(tp_s_1));
                tp_s_1 = _dot_9;
            }
            part[1] = tp_s_1;
        }
        if (t + step < T) {
            long long sg_row = (long long)(t + step) * 7168ll + (long long)col;
            if (0 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * tid)), "l"(v0 + sg_row));
            }
            if (1 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (224 + tid))), "l"(v1 + sg_row));
            }
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (448 + tid))), "l"(dout + sg_row));
            asm volatile("cp.async.commit_group;");
            #pragma unroll
            for (int i_3 = 0; i_3 < 3; i_3++) {
                sn[i_3] = 0.0f;
            }
            if (act != 0) {
                long long sg_at = (long long)(t + step) * (long long)n + (long long)jl;
                sn[0] = p[sg_at];
                sn[1] = r[sg_at];
                sn[2] = d[sg_at];
            }
        }
        int xp_slot = xk % 2;
        int xp_row = xp_slot * 28 + warp * 4;
        #pragma unroll
        for (int i_0 = 0; i_0 < 2; i_0++) {
            int ws_upper_0 = lane & 16;
            float ws_keep_0 = ((ws_upper_0 != 0) ? part[2 + i_0] : part[i_0]);
            float ws_send_0 = ((ws_upper_0 != 0) ? part[i_0] : part[2 + i_0]);
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
            part[i_0] = ws_keep_0 + _shfl_xor_sync_10;
        }
        #pragma unroll
        for (int i_1_1 = 0; i_1_1 < 1; i_1_1++) {
            int ws_upper_1 = lane & 8;
            float ws_keep_1 = ((ws_upper_1 != 0) ? part[1 + i_1_1] : part[i_1_1]);
            float ws_send_1 = ((ws_upper_1 != 0) ? part[i_1_1] : part[1 + i_1_1]);
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
            part[i_1_1] = ws_keep_1 + _shfl_xor_sync_11;
        }
        float ws_own = part[0];
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 4);
        ws_own = ws_own + _shfl_xor_sync_12;
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 2);
        ws_own = ws_own + _shfl_xor_sync_13;
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
        ws_own = ws_own + _shfl_xor_sync_14;
        float xp_word = ws_own;
        if ((lane & 7) == 0) {
            if (lane >> 3 < 4) {
                wtab[xp_row + (lane >> 3)] = xp_word;
            }
        }
        __syncthreads();
        if (warp == 0) {
            if (lane < 4) {
                int xp_job = 0;
                int xp_dst = lane;
                int xp_base = xp_slot * 28 + 4 * xp_job;
                unsigned int xp_to = recv_addr + (unsigned int)(4 * (xp_slot * 16 + rank * 4));
                float xp_s[4];
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    xp_s[c] = wtab[xp_base + c];
                }
                #pragma unroll
                for (int wi = 1; wi < 7; wi++) {
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 4; c_1++) {
                        xp_s[c_1] = xp_s[c_1] + wtab[xp_base + wi * 4 + c_1];
                    }
                }
                uint32_t _cluster_mapa_15;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_15) : "r"(xp_to + (unsigned int)(16 * xp_job)), "r"(xp_dst));
                uint32_t _mbar_peer_16;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_16) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_15), "r"(__float_as_uint(xp_s[0])), "r"(__float_as_uint(xp_s[1])), "r"(__float_as_uint(xp_s[2])), "r"(__float_as_uint(xp_s[3])), "r"(_mbar_peer_16) : "memory");
            }
        }
        if (tid == 0) {
            mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 64);
        }
        int _stage_xbar = (int)((unsigned int)xk % 2);
        int _parity_xbar = (int)((unsigned int)xk / 2 % 2);
        asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(xbar_addr + (_stage_xbar) * 8), "r"(uint32_t(_parity_xbar)), "r"(uint32_t(10000000)) : "memory");
        asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
        int xt_at = xk % 2 * 16 + jl;
        float xt_t = recv[xt_at];
        #pragma unroll
        for (int rk = 1; rk < 4; rk++) {
            xt_t = xt_t + recv[xt_at + rk * 4];
        }
        float tk_dp = xt_t;
        xk += 1;
        float tk_p = st[0];
        float tk_r = st[1];
        float tk_d = st[2];
        float tk_pd = ((act != 0) ? tk_p * tk_dp : 0.0f);
        float tk_S = tk_pd;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tk_S += __shfl_xor_sync(0xFFFFFFFF, tk_S, offset);
        float tk_ds = tk_p * (tk_dp - tk_S);
        float tk_a = tk_ds * tk_r;
        float tk_b = __fdiv_rn(tk_ds * tk_d * (tk_r * tk_r * tk_r), 7168.0f);
        float tk_w[8];
        #pragma unroll
        for (int i_4 = 0; i_4 < 8; i_4++) {
            tk_w[i_4] = 0.0f;
        }
        if (0 < n) {
            float tj_p_0 = __shfl_sync(0xFFFFFFFF, tk_p, 0);
            float tj_a_0 = __shfl_sync(0xFFFFFFFF, tk_a, 0);
            float _shfl_sync_17 = __shfl_sync(0xFFFFFFFF, tk_b, 0);
            float tj_nb_0;
            asm("neg.f32 %0, %1;" : "=f"(tj_nb_0) : "f"(_shfl_sync_17));
            float2 tj_p2_0 = make_float2(tj_p_0, tj_p_0);
            float2 tj_a2_0 = make_float2(tj_a_0, tj_a_0);
            float2 tj_b2_0 = make_float2(tj_nb_0, tj_nb_0);
            float tj_dv_0[8];
            #pragma unroll
            for (int m_2 = 0; m_2 < 4; m_2++) {
                float2 tj_v_0 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_2]));
                float2 tj_g_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[m_2]));
                float2 tj_wv_0 = make_float2(wf[2 * m_2], wf[2 * m_2 + 1]);
                float2 tj_o_0 = fma_f32x2_rn_noftz(tj_b2_0, tj_v_0, fma_f32x2_rn_noftz(tj_p2_0, tj_g_0, mul_f32x2_noftz(tj_a2_0, tj_wv_0)));
                float2 tj_t_0 = fma_f32x2_rn_noftz(tj_a2_0, tj_v_0, make_float2(tk_w[2 * m_2], tk_w[2 * m_2 + 1]));
                tj_dv_0[2 * m_2] = tj_o_0.x;
                tk_w[2 * m_2] = tj_t_0.x;
                tj_dv_0[2 * m_2 + 1] = tj_o_0.y;
                tk_w[2 * m_2 + 1] = tj_t_0.y;
            }
            unsigned int ss_packed[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tj_dv_0[_lp*2 + 0], tj_dv_0[_lp*2+1 + 0]));
                ss_packed[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dv0 + ((long long)t * 7168ll + (long long)col))[0] = reinterpret_cast<int4*>(ss_packed)[0];
        }
        if (1 < n) {
            float tj_p_1 = __shfl_sync(0xFFFFFFFF, tk_p, 1);
            float tj_a_1 = __shfl_sync(0xFFFFFFFF, tk_a, 1);
            float _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, tk_b, 1);
            float tj_nb_1;
            asm("neg.f32 %0, %1;" : "=f"(tj_nb_1) : "f"(_shfl_sync_18));
            float2 tj_p2_1 = make_float2(tj_p_1, tj_p_1);
            float2 tj_a2_1 = make_float2(tj_a_1, tj_a_1);
            float2 tj_b2_1 = make_float2(tj_nb_1, tj_nb_1);
            float tj_dv_1[8];
            #pragma unroll
            for (int m_3 = 0; m_3 < 4; m_3++) {
                float2 tj_v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_3]));
                float2 tj_g_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[m_3]));
                float2 tj_wv_1 = make_float2(wf[2 * m_3], wf[2 * m_3 + 1]);
                float2 tj_o_1 = fma_f32x2_rn_noftz(tj_b2_1, tj_v_1, fma_f32x2_rn_noftz(tj_p2_1, tj_g_1, mul_f32x2_noftz(tj_a2_1, tj_wv_1)));
                float2 tj_t_1 = fma_f32x2_rn_noftz(tj_a2_1, tj_v_1, make_float2(tk_w[2 * m_3], tk_w[2 * m_3 + 1]));
                tj_dv_1[2 * m_3] = tj_o_1.x;
                tk_w[2 * m_3] = tj_t_1.x;
                tj_dv_1[2 * m_3 + 1] = tj_o_1.y;
                tk_w[2 * m_3 + 1] = tj_t_1.y;
            }
            unsigned int ss_packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tj_dv_1[_lp*2 + 0], tj_dv_1[_lp*2+1 + 0]));
                ss_packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dv1 + ((long long)t * 7168ll + (long long)col))[0] = reinterpret_cast<int4*>(ss_packed_1)[0];
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 8; i_5++) {
            float _add_19;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(blk[i_5]), "f"(tk_w[i_5]));
            blk[i_5] = _add_19;
        }
        nblk = nblk + 1;
        if (nblk == 64) {
            #pragma unroll
            for (int i_6 = 0; i_6 < 8; i_6++) {
                float _add_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(tot[i_6]), "f"(blk[i_6]));
                tot[i_6] = _add_20;
                blk[i_6] = 0.0f;
            }
            nblk = 0;
        }
        if (t + step < T) {
            asm volatile("cp.async.wait_group 0;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gw[0])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * (448 + tid))));
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
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 3; i_7++) {
            st[i_7] = sn[i_7];
        }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; i_8++) {
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(tot[i_8]), "f"(blk[i_8]));
        tot[i_8] = _add_21;
    }
    #pragma unroll
    for (int half_1 = 0; half_1 < 2; half_1++) {
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dw[col + 4 * half_1])), "f"(tot[4 * half_1]), "f"(tot[4 * half_1 + 1]), "f"(tot[4 * half_1 + 2]), "f"(tot[4 * half_1 + 3]) : "memory");
    }
}

} // extern "C"
