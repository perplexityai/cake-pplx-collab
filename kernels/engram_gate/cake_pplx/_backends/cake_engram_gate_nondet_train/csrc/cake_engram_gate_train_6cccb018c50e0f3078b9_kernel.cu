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
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_engram_gate_train_6cccb018c50e0f3078b9(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, float* __restrict__ w, bool* __restrict__ mask, __nv_bfloat16* __restrict__ out, int T, float eps, float clamp, long long key_stride_p0, long long key_stride_p1, long long value_stride_p0)
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
    float* recv = reinterpret_cast<float*>(smem_raw + 1824);
    const int recv_addr = smem + 1824;

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
    int col = rank * 1280 + tid * 8;
    int first = (int)cluster_id;
    int step = (int)num_clusters;
    float wf[32];
    unsigned int xw[16];
    unsigned int kw[16];
    unsigned int vw[4];
    float plain[16];
    #pragma unroll
    for (int i = 12; i < 16; i++) {
        plain[i] = 0.0f;
    }
    float ps[4];
    float pe[4];
    float tot[3];
    float gates[4];
    int xk = 0;
    int keep = 1;
    unsigned int xn[16];
    unsigned int kn[16];
    unsigned int vn[4];
    int keep_n = 1;
    int ftok = ((first) < (T - 1) ? (first) : (T - 1));
    long long ls_at = (long long)ftok * 20480ll + (long long)col;
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (ls_at + (long long)(s * 5120)) + 0);
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&xw[4 * s]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
    }
    long long lk_at = (long long)ftok * key_stride_p0 + (long long)col;
    #pragma unroll
    for (int s_1 = 0; s_1 < 4; s_1++) {
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(key + (lk_at + (long long)s_1 * key_stride_p1) + 0);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&kw[4 * s_1]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
            }
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(value + ((long long)ftok * value_stride_p0 + (long long)col) + 0);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&vw[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_2[_blk] = _vptr_2[_blk];
        }
    }
    int tk_keep = 1;
    if ((unsigned long long)mask != 0ull) {
        if (!mask[ftok]) {
            tk_keep = 0;
        }
    }
    keep = tk_keep;
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    #pragma unroll 1
    for (int t = first; t < T; t += step) {
        long long to_base = (long long)t * 20480ll + (long long)col;
        if (keep != 0) {
            #pragma unroll
            for (int s_2 = 0; s_2 < 4; s_2++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    {
                        unsigned _v4_3_0;
                        unsigned _v4_3_1;
                        unsigned _v4_3_2;
                        unsigned _v4_3_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(w + (s_2 * 5120 + col + 4 * half) + (0))) : "memory");
                        wf[8 * s_2 + 4 * half + 0] = __uint_as_float(_v4_3_0);
                        wf[8 * s_2 + 4 * half + 1] = __uint_as_float(_v4_3_1);
                        wf[8 * s_2 + 4 * half + 2] = __uint_as_float(_v4_3_2);
                        wf[8 * s_2 + 4 * half + 3] = __uint_as_float(_v4_3_3);
                    }
                }
            }
            #pragma unroll
            for (int s_3 = 0; s_3 < 4; s_3++) {
                float ss_h = 0.0f;
                float ss_k = 0.0f;
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    float _dot_4;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_4) : "r"(xw[4 * s_3 + m]), "r"(xw[4 * s_3 + m]), "f"(ss_h));
                    ss_h = _dot_4;
                    float _dot_5;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_5) : "r"(kw[4 * s_3 + m]), "r"(kw[4 * s_3 + m]), "f"(ss_k));
                    ss_k = _dot_5;
                }
                plain[s_3] = ss_h;
                plain[4 + s_3] = ss_k;
                float2 ss_p = make_float2(0.0f, 0.0f);
                #pragma unroll
                for (int m_1 = 0; m_1 < 4; m_1++) {
                    float2 _f32x2_mul_a_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_3 + m_1]));
                    float2 _f32x2_mul_b_7 = cake_bf16x2_to_f32x2(__as_bf16x2(kw[4 * s_3 + m_1]));
                    float2 ss_hk;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ss_hk) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
                    ss_p = fma_f32x2_rn_noftz(make_float2(wf[8 * s_3 + 2 * m_1], wf[8 * s_3 + 2 * m_1 + 1]), ss_hk, ss_p);
                }
                float _add_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(ss_p.x), "f"(ss_p.y));
                plain[8 + s_3] = _add_8;
            }
            int pn_tok = t + step;
            if (pn_tok < T) {
                long long ls_at_0 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
                for (int s_4 = 0; s_4 < 4; s_4++) {
                    {
                        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + (ls_at_0 + (long long)(s_4 * 5120)) + 0);
                        uint4* _vdst_9 = reinterpret_cast<uint4*>(&xn[4 * s_4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_9[_blk] = _vptr_9[_blk];
                        }
                    }
                }
                long long lk_at_0 = (long long)pn_tok * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_5 = 0; s_5 < 4; s_5++) {
                    {
                        const uint4* _vptr_10 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_5 * key_stride_p1) + 0);
                        uint4* _vdst_10 = reinterpret_cast<uint4*>(&kn[4 * s_5]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_10[_blk] = _vptr_10[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_11 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_11 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_11[_blk] = _vptr_11[_blk];
                    }
                }
                int tk_keep_0 = 1;
                if ((unsigned long long)mask != 0ull) {
                    if (!mask[pn_tok]) {
                        tk_keep_0 = 0;
                    }
                }
                keep_n = tk_keep_0;
            }
            int xc_slot = xk % 2;
            int xc_row = xc_slot * 100 + warp * 12;
            #pragma unroll
            for (int i_0 = 0; i_0 < 8; i_0++) {
                int ws_upper_0 = lane & 16;
                float ws_keep_0 = ((ws_upper_0 != 0) ? plain[8 + i_0] : plain[i_0]);
                float ws_send_0 = ((ws_upper_0 != 0) ? plain[i_0] : plain[8 + i_0]);
                float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_12;
            }
            #pragma unroll
            for (int i_1 = 0; i_1 < 4; i_1++) {
                int ws_upper_1 = lane & 8;
                float ws_keep_1 = ((ws_upper_1 != 0) ? plain[4 + i_1] : plain[i_1]);
                float ws_send_1 = ((ws_upper_1 != 0) ? plain[i_1] : plain[4 + i_1]);
                float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1] = ws_keep_1 + _shfl_xor_sync_13;
            }
            #pragma unroll
            for (int i_2 = 0; i_2 < 2; i_2++) {
                int ws_upper_2 = lane & 4;
                float ws_keep_2 = ((ws_upper_2 != 0) ? plain[2 + i_2] : plain[i_2]);
                float ws_send_2 = ((ws_upper_2 != 0) ? plain[i_2] : plain[2 + i_2]);
                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2] = ws_keep_2 + _shfl_xor_sync_14;
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 1; i_3++) {
                int ws_upper_3 = lane & 2;
                float ws_keep_3 = ((ws_upper_3 != 0) ? plain[1 + i_3] : plain[i_3]);
                float ws_send_3 = ((ws_upper_3 != 0) ? plain[i_3] : plain[1 + i_3]);
                float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
                plain[i_3] = ws_keep_3 + _shfl_xor_sync_15;
            }
            float ws_own = plain[0];
            float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_16;
            float xc_word = ws_own;
            if ((lane & 1) == 0) {
                if (lane >> 1 < 12) {
                    wtab[xc_row + (lane >> 1)] = xc_word;
                }
            }
            __syncthreads();
            if (warp == 0) {
                if (lane < 12) {
                    int xc_job = lane % 3;
                    int xc_dst = lane / 3;
                    int xc_base = xc_slot * 100 + 4 * xc_job;
                    unsigned int xc_to = recv_addr + (unsigned int)(4 * (xc_slot * 80 + rank * 12));
                    float xc_s[4];
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        xc_s[c] = wtab[xc_base + c];
                    }
                    if (xc_job < 3) {
                        #pragma unroll
                        for (int wi = 1; wi < 5; wi++) {
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 4; c_1++) {
                                xc_s[c_1] = xc_s[c_1] + wtab[xc_base + wi * 12 + c_1];
                            }
                        }
                    } else {
                        float xc_e[4];
                        #pragma unroll
                        for (int c_2 = 0; c_2 < 4; c_2++) {
                            float xc_fs = xc_s[c_2];
                            float xc_fe = wtab[xc_base + 4 + c_2];
                            #pragma unroll
                            for (int wi_1 = 1; wi_1 < 5; wi_1++) {
                                float ts_s;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(xc_fs), "f"(wtab[xc_base + wi_1 * 12 + c_2]));
                                float ts_bb;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(xc_fs));
                                float _sub_18;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(ts_s), "f"(ts_bb));
                                float _sub_17;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(xc_fs), "f"(_sub_18));
                                float _sub_19;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(wtab[xc_base + wi_1 * 12 + c_2]), "f"(ts_bb));
                                float ts_e;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_17), "f"(_sub_19));
                                xc_fs = ts_s;
                                float _add_21;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(xc_fe), "f"(wtab[xc_base + wi_1 * 12 + 4 + c_2]));
                                float _add_20;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_add_21), "f"(ts_e));
                                xc_fe = _add_20;
                            }
                            xc_s[c_2] = xc_fs;
                            xc_e[c_2] = xc_fe;
                        }
                        uint32_t _cluster_mapa_22;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_22) : "r"(xc_to + (unsigned int)(16 * (xc_job + 1))), "r"(xc_dst));
                        uint32_t _mbar_peer_23;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_23) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xc_dst));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_22), "r"(__float_as_uint(xc_e[0])), "r"(__float_as_uint(xc_e[1])), "r"(__float_as_uint(xc_e[2])), "r"(__float_as_uint(xc_e[3])), "r"(_mbar_peer_23) : "memory");
                    }
                    uint32_t _cluster_mapa_24;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_24) : "r"(xc_to + (unsigned int)(16 * xc_job)), "r"(xc_dst));
                    uint32_t _mbar_peer_25;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_25) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xc_dst));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_24), "r"(__float_as_uint(xc_s[0])), "r"(__float_as_uint(xc_s[1])), "r"(__float_as_uint(xc_s[2])), "r"(__float_as_uint(xc_s[3])), "r"(_mbar_peer_25) : "memory");
                }
            }
            if (tid == 0) {
                mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 192);
            }
            int _stage_xbar = (int)((unsigned int)xk % 2);
            int _parity_xbar = (int)((unsigned int)xk / 2 % 2);
            asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(xbar_addr + (_stage_xbar) * 8), "r"(uint32_t(_parity_xbar)), "r"(uint32_t(10000000)) : "memory");
            asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
            int xc_from = xc_slot * 80 + (lane & 3);
            #pragma unroll
            for (int q = 0; q < 3; q++) {
                float xc_t = recv[xc_from + 4 * q];
                #pragma unroll
                for (int r = 1; r < 4; r++) {
                    xc_t = xc_t + recv[xc_from + r * 12 + 4 * q];
                }
                tot[q] = xc_t;
            }
            xk += 1;
            float _fdiv_27 = __fdiv_rn(tot[0], 5120.0f);
            float _sqrt_26;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_26) : "f"(_fdiv_27 + eps));
            float sg_rh = __frcp_rn(_sqrt_26);
            float _fdiv_29 = __fdiv_rn(tot[1], 5120.0f);
            float _sqrt_28;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_28) : "f"(_fdiv_29 + eps));
            float sg_rk = __frcp_rn(_sqrt_28);
            float sg_dot;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(sg_dot) : "f"(mul_f32_rn_noftz(tot[2], mul_f32_rn_noftz(sg_rh, sg_rk))), "f"(0.013975425f));
            float _abs_31;
            asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(sg_dot));
            float _max_30;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_30) : "f"(_abs_31), "f"(clamp));
            float sg_mag;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_30));
            float sg_u = __uint_as_float(__as_u32(sg_mag) & 2147483647 | __as_u32(sg_dot) & 2147483648u);
            float _neg_33;
            asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(sg_u));
            float _exp_32 = __nv_expf(_neg_33);
            float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_32);
            #pragma unroll
            for (int s_6 = 0; s_6 < 4; s_6++) {
                float _shfl_sync_34 = __shfl_sync(0xFFFFFFFF, sg_gate, s_6);
                gates[s_6] = _shfl_sync_34;
            }
            #pragma unroll
            for (int s_7 = 0; s_7 < 4; s_7++) {
                float2 to_g2 = make_float2(gates[s_7], gates[s_7]);
                float to_out[8];
                #pragma unroll
                for (int m_2 = 0; m_2 < 4; m_2++) {
                    float2 to_o2 = fma_f32x2_rn_noftz(to_g2, cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_2])), cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_7 + m_2])));
                    to_out[2 * m_2] = to_o2.x;
                    to_out[2 * m_2 + 1] = to_o2.y;
                }
                unsigned int to_packed[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_out[_lp*2 + 0], to_out[_lp*2+1 + 0]));
                    to_packed[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (to_base + (long long)(s_7 * 5120)))[0] = reinterpret_cast<int4*>(to_packed)[0];
            }
        } else {
            int pn_tok_1 = t + step;
            if (pn_tok_1 < T) {
                long long ls_at_1 = (long long)pn_tok_1 * 20480ll + (long long)col;
                #pragma unroll
                for (int s_8 = 0; s_8 < 4; s_8++) {
                    {
                        const uint4* _vptr_35 = reinterpret_cast<const uint4*>(x + (ls_at_1 + (long long)(s_8 * 5120)) + 0);
                        uint4* _vdst_35 = reinterpret_cast<uint4*>(&xn[4 * s_8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_35[_blk] = _vptr_35[_blk];
                        }
                    }
                }
                long long lk_at_1 = (long long)pn_tok_1 * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_9 = 0; s_9 < 4; s_9++) {
                    {
                        const uint4* _vptr_36 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_9 * key_stride_p1) + 0);
                        uint4* _vdst_36 = reinterpret_cast<uint4*>(&kn[4 * s_9]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_36[_blk] = _vptr_36[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_37 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_37 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_37[_blk] = _vptr_37[_blk];
                    }
                }
                int tk_keep_1 = 1;
                if ((unsigned long long)mask != 0ull) {
                    if (!mask[pn_tok_1]) {
                        tk_keep_1 = 0;
                    }
                }
                keep_n = tk_keep_1;
            }
            #pragma unroll
            for (int s_10 = 0; s_10 < 4; s_10++) {
                reinterpret_cast<int4*>(out + (to_base + (long long)(s_10 * 5120)))[0] = reinterpret_cast<int4*>(xw + (4 * s_10))[0];
            }
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 16; i_4++) {
            xw[i_4] = xn[i_4];
            kw[i_4] = kn[i_4];
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; i_5++) {
            vw[i_5] = vn[i_5];
        }
        keep = keep_n;
    }
}

} // extern "C"
