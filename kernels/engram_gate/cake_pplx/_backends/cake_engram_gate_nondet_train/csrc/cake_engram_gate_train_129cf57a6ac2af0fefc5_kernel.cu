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

#define LAUNCH_MIN_BLOCKS 3

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
kernel_cake_engram_gate_train_129cf57a6ac2af0fefc5(__nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, float* __restrict__ w, bool* __restrict__ mask, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dkey, __nv_bfloat16* __restrict__ dvalue, float* __restrict__ dw_acc, int T, float eps, float clamp, long long key_stride_p0, long long key_stride_p1, long long value_stride_p0, long long dkey_stride_p0, long long dkey_stride_p1, long long dvalue_stride_p0)
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
    float* wslots = reinterpret_cast<float*>(smem_raw + 2464);
    const int wslots_addr = smem + 2464;
    float* dslots = reinterpret_cast<float*>(smem_raw + 22944);
    const int dslots_addr = smem + 22944;

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
    unsigned int gw[16];
    unsigned int xw[16];
    unsigned int kw[16];
    unsigned int vw[4];
    float dwa[32];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            {
                unsigned _v4_0_0;
                unsigned _v4_0_1;
                unsigned _v4_0_2;
                unsigned _v4_0_3;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(w + (s * 5120 + col + 4 * half) + (0))) : "memory");
                wf[8 * s + 4 * half + 0] = __uint_as_float(_v4_0_0);
                wf[8 * s + 4 * half + 1] = __uint_as_float(_v4_0_1);
                wf[8 * s + 4 * half + 2] = __uint_as_float(_v4_0_2);
                wf[8 * s + 4 * half + 3] = __uint_as_float(_v4_0_3);
            }
        }
    }
    #pragma unroll
    for (int s_1 = 0; s_1 < 4; s_1++) {
        #pragma unroll
        for (int half_1 = 0; half_1 < 2; half_1++) {
            unsigned int bw_at = (unsigned int)(16 * ((2 * s_1 + half_1) * 160 + tid));
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(wslots_addr + bw_at), "f"(wf[8 * s_1 + 4 * half_1]), "f"(wf[8 * s_1 + 4 * half_1 + 1]), "f"(wf[8 * s_1 + 4 * half_1 + 2]), "f"(wf[8 * s_1 + 4 * half_1 + 3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dslots_addr + bw_at), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f) : "memory");
        }
    }
    float plain[16];
    #pragma unroll
    for (int i = 12; i < 16; i++) {
        plain[i] = 0.0f;
    }
    float ps[4];
    float pe[4];
    float tot[4];
    unsigned int zw[4];
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; i_1++) {
        zw[i_1] = 0;
    }
    int xk = 0;
    int keep = 1;
    unsigned int gn[16];
    unsigned int xn[16];
    unsigned int kn[16];
    unsigned int vn[4];
    int keep_n = 1;
    int ftok = ((first) < (T - 1) ? (first) : (T - 1));
    long long ls_at = (long long)ftok * 20480ll + (long long)col;
    #pragma unroll
    for (int s_2 = 0; s_2 < 4; s_2++) {
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(dout + (ls_at + (long long)(s_2 * 5120)) + 0);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&gw[4 * s_2]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
            }
        }
    }
    long long ls_at_0 = (long long)ftok * 20480ll + (long long)col;
    #pragma unroll
    for (int s_3 = 0; s_3 < 4; s_3++) {
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + (ls_at_0 + (long long)(s_3 * 5120)) + 0);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&xw[4 * s_3]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
    }
    long long lk_at = (long long)ftok * key_stride_p0 + (long long)col;
    #pragma unroll
    for (int s_4 = 0; s_4 < 4; s_4++) {
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(key + (lk_at + (long long)s_4 * key_stride_p1) + 0);
            uint4* _vdst_3 = reinterpret_cast<uint4*>(&kw[4 * s_4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_3[_blk] = _vptr_3[_blk];
            }
        }
    }
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(value + ((long long)ftok * value_stride_p0 + (long long)col) + 0);
        uint4* _vdst_4 = reinterpret_cast<uint4*>(&vw[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_4[_blk] = _vptr_4[_blk];
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
        long long tg_x = (long long)t * 20480ll + (long long)col;
        long long tg_k = (long long)t * dkey_stride_p0 + (long long)col;
        long long tg_v = (long long)t * dvalue_stride_p0 + (long long)col;
        if (keep != 0) {
            #pragma unroll
            for (int s_5 = 0; s_5 < 4; s_5++) {
                #pragma unroll
                for (int half_2 = 0; half_2 < 2; half_2++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_5 + 4 * half_2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_5 + 4 * half_2) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_5 + 4 * half_2) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_5 + 4 * half_2) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_5 + half_2) * 160 + tid))));
                }
            }
            #pragma unroll
            for (int s_6 = 0; s_6 < 4; s_6++) {
                float ss_h = 0.0f;
                float ss_k = 0.0f;
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    float _dot_5;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_5) : "r"(xw[4 * s_6 + m]), "r"(xw[4 * s_6 + m]), "f"(ss_h));
                    ss_h = _dot_5;
                    float _dot_6;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_6) : "r"(kw[4 * s_6 + m]), "r"(kw[4 * s_6 + m]), "f"(ss_k));
                    ss_k = _dot_6;
                }
                plain[s_6] = ss_h;
                plain[4 + s_6] = ss_k;
            }
            #pragma unroll
            for (int s_7 = 0; s_7 < 4; s_7++) {
                float sp_s = 0.0f;
                float sp_e = 0.0f;
                #pragma unroll
                for (int m_1 = 0; m_1 < 4; m_1++) {
                    float sp_hk[2] = {0.0f, 0.0f};
                    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk[0]) : "h"(static_cast<uint16_t>((xw[4 * s_7 + m_1]))), "h"(static_cast<uint16_t>((kw[4 * s_7 + m_1]))));
                    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk[1]) : "h"(static_cast<uint16_t>((xw[4 * s_7 + m_1]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_7 + m_1]) >> 16)));
                    float sp_w_0 = wf[8 * s_7 + 2 * m_1];
                    float sp_p_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p_0) : "f"(sp_w_0), "f"(sp_hk[0]));
                    float _neg_7;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_7) : "f"(sp_p_0));
                    float sp_q_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q_0) : "f"(sp_w_0), "f"(sp_hk[0]), "f"(_neg_7));
                    float ts_s;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(sp_s), "f"(sp_p_0));
                    float ts_bb;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(sp_s));
                    float _sub_9;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(ts_s), "f"(ts_bb));
                    float _sub_8;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(sp_s), "f"(_sub_9));
                    float _sub_10;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(sp_p_0), "f"(ts_bb));
                    float ts_e;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_8), "f"(_sub_10));
                    sp_s = ts_s;
                    float _add_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(sp_q_0), "f"(ts_e));
                    float _add_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(sp_e), "f"(_add_12));
                    sp_e = _add_11;
                    float sp_w_1 = wf[8 * s_7 + 2 * m_1 + 1];
                    float sp_p_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p_1) : "f"(sp_w_1), "f"(sp_hk[1]));
                    float _neg_13;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(sp_p_1));
                    float sp_q_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q_1) : "f"(sp_w_1), "f"(sp_hk[1]), "f"(_neg_13));
                    float ts_s_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(sp_s), "f"(sp_p_1));
                    float ts_bb_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_0), "f"(sp_s));
                    float _sub_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(ts_s_0), "f"(ts_bb_0));
                    float _sub_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(sp_s), "f"(_sub_15));
                    float _sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(sp_p_1), "f"(ts_bb_0));
                    float ts_e_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_14), "f"(_sub_16));
                    sp_s = ts_s_0;
                    float _add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(sp_q_1), "f"(ts_e_0));
                    float _add_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(sp_e), "f"(_add_18));
                    sp_e = _add_17;
                }
                ps[s_7] = sp_s;
                pe[s_7] = sp_e;
            }
            #pragma unroll
            for (int s_8 = 0; s_8 < 4; s_8++) {
                float tf_q = 0.0f;
                #pragma unroll
                for (int m_2 = 0; m_2 < 4; m_2++) {
                    float _dot_19;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_19) : "r"(gw[4 * s_8 + m_2]), "r"(vw[m_2]), "f"(tf_q));
                    tf_q = _dot_19;
                }
                plain[8 + s_8] = tf_q;
            }
            int pn_tok = t + step;
            if (pn_tok < T) {
                long long ls_at_1 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
                for (int s_9 = 0; s_9 < 4; s_9++) {
                    {
                        const uint4* _vptr_20 = reinterpret_cast<const uint4*>(dout + (ls_at_1 + (long long)(s_9 * 5120)) + 0);
                        uint4* _vdst_20 = reinterpret_cast<uint4*>(&gn[4 * s_9]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_20[_blk] = _vptr_20[_blk];
                        }
                    }
                }
                long long ls_at_2 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
                for (int s_10 = 0; s_10 < 4; s_10++) {
                    {
                        const uint4* _vptr_21 = reinterpret_cast<const uint4*>(x + (ls_at_2 + (long long)(s_10 * 5120)) + 0);
                        uint4* _vdst_21 = reinterpret_cast<uint4*>(&xn[4 * s_10]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_21[_blk] = _vptr_21[_blk];
                        }
                    }
                }
                long long lk_at_0 = (long long)pn_tok * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_11 = 0; s_11 < 4; s_11++) {
                    {
                        const uint4* _vptr_22 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_11 * key_stride_p1) + 0);
                        uint4* _vdst_22 = reinterpret_cast<uint4*>(&kn[4 * s_11]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_22[_blk] = _vptr_22[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_23 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_23 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_23[_blk] = _vptr_23[_blk];
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
            int xc_row = xc_slot * 100 + warp * 20;
            #pragma unroll
            for (int i_0 = 0; i_0 < 8; i_0++) {
                int ws_upper_0 = lane & 16;
                float ws_keep_0 = ((ws_upper_0 != 0) ? plain[8 + i_0] : plain[i_0]);
                float ws_send_0 = ((ws_upper_0 != 0) ? plain[i_0] : plain[8 + i_0]);
                float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_24;
            }
            #pragma unroll
            for (int i_1_1 = 0; i_1_1 < 4; i_1_1++) {
                int ws_upper_1 = lane & 8;
                float ws_keep_1 = ((ws_upper_1 != 0) ? plain[4 + i_1_1] : plain[i_1_1]);
                float ws_send_1 = ((ws_upper_1 != 0) ? plain[i_1_1] : plain[4 + i_1_1]);
                float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1_1] = ws_keep_1 + _shfl_xor_sync_25;
            }
            #pragma unroll
            for (int i_2 = 0; i_2 < 2; i_2++) {
                int ws_upper_2 = lane & 4;
                float ws_keep_2 = ((ws_upper_2 != 0) ? plain[2 + i_2] : plain[i_2]);
                float ws_send_2 = ((ws_upper_2 != 0) ? plain[i_2] : plain[2 + i_2]);
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2] = ws_keep_2 + _shfl_xor_sync_26;
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 1; i_3++) {
                int ws_upper_3 = lane & 2;
                float ws_keep_3 = ((ws_upper_3 != 0) ? plain[1 + i_3] : plain[i_3]);
                float ws_send_3 = ((ws_upper_3 != 0) ? plain[i_3] : plain[1 + i_3]);
                float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
                plain[i_3] = ws_keep_3 + _shfl_xor_sync_27;
            }
            float ws_own = plain[0];
            float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_28;
            float xc_word = ws_own;
            if ((lane & 1) == 0) {
                if (lane >> 1 < 12) {
                    wtab[xc_row + (lane >> 1)] = xc_word;
                }
            }
            #pragma unroll
            for (int i_4 = 0; i_4 < 2; i_4++) {
                int wp_upper_0 = lane & 16;
                float wp_ks_0 = ((wp_upper_0 != 0) ? ps[2 + i_4] : ps[i_4]);
                float wp_ke_0 = ((wp_upper_0 != 0) ? pe[2 + i_4] : pe[i_4]);
                float wp_rs_0 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_0 != 0) ? ps[i_4] : ps[2 + i_4]), 16);
                float wp_re_0 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_0 != 0) ? pe[i_4] : pe[2 + i_4]), 16);
                float ts_s_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(wp_ks_0), "f"(wp_rs_0));
                float ts_bb_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_1), "f"(wp_ks_0));
                float _sub_30;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(ts_s_1), "f"(ts_bb_1));
                float _sub_29;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(wp_ks_0), "f"(_sub_30));
                float _sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(wp_rs_0), "f"(ts_bb_1));
                float ts_e_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_29), "f"(_sub_31));
                wp_ks_0 = ts_s_1;
                float _add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(wp_ke_0), "f"(wp_re_0));
                float _add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_add_33), "f"(ts_e_1));
                wp_ke_0 = _add_32;
                ps[i_4] = wp_ks_0;
                pe[i_4] = wp_ke_0;
            }
            #pragma unroll
            for (int i_5 = 0; i_5 < 1; i_5++) {
                int wp_upper_1 = lane & 8;
                float wp_ks_1 = ((wp_upper_1 != 0) ? ps[1 + i_5] : ps[i_5]);
                float wp_ke_1 = ((wp_upper_1 != 0) ? pe[1 + i_5] : pe[i_5]);
                float wp_rs_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? ps[i_5] : ps[1 + i_5]), 8);
                float wp_re_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? pe[i_5] : pe[1 + i_5]), 8);
                float ts_s_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(wp_ks_1), "f"(wp_rs_1));
                float ts_bb_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_2), "f"(wp_ks_1));
                float _sub_35;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(ts_s_2), "f"(ts_bb_2));
                float _sub_34;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(wp_ks_1), "f"(_sub_35));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(wp_rs_1), "f"(ts_bb_2));
                float ts_e_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_34), "f"(_sub_36));
                wp_ks_1 = ts_s_2;
                float _add_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(wp_ke_1), "f"(wp_re_1));
                float _add_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_38), "f"(ts_e_2));
                wp_ke_1 = _add_37;
                ps[i_5] = wp_ks_1;
                pe[i_5] = wp_ke_1;
            }
            float wp_s = ps[0];
            float wp_e = pe[0];
            float wp_bs_0 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 4);
            float wp_be_0 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 4);
            float ts_s_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3) : "f"(wp_s), "f"(wp_bs_0));
            float ts_bb_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3) : "f"(ts_s_3), "f"(wp_s));
            float _sub_40;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(ts_s_3), "f"(ts_bb_3));
            float _sub_39;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(wp_s), "f"(_sub_40));
            float _sub_41;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(wp_bs_0), "f"(ts_bb_3));
            float ts_e_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_39), "f"(_sub_41));
            wp_s = ts_s_3;
            float _add_43;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(wp_e), "f"(wp_be_0));
            float _add_42;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_add_43), "f"(ts_e_3));
            wp_e = _add_42;
            float wp_bs_1 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 2);
            float wp_be_1 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 2);
            float ts_s_1_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1_1) : "f"(wp_s), "f"(wp_bs_1));
            float ts_bb_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1_1) : "f"(ts_s_1_1), "f"(wp_s));
            float _sub_45;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(ts_s_1_1), "f"(ts_bb_1_1));
            float _sub_44;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(wp_s), "f"(_sub_45));
            float _sub_46;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(wp_bs_1), "f"(ts_bb_1_1));
            float ts_e_1_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1_1) : "f"(_sub_44), "f"(_sub_46));
            wp_s = ts_s_1_1;
            float _add_48;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(wp_e), "f"(wp_be_1));
            float _add_47;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_add_48), "f"(ts_e_1_1));
            wp_e = _add_47;
            float wp_bs_2 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 1);
            float wp_be_2 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 1);
            float ts_s_2_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2_1) : "f"(wp_s), "f"(wp_bs_2));
            float ts_bb_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2_1) : "f"(ts_s_2_1), "f"(wp_s));
            float _sub_50;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(ts_s_2_1), "f"(ts_bb_2_1));
            float _sub_49;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(wp_s), "f"(_sub_50));
            float _sub_51;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(wp_bs_2), "f"(ts_bb_2_1));
            float ts_e_2_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2_1) : "f"(_sub_49), "f"(_sub_51));
            wp_s = ts_s_2_1;
            float _add_53;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(wp_e), "f"(wp_be_2));
            float _add_52;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_53), "f"(ts_e_2_1));
            wp_e = _add_52;
            if ((lane & 7) == 0) {
                wtab[xc_row + 12 + (lane >> 3)] = wp_s;
                wtab[xc_row + 16 + (lane >> 3)] = wp_e;
            }
            __syncthreads();
            if (warp == 0) {
                if (lane < 16) {
                    int xc_job = lane % 4;
                    int xc_dst = lane / 4;
                    int xc_base = xc_slot * 100 + 4 * xc_job;
                    unsigned int xc_to = recv_addr + (unsigned int)(4 * (xc_slot * 80 + rank * 20));
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
                                xc_s[c_1] = xc_s[c_1] + wtab[xc_base + wi * 20 + c_1];
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
                                float ts_s_3_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3_1) : "f"(xc_fs), "f"(wtab[xc_base + wi_1 * 20 + c_2]));
                                float ts_bb_3_1;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3_1) : "f"(ts_s_3_1), "f"(xc_fs));
                                float _sub_55;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(ts_s_3_1), "f"(ts_bb_3_1));
                                float _sub_54;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(xc_fs), "f"(_sub_55));
                                float _sub_56;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(wtab[xc_base + wi_1 * 20 + c_2]), "f"(ts_bb_3_1));
                                float ts_e_3_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3_1) : "f"(_sub_54), "f"(_sub_56));
                                xc_fs = ts_s_3_1;
                                float _add_58;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(xc_fe), "f"(wtab[xc_base + wi_1 * 20 + 4 + c_2]));
                                float _add_57;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_add_58), "f"(ts_e_3_1));
                                xc_fe = _add_57;
                            }
                            xc_s[c_2] = xc_fs;
                            xc_e[c_2] = xc_fe;
                        }
                        uint32_t _cluster_mapa_59;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_59) : "r"(xc_to + (unsigned int)(16 * (xc_job + 1))), "r"(xc_dst));
                        uint32_t _mbar_peer_60;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_60) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xc_dst));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_59), "r"(__float_as_uint(xc_e[0])), "r"(__float_as_uint(xc_e[1])), "r"(__float_as_uint(xc_e[2])), "r"(__float_as_uint(xc_e[3])), "r"(_mbar_peer_60) : "memory");
                    }
                    uint32_t _cluster_mapa_61;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_61) : "r"(xc_to + (unsigned int)(16 * xc_job)), "r"(xc_dst));
                    uint32_t _mbar_peer_62;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_62) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xc_dst));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_61), "r"(__float_as_uint(xc_s[0])), "r"(__float_as_uint(xc_s[1])), "r"(__float_as_uint(xc_s[2])), "r"(__float_as_uint(xc_s[3])), "r"(_mbar_peer_62) : "memory");
                }
            }
            if (tid == 0) {
                mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 320);
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
                    xc_t = xc_t + recv[xc_from + r * 20 + 4 * q];
                }
                tot[q] = xc_t;
            }
            float xc_ts = recv[xc_from + 12];
            float xc_te = recv[xc_from + 16];
            #pragma unroll
            for (int r_1 = 1; r_1 < 4; r_1++) {
                float ts_s_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_4) : "f"(xc_ts), "f"(recv[xc_from + r_1 * 20 + 12]));
                float ts_bb_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_4) : "f"(ts_s_4), "f"(xc_ts));
                float _sub_64;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(ts_s_4), "f"(ts_bb_4));
                float _sub_63;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(xc_ts), "f"(_sub_64));
                float _sub_65;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(recv[xc_from + r_1 * 20 + 12]), "f"(ts_bb_4));
                float ts_e_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_63), "f"(_sub_65));
                xc_ts = ts_s_4;
                float _add_67;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(xc_te), "f"(recv[xc_from + r_1 * 20 + 16]));
                float _add_66;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(_add_67), "f"(ts_e_4));
                xc_te = _add_66;
            }
            float _add_68;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(xc_ts), "f"(xc_te));
            tot[3] = _add_68;
            xk += 1;
            float tf_p = tot[3];
            float _fdiv_70 = __fdiv_rn(tot[0], 5120.0f);
            float _sqrt_69;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_69) : "f"(_fdiv_70 + eps));
            float sg_rh = __frcp_rn(_sqrt_69);
            float _fdiv_72 = __fdiv_rn(tot[1], 5120.0f);
            float _sqrt_71;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_71) : "f"(_fdiv_72 + eps));
            float sg_rk = __frcp_rn(_sqrt_71);
            float sg_dot;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(sg_dot) : "f"(mul_f32_rn_noftz(tf_p, mul_f32_rn_noftz(sg_rh, sg_rk))), "f"(0.013975425f));
            float _abs_74;
            asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(sg_dot));
            float _max_73;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_73) : "f"(_abs_74), "f"(clamp));
            float sg_mag;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_73));
            float sg_u = __uint_as_float(__as_u32(sg_mag) & 2147483647 | __as_u32(sg_dot) & 2147483648u);
            float _neg_76;
            asm("neg.f32 %0, %1;" : "=f"(_neg_76) : "f"(sg_u));
            float _exp_75 = __nv_expf(_neg_76);
            float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_75);
            float sd_mag;
            asm("abs.f32 %0, %1;" : "=f"(sd_mag) : "f"(sg_dot));
            float sd_du = 0.0f;
            if (sd_mag > clamp) {
                float _sqrt_78;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_78) : "f"(sd_mag));
                float _fdiv_77 = __fdiv_rn(0.5f, _sqrt_78);
                sd_du = _fdiv_77;
            }
            float sr_du = sd_du;
            float sr_c = tot[2] * sg_gate * (1.0f - sg_gate) * sr_du * sg_rh * sg_rk * 0.013975425f;
            float sr_ah = __fdiv_rn(tf_p * (sg_rh * sg_rh), 5120.0f);
            float sr_ak = __fdiv_rn(tf_p * (sg_rk * sg_rk), 5120.0f);
            float tg_dv[8];
            #pragma unroll
            for (int i_6 = 0; i_6 < 8; i_6++) {
                tg_dv[i_6] = 0.0f;
            }
            #pragma unroll
            for (int s_12 = 0; s_12 < 4; s_12++) {
                float tg_sg = __shfl_sync(0xFFFFFFFF, sg_gate, s_12);
                float tg_sc = __shfl_sync(0xFFFFFFFF, sr_c, s_12);
                float tg_sh = __shfl_sync(0xFFFFFFFF, sr_ah, s_12);
                float tg_sk = __shfl_sync(0xFFFFFFFF, sr_ak, s_12);
                float2 tg_g2 = make_float2(tg_sg, tg_sg);
                float2 tg_c2 = make_float2(tg_sc, tg_sc);
                float _neg_79;
                asm("neg.f32 %0, %1;" : "=f"(_neg_79) : "f"(tg_sh));
                float _neg_80;
                asm("neg.f32 %0, %1;" : "=f"(_neg_80) : "f"(tg_sh));
                float2 tg_h2 = make_float2(_neg_79, _neg_80);
                float _neg_81;
                asm("neg.f32 %0, %1;" : "=f"(_neg_81) : "f"(tg_sk));
                float _neg_82;
                asm("neg.f32 %0, %1;" : "=f"(_neg_82) : "f"(tg_sk));
                float2 tg_k2 = make_float2(_neg_81, _neg_82);
                #pragma unroll
                for (int half_3 = 0; half_3 < 2; half_3++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_12 + 4 * half_3])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_12 + 4 * half_3) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_12 + 4 * half_3) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_12 + 4 * half_3) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_12 + half_3) * 160 + tid))));
                }
                #pragma unroll
                for (int half_4 = 0; half_4 < 2; half_4++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dwa[8 * s_12 + 4 * half_4])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_12 + 4 * half_4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_12 + 4 * half_4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_12 + 4 * half_4) + 3]))
                        : "r"(dslots_addr + (unsigned int)(16 * ((2 * s_12 + half_4) * 160 + tid))));
                }
                float tg_dx[8];
                float tg_dk[8];
                #pragma unroll
                for (int m_3 = 0; m_3 < 4; m_3++) {
                    float2 tg_hv = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_12 + m_3]));
                    float2 tg_kv = cake_bf16x2_to_f32x2(__as_bf16x2(kw[4 * s_12 + m_3]));
                    float2 tg_gv = cake_bf16x2_to_f32x2(__as_bf16x2(gw[4 * s_12 + m_3]));
                    float2 tg_wv = make_float2(wf[8 * s_12 + 2 * m_3], wf[8 * s_12 + 2 * m_3 + 1]);
                    float2 tg_ox = fma_f32x2_rn_noftz(tg_c2, fma_f32x2_rn_noftz(tg_h2, tg_hv, mul_f32x2_noftz(tg_wv, tg_kv)), tg_gv);
                    float2 _f32x2_mul_b_83 = fma_f32x2_rn_noftz(tg_k2, tg_kv, mul_f32x2_noftz(tg_wv, tg_hv));
                    float2 tg_ok;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_ok) : "l"(*(const unsigned long long*)&tg_c2), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
                    float2 tg_ov = fma_f32x2_rn_noftz(tg_g2, tg_gv, make_float2(tg_dv[2 * m_3], tg_dv[2 * m_3 + 1]));
                    float2 tg_hk;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_hk) : "l"(*(const unsigned long long*)&tg_hv), "l"(*(const unsigned long long*)&tg_kv));
                    float2 tg_ow = fma_f32x2_rn_noftz(tg_c2, tg_hk, make_float2(dwa[8 * s_12 + 2 * m_3], dwa[8 * s_12 + 2 * m_3 + 1]));
                    tg_dx[2 * m_3] = tg_ox.x;
                    tg_dk[2 * m_3] = tg_ok.x;
                    tg_dv[2 * m_3] = tg_ov.x;
                    dwa[8 * s_12 + 2 * m_3] = tg_ow.x;
                    tg_dx[2 * m_3 + 1] = tg_ox.y;
                    tg_dk[2 * m_3 + 1] = tg_ok.y;
                    tg_dv[2 * m_3 + 1] = tg_ov.y;
                    dwa[8 * s_12 + 2 * m_3 + 1] = tg_ow.y;
                }
                unsigned int tg_px[4];
                unsigned int tg_pk[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tg_dx[_lp*2 + 0], tg_dx[_lp*2+1 + 0]));
                    tg_px[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tg_dk[_lp*2 + 0], tg_dk[_lp*2+1 + 0]));
                    tg_pk[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(dx + (tg_x + (long long)(s_12 * 5120)))[0] = reinterpret_cast<int4*>(tg_px)[0];
                reinterpret_cast<int4*>(dkey + (tg_k + (long long)s_12 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(tg_pk)[0];
                #pragma unroll
                for (int half_5 = 0; half_5 < 2; half_5++) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dslots_addr + (unsigned int)(16 * ((2 * s_12 + half_5) * 160 + tid))), "f"(dwa[8 * s_12 + 4 * half_5]), "f"(dwa[8 * s_12 + 4 * half_5 + 1]), "f"(dwa[8 * s_12 + 4 * half_5 + 2]), "f"(dwa[8 * s_12 + 4 * half_5 + 3]) : "memory");
                }
            }
            unsigned int tg_pv[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tg_dv[_lp*2 + 0], tg_dv[_lp*2+1 + 0]));
                tg_pv[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dvalue + tg_v)[0] = reinterpret_cast<int4*>(tg_pv)[0];
        } else {
            int pn_tok_1 = t + step;
            if (pn_tok_1 < T) {
                long long ls_at_3 = (long long)pn_tok_1 * 20480ll + (long long)col;
                #pragma unroll
                for (int s_13 = 0; s_13 < 4; s_13++) {
                    {
                        const uint4* _vptr_84 = reinterpret_cast<const uint4*>(dout + (ls_at_3 + (long long)(s_13 * 5120)) + 0);
                        uint4* _vdst_84 = reinterpret_cast<uint4*>(&gn[4 * s_13]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_84[_blk] = _vptr_84[_blk];
                        }
                    }
                }
                long long ls_at_4 = (long long)pn_tok_1 * 20480ll + (long long)col;
                #pragma unroll
                for (int s_14 = 0; s_14 < 4; s_14++) {
                    {
                        const uint4* _vptr_85 = reinterpret_cast<const uint4*>(x + (ls_at_4 + (long long)(s_14 * 5120)) + 0);
                        uint4* _vdst_85 = reinterpret_cast<uint4*>(&xn[4 * s_14]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_85[_blk] = _vptr_85[_blk];
                        }
                    }
                }
                long long lk_at_1 = (long long)pn_tok_1 * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_15 = 0; s_15 < 4; s_15++) {
                    {
                        const uint4* _vptr_86 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_15 * key_stride_p1) + 0);
                        uint4* _vdst_86 = reinterpret_cast<uint4*>(&kn[4 * s_15]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_86[_blk] = _vptr_86[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_87 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_87 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_87[_blk] = _vptr_87[_blk];
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
            for (int s_16 = 0; s_16 < 4; s_16++) {
                reinterpret_cast<int4*>(dx + (tg_x + (long long)(s_16 * 5120)))[0] = reinterpret_cast<int4*>(gw + (4 * s_16))[0];
                reinterpret_cast<int4*>(dkey + (tg_k + (long long)s_16 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(zw)[0];
            }
            reinterpret_cast<int4*>(dvalue + tg_v)[0] = reinterpret_cast<int4*>(zw)[0];
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 16; i_7++) {
            gw[i_7] = gn[i_7];
            xw[i_7] = xn[i_7];
            kw[i_7] = kn[i_7];
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 4; i_8++) {
            vw[i_8] = vn[i_8];
        }
        keep = keep_n;
    }
    #pragma unroll
    for (int s_17 = 0; s_17 < 4; s_17++) {
        #pragma unroll
        for (int half_6 = 0; half_6 < 2; half_6++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dwa[8 * s_17 + 4 * half_6])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_17 + 4 * half_6) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_17 + 4 * half_6) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_17 + 4 * half_6) + 3]))
                : "r"(dslots_addr + (unsigned int)(16 * ((2 * s_17 + half_6) * 160 + tid))));
        }
    }
    #pragma unroll
    for (int s_18 = 0; s_18 < 4; s_18++) {
        #pragma unroll
        for (int half_7 = 0; half_7 < 2; half_7++) {
            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dw_acc[s_18 * 5120 + col + 4 * half_7])), "f"(dwa[8 * s_18 + 4 * half_7]), "f"(dwa[8 * s_18 + 4 * half_7 + 1]), "f"(dwa[8 * s_18 + 4 * half_7 + 2]), "f"(dwa[8 * s_18 + 4 * half_7 + 3]) : "memory");
        }
    }
}

} // extern "C"
