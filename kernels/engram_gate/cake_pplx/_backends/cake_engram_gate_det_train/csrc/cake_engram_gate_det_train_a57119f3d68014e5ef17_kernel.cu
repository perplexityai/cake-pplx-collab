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
kernel_cake_engram_gate_det_train_a57119f3d68014e5ef17(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, float* __restrict__ w, bool* __restrict__ mask, __nv_bfloat16* __restrict__ out, int T, float eps, float clamp, long long key_stride_p0, long long key_stride_p1, long long value_stride_p0)
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
    float* gtab = reinterpret_cast<float*>(smem_raw + 22944);
    const int gtab_addr = smem + 22944;

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
    float plain[8];
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
    int tk_keep = 1;
    if ((unsigned long long)mask != 0ull) {
        if (!mask[ftok]) {
            tk_keep = 0;
        }
    }
    keep = tk_keep;
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    #pragma unroll
    for (int s_3 = 0; s_3 < 4; s_3++) {
        #pragma unroll
        for (int half_1 = 0; half_1 < 2; half_1++) {
            unsigned int fw_at = (unsigned int)(16 * ((2 * s_3 + half_1) * 160 + tid));
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(wslots_addr + fw_at), "f"(wf[8 * s_3 + 4 * half_1]), "f"(wf[8 * s_3 + 4 * half_1 + 1]), "f"(wf[8 * s_3 + 4 * half_1 + 2]), "f"(wf[8 * s_3 + 4 * half_1 + 3]) : "memory");
        }
    }
    #pragma unroll 1
    for (int t = first; t < T; t += step) {
        long long to_base = (long long)t * 20480ll + (long long)col;
        if (keep != 0) {
            #pragma unroll
            for (int s_4 = 0; s_4 < 4; s_4++) {
                #pragma unroll
                for (int half_2 = 0; half_2 < 2; half_2++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_4 + 4 * half_2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_4 + 4 * half_2) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_4 + 4 * half_2) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_4 + 4 * half_2) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_4 + half_2) * 160 + tid))));
                }
                float ts_h = 0.0f;
                float ts_k = 0.0f;
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
                        : "=f"(_dot_4) : "r"(xw[4 * s_4 + m]), "r"(xw[4 * s_4 + m]), "f"(ts_h));
                    ts_h = _dot_4;
                    float _dot_5;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_5) : "r"(kw[4 * s_4 + m]), "r"(kw[4 * s_4 + m]), "f"(ts_k));
                    ts_k = _dot_5;
                }
                plain[s_4] = ts_h;
                plain[4 + s_4] = ts_k;
                float ts_s = 0.0f;
                float ts_e = 0.0f;
                float ts_hk_0[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_0[0]) : "h"(static_cast<uint16_t>((xw[4 * s_4]))), "h"(static_cast<uint16_t>((kw[4 * s_4]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_0[1]) : "h"(static_cast<uint16_t>((xw[4 * s_4]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_4]) >> 16)));
                float ts_w_0 = wf[8 * s_4];
                float ts_p_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_0) : "f"(ts_w_0), "f"(ts_hk_0[0]));
                float _neg_6;
                asm("neg.f32 %0, %1;" : "=f"(_neg_6) : "f"(ts_p_0));
                float ts_q_0;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_0) : "f"(ts_w_0), "f"(ts_hk_0[0]), "f"(_neg_6));
                float _add_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(0.0f), "f"(ts_p_0));
                ts_s = _add_7;
                float _add_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(0.0f), "f"(ts_q_0));
                ts_e = _add_8;
                float ts_w_1 = wf[8 * s_4 + 1];
                float ts_p_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_1) : "f"(ts_w_1), "f"(ts_hk_0[1]));
                float _neg_9;
                asm("neg.f32 %0, %1;" : "=f"(_neg_9) : "f"(ts_p_1));
                float ts_q_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_1) : "f"(ts_w_1), "f"(ts_hk_0[1]), "f"(_neg_9));
                float ts_s_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(ts_s), "f"(ts_p_1));
                float ts_bb;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s_0), "f"(ts_s));
                float _sub_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(ts_s_0), "f"(ts_bb));
                float _sub_10;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(ts_s), "f"(_sub_11));
                float _sub_12;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(ts_p_1), "f"(ts_bb));
                float ts_e_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_10), "f"(_sub_12));
                ts_s = ts_s_0;
                float _add_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(ts_q_1), "f"(ts_e_0));
                float _add_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(ts_e), "f"(_add_14));
                ts_e = _add_13;
                float ts_hk_1[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_1[0]) : "h"(static_cast<uint16_t>((xw[4 * s_4 + 1]))), "h"(static_cast<uint16_t>((kw[4 * s_4 + 1]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_1[1]) : "h"(static_cast<uint16_t>((xw[4 * s_4 + 1]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_4 + 1]) >> 16)));
                float ts_w_2 = wf[8 * s_4 + 2];
                float ts_p_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_2) : "f"(ts_w_2), "f"(ts_hk_1[0]));
                float _neg_15;
                asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(ts_p_2));
                float ts_q_2;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_2) : "f"(ts_w_2), "f"(ts_hk_1[0]), "f"(_neg_15));
                float ts_s_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(ts_s), "f"(ts_p_2));
                float ts_bb_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_1), "f"(ts_s));
                float _sub_17;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(ts_s_1), "f"(ts_bb_0));
                float _sub_16;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(ts_s), "f"(_sub_17));
                float _sub_18;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(ts_p_2), "f"(ts_bb_0));
                float ts_e_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_16), "f"(_sub_18));
                ts_s = ts_s_1;
                float _add_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(ts_q_2), "f"(ts_e_1));
                float _add_19;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(ts_e), "f"(_add_20));
                ts_e = _add_19;
                float ts_w_3 = wf[8 * s_4 + 2 + 1];
                float ts_p_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_3) : "f"(ts_w_3), "f"(ts_hk_1[1]));
                float _neg_21;
                asm("neg.f32 %0, %1;" : "=f"(_neg_21) : "f"(ts_p_3));
                float ts_q_3;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_3) : "f"(ts_w_3), "f"(ts_hk_1[1]), "f"(_neg_21));
                float ts_s_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(ts_s), "f"(ts_p_3));
                float ts_bb_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_2), "f"(ts_s));
                float _sub_23;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(ts_s_2), "f"(ts_bb_1));
                float _sub_22;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(ts_s), "f"(_sub_23));
                float _sub_24;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(ts_p_3), "f"(ts_bb_1));
                float ts_e_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_22), "f"(_sub_24));
                ts_s = ts_s_2;
                float _add_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(ts_q_3), "f"(ts_e_2));
                float _add_25;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(ts_e), "f"(_add_26));
                ts_e = _add_25;
                float ts_hk_2[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_2[0]) : "h"(static_cast<uint16_t>((xw[4 * s_4 + 2]))), "h"(static_cast<uint16_t>((kw[4 * s_4 + 2]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_2[1]) : "h"(static_cast<uint16_t>((xw[4 * s_4 + 2]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_4 + 2]) >> 16)));
                float ts_w_4 = wf[8 * s_4 + 4];
                float ts_p_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_4) : "f"(ts_w_4), "f"(ts_hk_2[0]));
                float _neg_27;
                asm("neg.f32 %0, %1;" : "=f"(_neg_27) : "f"(ts_p_4));
                float ts_q_4;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_4) : "f"(ts_w_4), "f"(ts_hk_2[0]), "f"(_neg_27));
                float ts_s_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3) : "f"(ts_s), "f"(ts_p_4));
                float ts_bb_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_3), "f"(ts_s));
                float _sub_29;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(ts_s_3), "f"(ts_bb_2));
                float _sub_28;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(ts_s), "f"(_sub_29));
                float _sub_30;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(ts_p_4), "f"(ts_bb_2));
                float ts_e_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_28), "f"(_sub_30));
                ts_s = ts_s_3;
                float _add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(ts_q_4), "f"(ts_e_3));
                float _add_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(ts_e), "f"(_add_32));
                ts_e = _add_31;
                float ts_w_5 = wf[8 * s_4 + 4 + 1];
                float ts_p_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_5) : "f"(ts_w_5), "f"(ts_hk_2[1]));
                float _neg_33;
                asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(ts_p_5));
                float ts_q_5;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_5) : "f"(ts_w_5), "f"(ts_hk_2[1]), "f"(_neg_33));
                float ts_s_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_4) : "f"(ts_s), "f"(ts_p_5));
                float ts_bb_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3) : "f"(ts_s_4), "f"(ts_s));
                float _sub_35;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(ts_s_4), "f"(ts_bb_3));
                float _sub_34;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(ts_s), "f"(_sub_35));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(ts_p_5), "f"(ts_bb_3));
                float ts_e_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_34), "f"(_sub_36));
                ts_s = ts_s_4;
                float _add_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(ts_q_5), "f"(ts_e_4));
                float _add_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(ts_e), "f"(_add_38));
                ts_e = _add_37;
                float ts_hk_3[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_3[0]) : "h"(static_cast<uint16_t>((xw[4 * s_4 + 3]))), "h"(static_cast<uint16_t>((kw[4 * s_4 + 3]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_3[1]) : "h"(static_cast<uint16_t>((xw[4 * s_4 + 3]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_4 + 3]) >> 16)));
                float ts_w_6 = wf[8 * s_4 + 6];
                float ts_p_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_6) : "f"(ts_w_6), "f"(ts_hk_3[0]));
                float _neg_39;
                asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(ts_p_6));
                float ts_q_6;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_6) : "f"(ts_w_6), "f"(ts_hk_3[0]), "f"(_neg_39));
                float ts_s_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_5) : "f"(ts_s), "f"(ts_p_6));
                float ts_bb_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_4) : "f"(ts_s_5), "f"(ts_s));
                float _sub_41;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(ts_s_5), "f"(ts_bb_4));
                float _sub_40;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(ts_s), "f"(_sub_41));
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(ts_p_6), "f"(ts_bb_4));
                float ts_e_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_5) : "f"(_sub_40), "f"(_sub_42));
                ts_s = ts_s_5;
                float _add_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(ts_q_6), "f"(ts_e_5));
                float _add_43;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(ts_e), "f"(_add_44));
                ts_e = _add_43;
                float ts_w_7 = wf[8 * s_4 + 6 + 1];
                float ts_p_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_7) : "f"(ts_w_7), "f"(ts_hk_3[1]));
                float _neg_45;
                asm("neg.f32 %0, %1;" : "=f"(_neg_45) : "f"(ts_p_7));
                float ts_q_7;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_7) : "f"(ts_w_7), "f"(ts_hk_3[1]), "f"(_neg_45));
                float ts_s_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_6) : "f"(ts_s), "f"(ts_p_7));
                float ts_bb_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_5) : "f"(ts_s_6), "f"(ts_s));
                float _sub_47;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(ts_s_6), "f"(ts_bb_5));
                float _sub_46;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(ts_s), "f"(_sub_47));
                float _sub_48;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(ts_p_7), "f"(ts_bb_5));
                float ts_e_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6) : "f"(_sub_46), "f"(_sub_48));
                ts_s = ts_s_6;
                float _add_50;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(ts_q_7), "f"(ts_e_6));
                float _add_49;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(ts_e), "f"(_add_50));
                ts_e = _add_49;
                ps[s_4] = ts_s;
                pe[s_4] = ts_e;
            }
            int pn_tok = t + step;
            if (pn_tok < T) {
                long long ls_at_0 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
                for (int s_5 = 0; s_5 < 4; s_5++) {
                    {
                        const uint4* _vptr_51 = reinterpret_cast<const uint4*>(x + (ls_at_0 + (long long)(s_5 * 5120)) + 0);
                        uint4* _vdst_51 = reinterpret_cast<uint4*>(&xn[4 * s_5]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_51[_blk] = _vptr_51[_blk];
                        }
                    }
                }
                long long lk_at_0 = (long long)pn_tok * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_6 = 0; s_6 < 4; s_6++) {
                    {
                        const uint4* _vptr_52 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_6 * key_stride_p1) + 0);
                        uint4* _vdst_52 = reinterpret_cast<uint4*>(&kn[4 * s_6]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_52[_blk] = _vptr_52[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_53 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_53 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_53[_blk] = _vptr_53[_blk];
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
            int xl_slot = xk % 2;
            int xl_row = xl_slot * 100 + warp * 16;
            #pragma unroll
            for (int i_0 = 0; i_0 < 4; i_0++) {
                int ws_upper_0 = lane & 16;
                float ws_keep_0 = ((ws_upper_0 != 0) ? plain[4 + i_0] : plain[i_0]);
                float ws_send_0 = ((ws_upper_0 != 0) ? plain[i_0] : plain[4 + i_0]);
                float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_54;
            }
            #pragma unroll
            for (int i_1 = 0; i_1 < 2; i_1++) {
                int ws_upper_1 = lane & 8;
                float ws_keep_1 = ((ws_upper_1 != 0) ? plain[2 + i_1] : plain[i_1]);
                float ws_send_1 = ((ws_upper_1 != 0) ? plain[i_1] : plain[2 + i_1]);
                float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1] = ws_keep_1 + _shfl_xor_sync_55;
            }
            #pragma unroll
            for (int i_2 = 0; i_2 < 1; i_2++) {
                int ws_upper_2 = lane & 4;
                float ws_keep_2 = ((ws_upper_2 != 0) ? plain[1 + i_2] : plain[i_2]);
                float ws_send_2 = ((ws_upper_2 != 0) ? plain[i_2] : plain[1 + i_2]);
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2] = ws_keep_2 + _shfl_xor_sync_56;
            }
            float ws_own = plain[0];
            float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 2);
            ws_own = ws_own + _shfl_xor_sync_57;
            float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_58;
            float xl_word = ws_own;
            if ((lane & 3) == 0) {
                if (lane >> 2 < 8) {
                    wtab[xl_row + (lane >> 2)] = xl_word;
                }
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 2; i_3++) {
                int wp_upper_0 = lane & 16;
                float wp_ks_0 = ((wp_upper_0 != 0) ? ps[2 + i_3] : ps[i_3]);
                float wp_ke_0 = ((wp_upper_0 != 0) ? pe[2 + i_3] : pe[i_3]);
                float wp_rs_0 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_0 != 0) ? ps[i_3] : ps[2 + i_3]), 16);
                float wp_re_0 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_0 != 0) ? pe[i_3] : pe[2 + i_3]), 16);
                float ts_s_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7) : "f"(wp_ks_0), "f"(wp_rs_0));
                float ts_bb_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6) : "f"(ts_s_7), "f"(wp_ks_0));
                float _sub_60;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(ts_s_7), "f"(ts_bb_6));
                float _sub_59;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(wp_ks_0), "f"(_sub_60));
                float _sub_61;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(wp_rs_0), "f"(ts_bb_6));
                float ts_e_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7) : "f"(_sub_59), "f"(_sub_61));
                wp_ks_0 = ts_s_7;
                float _add_63;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(wp_ke_0), "f"(wp_re_0));
                float _add_62;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(_add_63), "f"(ts_e_7));
                wp_ke_0 = _add_62;
                ps[i_3] = wp_ks_0;
                pe[i_3] = wp_ke_0;
            }
            #pragma unroll
            for (int i_4 = 0; i_4 < 1; i_4++) {
                int wp_upper_1 = lane & 8;
                float wp_ks_1 = ((wp_upper_1 != 0) ? ps[1 + i_4] : ps[i_4]);
                float wp_ke_1 = ((wp_upper_1 != 0) ? pe[1 + i_4] : pe[i_4]);
                float wp_rs_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? ps[i_4] : ps[1 + i_4]), 8);
                float wp_re_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? pe[i_4] : pe[1 + i_4]), 8);
                float ts_s_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8) : "f"(wp_ks_1), "f"(wp_rs_1));
                float ts_bb_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7) : "f"(ts_s_8), "f"(wp_ks_1));
                float _sub_65;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(ts_s_8), "f"(ts_bb_7));
                float _sub_64;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(wp_ks_1), "f"(_sub_65));
                float _sub_66;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(wp_rs_1), "f"(ts_bb_7));
                float ts_e_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8) : "f"(_sub_64), "f"(_sub_66));
                wp_ks_1 = ts_s_8;
                float _add_68;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(wp_ke_1), "f"(wp_re_1));
                float _add_67;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_add_68), "f"(ts_e_8));
                wp_ke_1 = _add_67;
                ps[i_4] = wp_ks_1;
                pe[i_4] = wp_ke_1;
            }
            float wp_s = ps[0];
            float wp_e = pe[0];
            float wp_bs_0 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 4);
            float wp_be_0 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 4);
            float ts_s_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_9) : "f"(wp_s), "f"(wp_bs_0));
            float ts_bb_8;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8) : "f"(ts_s_9), "f"(wp_s));
            float _sub_70;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(ts_s_9), "f"(ts_bb_8));
            float _sub_69;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(wp_s), "f"(_sub_70));
            float _sub_71;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(wp_bs_0), "f"(ts_bb_8));
            float ts_e_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9) : "f"(_sub_69), "f"(_sub_71));
            wp_s = ts_s_9;
            float _add_73;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(wp_e), "f"(wp_be_0));
            float _add_72;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_add_73), "f"(ts_e_9));
            wp_e = _add_72;
            float wp_bs_1 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 2);
            float wp_be_1 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 2);
            float ts_s_7_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7_1) : "f"(wp_s), "f"(wp_bs_1));
            float ts_bb_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6_1) : "f"(ts_s_7_1), "f"(wp_s));
            float _sub_75;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(ts_s_7_1), "f"(ts_bb_6_1));
            float _sub_74;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(wp_s), "f"(_sub_75));
            float _sub_76;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(wp_bs_1), "f"(ts_bb_6_1));
            float ts_e_7_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7_1) : "f"(_sub_74), "f"(_sub_76));
            wp_s = ts_s_7_1;
            float _add_78;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(wp_e), "f"(wp_be_1));
            float _add_77;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_add_78), "f"(ts_e_7_1));
            wp_e = _add_77;
            float wp_bs_2 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 1);
            float wp_be_2 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 1);
            float ts_s_8_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8_1) : "f"(wp_s), "f"(wp_bs_2));
            float ts_bb_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7_1) : "f"(ts_s_8_1), "f"(wp_s));
            float _sub_80;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(ts_s_8_1), "f"(ts_bb_7_1));
            float _sub_79;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(wp_s), "f"(_sub_80));
            float _sub_81;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(wp_bs_2), "f"(ts_bb_7_1));
            float ts_e_8_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8_1) : "f"(_sub_79), "f"(_sub_81));
            wp_s = ts_s_8_1;
            float _add_83;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(wp_e), "f"(wp_be_2));
            float _add_82;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_add_83), "f"(ts_e_8_1));
            wp_e = _add_82;
            if ((lane & 7) == 0) {
                wtab[xl_row + 8 + (lane >> 3)] = wp_s;
                wtab[xl_row + 12 + (lane >> 3)] = wp_e;
            }
            __syncthreads();
            if (warp == 0) {
                if (lane < 12) {
                    int xl_job = lane % 3;
                    int xl_dst = lane / 3;
                    int xl_base = xl_slot * 100 + 4 * xl_job;
                    unsigned int xl_to = recv_addr + (unsigned int)(4 * (xl_slot * 80 + rank * 16));
                    float xl_s[4];
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        xl_s[c] = wtab[xl_base + c];
                    }
                    if (xl_job < 2) {
                        #pragma unroll
                        for (int wi = 1; wi < 5; wi++) {
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 4; c_1++) {
                                xl_s[c_1] = xl_s[c_1] + wtab[xl_base + wi * 16 + c_1];
                            }
                        }
                    } else {
                        float xl_e[4];
                        #pragma unroll
                        for (int c_2 = 0; c_2 < 4; c_2++) {
                            float xl_fs = xl_s[c_2];
                            float xl_fe = wtab[xl_base + 4 + c_2];
                            #pragma unroll
                            for (int wi_1 = 1; wi_1 < 5; wi_1++) {
                                float ts_s_9_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_9_1) : "f"(xl_fs), "f"(wtab[xl_base + wi_1 * 16 + c_2]));
                                float ts_bb_8_1;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8_1) : "f"(ts_s_9_1), "f"(xl_fs));
                                float _sub_85;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(ts_s_9_1), "f"(ts_bb_8_1));
                                float _sub_84;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(xl_fs), "f"(_sub_85));
                                float _sub_86;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(wtab[xl_base + wi_1 * 16 + c_2]), "f"(ts_bb_8_1));
                                float ts_e_9_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9_1) : "f"(_sub_84), "f"(_sub_86));
                                xl_fs = ts_s_9_1;
                                float _add_88;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(xl_fe), "f"(wtab[xl_base + wi_1 * 16 + 4 + c_2]));
                                float _add_87;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_add_88), "f"(ts_e_9_1));
                                xl_fe = _add_87;
                            }
                            xl_s[c_2] = xl_fs;
                            xl_e[c_2] = xl_fe;
                        }
                        uint32_t _cluster_mapa_89;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_89) : "r"(xl_to + (unsigned int)(16 * (xl_job + 1))), "r"(xl_dst));
                        uint32_t _mbar_peer_90;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_90) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xl_dst));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_89), "r"(__float_as_uint(xl_e[0])), "r"(__float_as_uint(xl_e[1])), "r"(__float_as_uint(xl_e[2])), "r"(__float_as_uint(xl_e[3])), "r"(_mbar_peer_90) : "memory");
                    }
                    uint32_t _cluster_mapa_91;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_91) : "r"(xl_to + (unsigned int)(16 * xl_job)), "r"(xl_dst));
                    uint32_t _mbar_peer_92;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_92) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xl_dst));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_91), "r"(__float_as_uint(xl_s[0])), "r"(__float_as_uint(xl_s[1])), "r"(__float_as_uint(xl_s[2])), "r"(__float_as_uint(xl_s[3])), "r"(_mbar_peer_92) : "memory");
                }
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 256);
                }
                int _stage_xbar = (int)((unsigned int)xk % 2);
                int _parity_xbar = (int)((unsigned int)xk / 2 % 2);
                asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(xbar_addr + (_stage_xbar) * 8), "r"(uint32_t(_parity_xbar)), "r"(uint32_t(10000000)) : "memory");
                asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
                int xl_from = xl_slot * 80 + (lane & 3);
                float xl_tot[3];
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    float xl_t = recv[xl_from + 4 * q];
                    #pragma unroll
                    for (int r = 1; r < 4; r++) {
                        xl_t = xl_t + recv[xl_from + r * 16 + 4 * q];
                    }
                    xl_tot[q] = xl_t;
                }
                float xl_ts = recv[xl_from + 8];
                float xl_te = recv[xl_from + 12];
                #pragma unroll
                for (int r_1 = 1; r_1 < 4; r_1++) {
                    float ts_s_10;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_10) : "f"(xl_ts), "f"(recv[xl_from + r_1 * 16 + 8]));
                    float ts_bb_9;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_9) : "f"(ts_s_10), "f"(xl_ts));
                    float _sub_94;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(ts_s_10), "f"(ts_bb_9));
                    float _sub_93;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(xl_ts), "f"(_sub_94));
                    float _sub_95;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(recv[xl_from + r_1 * 16 + 8]), "f"(ts_bb_9));
                    float ts_e_10;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_10) : "f"(_sub_93), "f"(_sub_95));
                    xl_ts = ts_s_10;
                    float _add_97;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(xl_te), "f"(recv[xl_from + r_1 * 16 + 12]));
                    float _add_96;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_add_97), "f"(ts_e_10));
                    xl_te = _add_96;
                }
                float _add_98;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(xl_ts), "f"(xl_te));
                xl_tot[2] = _add_98;
                float _fdiv_100 = __fdiv_rn(xl_tot[0], 5120.0f);
                float _sqrt_99;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_99) : "f"(_fdiv_100 + eps));
                float sg_rh = __frcp_rn(_sqrt_99);
                float _fdiv_102 = __fdiv_rn(xl_tot[1], 5120.0f);
                float _sqrt_101;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_101) : "f"(_fdiv_102 + eps));
                float sg_rk = __frcp_rn(_sqrt_101);
                float sg_dot;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sg_dot) : "f"(mul_f32_rn_noftz(xl_tot[2], mul_f32_rn_noftz(sg_rh, sg_rk))), "f"(0.013975425f));
                float _abs_104;
                asm("abs.f32 %0, %1;" : "=f"(_abs_104) : "f"(sg_dot));
                float _max_103;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_103) : "f"(_abs_104), "f"(clamp));
                float sg_mag;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_103));
                float sg_u = __uint_as_float(__as_u32(sg_mag) & 2147483647 | __as_u32(sg_dot) & 2147483648u);
                float _neg_106;
                asm("neg.f32 %0, %1;" : "=f"(_neg_106) : "f"(sg_u));
                float _exp_105 = __nv_expf(_neg_106);
                float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_105);
                if (lane < 4) {
                    gtab[lane] = sg_gate;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int s_7 = 0; s_7 < 4; s_7++) {
                gates[s_7] = gtab[s_7];
            }
            xk += 1;
            #pragma unroll
            for (int s_8 = 0; s_8 < 4; s_8++) {
                float2 to_g2 = make_float2(gates[s_8], gates[s_8]);
                float to_out[8];
                #pragma unroll
                for (int m_1 = 0; m_1 < 4; m_1++) {
                    float2 to_o2 = fma_f32x2_rn_noftz(to_g2, cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_1])), cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_8 + m_1])));
                    to_out[2 * m_1] = to_o2.x;
                    to_out[2 * m_1 + 1] = to_o2.y;
                }
                unsigned int to_packed[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_out[_lp*2 + 0], to_out[_lp*2+1 + 0]));
                    to_packed[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (to_base + (long long)(s_8 * 5120)))[0] = reinterpret_cast<int4*>(to_packed)[0];
            }
        } else {
            int pn_tok_1 = t + step;
            if (pn_tok_1 < T) {
                long long ls_at_1 = (long long)pn_tok_1 * 20480ll + (long long)col;
                #pragma unroll
                for (int s_9 = 0; s_9 < 4; s_9++) {
                    {
                        const uint4* _vptr_107 = reinterpret_cast<const uint4*>(x + (ls_at_1 + (long long)(s_9 * 5120)) + 0);
                        uint4* _vdst_107 = reinterpret_cast<uint4*>(&xn[4 * s_9]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_107[_blk] = _vptr_107[_blk];
                        }
                    }
                }
                long long lk_at_1 = (long long)pn_tok_1 * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_10 = 0; s_10 < 4; s_10++) {
                    {
                        const uint4* _vptr_108 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_10 * key_stride_p1) + 0);
                        uint4* _vdst_108 = reinterpret_cast<uint4*>(&kn[4 * s_10]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_108[_blk] = _vptr_108[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_109 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_109 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_109[_blk] = _vptr_109[_blk];
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
            for (int s_11 = 0; s_11 < 4; s_11++) {
                reinterpret_cast<int4*>(out + (to_base + (long long)(s_11 * 5120)))[0] = reinterpret_cast<int4*>(xw + (4 * s_11))[0];
            }
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            xw[i] = xn[i];
            kw[i] = kn[i];
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; i_5++) {
            vw[i_5] = vn[i_5];
        }
        keep = keep_n;
    }
}

} // extern "C"
