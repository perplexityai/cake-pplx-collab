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
kernel_cake_attn_res_det_train_106c9acdde9f5ecbb8e8(__nv_bfloat16* __restrict__ v0, __nv_bfloat16* __restrict__ v1, __nv_bfloat16* __restrict__ v2, __nv_bfloat16* __restrict__ v3, __nv_bfloat16* __restrict__ v4, __nv_bfloat16* __restrict__ v5, __nv_bfloat16* __restrict__ v6, __nv_bfloat16* __restrict__ v7, __nv_bfloat16* __restrict__ v8, float* __restrict__ w, __nv_bfloat16* __restrict__ out, float* __restrict__ p, float* __restrict__ r, float* __restrict__ d, int T, int n, float eps)
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
    float* recv = reinterpret_cast<float*>(smem_raw + 2144);
    const int recv_addr = smem + 2144;
    unsigned int* stg = reinterpret_cast<unsigned int*>(smem_raw + 2784);
    const int stage_addr = smem + 2784;

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
    int jl = ((lane) < (8) ? (lane) : (8));
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
    unsigned int vw[36];
    float part[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
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
    if (4 < n) {
        int _min_9 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(v4 + ((long long)_min_9 * 7168ll + (long long)col) + 0);
            uint4* _vdst_10 = reinterpret_cast<uint4*>(&vw[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_10[_blk] = _vptr_10[_blk];
            }
        }
    }
    if (5 < n) {
        int _min_11 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(v5 + ((long long)_min_11 * 7168ll + (long long)col) + 0);
            uint4* _vdst_12 = reinterpret_cast<uint4*>(&vw[20]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_12[_blk] = _vptr_12[_blk];
            }
        }
    }
    if (6 < n) {
        int _min_13 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(v6 + ((long long)_min_13 * 7168ll + (long long)col) + 0);
            uint4* _vdst_14 = reinterpret_cast<uint4*>(&vw[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_14[_blk] = _vptr_14[_blk];
            }
        }
    }
    if (7 < n) {
        int _min_15 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(v7 + ((long long)_min_15 * 7168ll + (long long)col) + 0);
            uint4* _vdst_16 = reinterpret_cast<uint4*>(&vw[28]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_16[_blk] = _vptr_16[_blk];
            }
        }
    }
    if (8 < n) {
        int _min_17 = ((first) < (T - 1) ? (first) : (T - 1));
        {
            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(v8 + ((long long)_min_17 * 7168ll + (long long)col) + 0);
            uint4* _vdst_18 = reinterpret_cast<uint4*>(&vw[32]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_18[_blk] = _vptr_18[_blk];
            }
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    #pragma unroll 1
    for (int t = first; t < T; t += step) {
        part[0] = 0.0f;
        part[9] = 0.0f;
        if (0 < n) {
            float ts_s_0 = 0.0f;
            #pragma unroll
            for (int m_0 = 0; m_0 < 4; m_0++) {
                float _dot_19;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_19) : "r"(vw[m_0]), "r"(vw[m_0]), "f"(ts_s_0));
                ts_s_0 = _dot_19;
            }
            float2 ts_f_0 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_1 = 0; m_1 < 4; m_1++) {
                ts_f_0 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_1], wf[2 * m_1 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_1])), ts_f_0);
            }
            part[0] = ts_s_0;
            float _add_20;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(ts_f_0.x), "f"(ts_f_0.y));
            part[9] = _add_20;
        }
        part[1] = 0.0f;
        part[10] = 0.0f;
        if (1 < n) {
            float ts_s_1 = 0.0f;
            #pragma unroll
            for (int m_2 = 0; m_2 < 4; m_2++) {
                float _dot_21;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_21) : "r"(vw[4 + m_2]), "r"(vw[4 + m_2]), "f"(ts_s_1));
                ts_s_1 = _dot_21;
            }
            float2 ts_f_1 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_3 = 0; m_3 < 4; m_3++) {
                ts_f_1 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_3], wf[2 * m_3 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_3])), ts_f_1);
            }
            part[1] = ts_s_1;
            float _add_22;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(ts_f_1.x), "f"(ts_f_1.y));
            part[10] = _add_22;
        }
        part[2] = 0.0f;
        part[11] = 0.0f;
        if (2 < n) {
            float ts_s_2 = 0.0f;
            #pragma unroll
            for (int m_4 = 0; m_4 < 4; m_4++) {
                float _dot_23;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_23) : "r"(vw[8 + m_4]), "r"(vw[8 + m_4]), "f"(ts_s_2));
                ts_s_2 = _dot_23;
            }
            float2 ts_f_2 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_5 = 0; m_5 < 4; m_5++) {
                ts_f_2 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_5], wf[2 * m_5 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[8 + m_5])), ts_f_2);
            }
            part[2] = ts_s_2;
            float _add_24;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(ts_f_2.x), "f"(ts_f_2.y));
            part[11] = _add_24;
        }
        part[3] = 0.0f;
        part[12] = 0.0f;
        if (3 < n) {
            float ts_s_3 = 0.0f;
            #pragma unroll
            for (int m_6 = 0; m_6 < 4; m_6++) {
                float _dot_25;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_25) : "r"(vw[12 + m_6]), "r"(vw[12 + m_6]), "f"(ts_s_3));
                ts_s_3 = _dot_25;
            }
            float2 ts_f_3 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_7 = 0; m_7 < 4; m_7++) {
                ts_f_3 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_7], wf[2 * m_7 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[12 + m_7])), ts_f_3);
            }
            part[3] = ts_s_3;
            float _add_26;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(ts_f_3.x), "f"(ts_f_3.y));
            part[12] = _add_26;
        }
        part[4] = 0.0f;
        part[13] = 0.0f;
        if (4 < n) {
            float ts_s_4 = 0.0f;
            #pragma unroll
            for (int m_8 = 0; m_8 < 4; m_8++) {
                float _dot_27;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_27) : "r"(vw[16 + m_8]), "r"(vw[16 + m_8]), "f"(ts_s_4));
                ts_s_4 = _dot_27;
            }
            float2 ts_f_4 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_9 = 0; m_9 < 4; m_9++) {
                ts_f_4 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_9], wf[2 * m_9 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[16 + m_9])), ts_f_4);
            }
            part[4] = ts_s_4;
            float _add_28;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(ts_f_4.x), "f"(ts_f_4.y));
            part[13] = _add_28;
        }
        part[5] = 0.0f;
        part[14] = 0.0f;
        if (5 < n) {
            float ts_s_5 = 0.0f;
            #pragma unroll
            for (int m_10 = 0; m_10 < 4; m_10++) {
                float _dot_29;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_29) : "r"(vw[20 + m_10]), "r"(vw[20 + m_10]), "f"(ts_s_5));
                ts_s_5 = _dot_29;
            }
            float2 ts_f_5 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_11 = 0; m_11 < 4; m_11++) {
                ts_f_5 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_11], wf[2 * m_11 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[20 + m_11])), ts_f_5);
            }
            part[5] = ts_s_5;
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(ts_f_5.x), "f"(ts_f_5.y));
            part[14] = _add_30;
        }
        part[6] = 0.0f;
        part[15] = 0.0f;
        if (6 < n) {
            float ts_s_6 = 0.0f;
            #pragma unroll
            for (int m_12 = 0; m_12 < 4; m_12++) {
                float _dot_31;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_31) : "r"(vw[24 + m_12]), "r"(vw[24 + m_12]), "f"(ts_s_6));
                ts_s_6 = _dot_31;
            }
            float2 ts_f_6 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_13 = 0; m_13 < 4; m_13++) {
                ts_f_6 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_13], wf[2 * m_13 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[24 + m_13])), ts_f_6);
            }
            part[6] = ts_s_6;
            float _add_32;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(ts_f_6.x), "f"(ts_f_6.y));
            part[15] = _add_32;
        }
        part[7] = 0.0f;
        part[16] = 0.0f;
        if (7 < n) {
            float ts_s_7 = 0.0f;
            #pragma unroll
            for (int m_14 = 0; m_14 < 4; m_14++) {
                float _dot_33;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_33) : "r"(vw[28 + m_14]), "r"(vw[28 + m_14]), "f"(ts_s_7));
                ts_s_7 = _dot_33;
            }
            float2 ts_f_7 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_15 = 0; m_15 < 4; m_15++) {
                ts_f_7 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_15], wf[2 * m_15 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[28 + m_15])), ts_f_7);
            }
            part[7] = ts_s_7;
            float _add_34;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(ts_f_7.x), "f"(ts_f_7.y));
            part[16] = _add_34;
        }
        part[8] = 0.0f;
        part[17] = 0.0f;
        if (8 < n) {
            float ts_s_8 = 0.0f;
            #pragma unroll
            for (int m_16 = 0; m_16 < 4; m_16++) {
                float _dot_35;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_35) : "r"(vw[32 + m_16]), "r"(vw[32 + m_16]), "f"(ts_s_8));
                ts_s_8 = _dot_35;
            }
            float2 ts_f_8 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_17 = 0; m_17 < 4; m_17++) {
                ts_f_8 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_17], wf[2 * m_17 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[32 + m_17])), ts_f_8);
            }
            part[8] = ts_s_8;
            float _add_36;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(ts_f_8.x), "f"(ts_f_8.y));
            part[17] = _add_36;
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
            if (4 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (896 + tid))), "l"(v4 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            if (5 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (1120 + tid))), "l"(v5 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            if (6 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (1344 + tid))), "l"(v6 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            if (7 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (1568 + tid))), "l"(v7 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            if (8 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (1792 + tid))), "l"(v8 + ((long long)(t + step) * 7168ll + (long long)col)));
            }
            asm volatile("cp.async.commit_group;");
        }
        int xp_slot = xk % 2;
        int xp_row = xp_slot * 140 + warp * 20;
        #pragma unroll
        for (int i_0 = 0; i_0 < 16; i_0++) {
            int ws_upper_0 = lane & 16;
            float ws_keep_0 = ((ws_upper_0 != 0) ? part[16 + i_0] : part[i_0]);
            float ws_send_0 = ((ws_upper_0 != 0) ? part[i_0] : part[16 + i_0]);
            float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
            part[i_0] = ws_keep_0 + _shfl_xor_sync_37;
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
            int ws_upper_1 = lane & 8;
            float ws_keep_1 = ((ws_upper_1 != 0) ? part[8 + i_1] : part[i_1]);
            float ws_send_1 = ((ws_upper_1 != 0) ? part[i_1] : part[8 + i_1]);
            float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
            part[i_1] = ws_keep_1 + _shfl_xor_sync_38;
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 4; i_2++) {
            int ws_upper_2 = lane & 4;
            float ws_keep_2 = ((ws_upper_2 != 0) ? part[4 + i_2] : part[i_2]);
            float ws_send_2 = ((ws_upper_2 != 0) ? part[i_2] : part[4 + i_2]);
            float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
            part[i_2] = ws_keep_2 + _shfl_xor_sync_39;
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; i_3++) {
            int ws_upper_3 = lane & 2;
            float ws_keep_3 = ((ws_upper_3 != 0) ? part[2 + i_3] : part[i_3]);
            float ws_send_3 = ((ws_upper_3 != 0) ? part[i_3] : part[2 + i_3]);
            float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
            part[i_3] = ws_keep_3 + _shfl_xor_sync_40;
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 1; i_4++) {
            int ws_upper_4 = lane & 1;
            float ws_keep_4 = ((ws_upper_4 != 0) ? part[1 + i_4] : part[i_4]);
            float ws_send_4 = ((ws_upper_4 != 0) ? part[i_4] : part[1 + i_4]);
            float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, ws_send_4, 1);
            part[i_4] = ws_keep_4 + _shfl_xor_sync_41;
        }
        float ws_own = part[0];
        float xp_word = ws_own;
        if ((lane & 0) == 0) {
            if (lane < 20) {
                wtab[xp_row + lane] = xp_word;
            }
        }
        __syncthreads();
        if (warp == 0) {
            if (lane < 20) {
                int xp_job = lane % 5;
                int xp_dst = lane / 5;
                int xp_base = xp_slot * 140 + 4 * xp_job;
                unsigned int xp_to = recv_addr + (unsigned int)(4 * (xp_slot * 80 + rank * 20));
                float xp_s[4];
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    xp_s[c] = wtab[xp_base + c];
                }
                #pragma unroll
                for (int wi = 1; wi < 7; wi++) {
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 4; c_1++) {
                        xp_s[c_1] = xp_s[c_1] + wtab[xp_base + wi * 20 + c_1];
                    }
                }
                uint32_t _cluster_mapa_42;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_42) : "r"(xp_to + (unsigned int)(16 * xp_job)), "r"(xp_dst));
                uint32_t _mbar_peer_43;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_43) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_42), "r"(__float_as_uint(xp_s[0])), "r"(__float_as_uint(xp_s[1])), "r"(__float_as_uint(xp_s[2])), "r"(__float_as_uint(xp_s[3])), "r"(_mbar_peer_43) : "memory");
            }
        }
        if (tid == 0) {
            mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 320);
        }
        int _stage_xbar = (int)((unsigned int)xk % 2);
        int _parity_xbar = (int)((unsigned int)xk / 2 % 2);
        asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(xbar_addr + (_stage_xbar) * 8), "r"(uint32_t(_parity_xbar)), "r"(uint32_t(10000000)) : "memory");
        asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
        int xt_at = xk % 2 * 80 + jl;
        float xt_t = recv[xt_at];
        #pragma unroll
        for (int rk = 1; rk < 4; rk++) {
            xt_t = xt_t + recv[xt_at + rk * 20];
        }
        float tk_ss = xt_t;
        int xt_at_0 = xk % 2 * 80 + (9 + jl);
        float xt_t_0 = recv[xt_at_0];
        #pragma unroll
        for (int rk_1 = 1; rk_1 < 4; rk_1++) {
            xt_t_0 = xt_t_0 + recv[xt_at_0 + rk_1 * 20];
        }
        float tk_d = xt_t_0;
        xk += 1;
        float _fdiv_45 = __fdiv_rn(tk_ss, 7168.0f);
        float _sqrt_44;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_44) : "f"(_fdiv_45 + eps));
        float tk_r = __frcp_rn(_sqrt_44);
        float tk_s = tk_r * tk_d;
        float tk_m = ((act != 0) ? tk_s : -3.4e+38f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tk_m = max_noftz(tk_m, __shfl_xor_sync(0xFFFFFFFF, tk_m, offset));
        float _exp_46 = __nv_expf(tk_s - tk_m);
        float tk_e = ((act != 0) ? _exp_46 : 0.0f);
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
        for (int i_5 = 0; i_5 < 8; i_5++) {
            to_o[i_5] = 0.0f;
        }
        if (0 < n) {
            float tj_p_0 = __shfl_sync(0xFFFFFFFF, tk_p, 0);
            float2 tj_p2_0 = make_float2(tj_p_0, tj_p_0);
            #pragma unroll
            for (int m_18 = 0; m_18 < 4; m_18++) {
                float2 tj_o_0 = fma_f32x2_rn_noftz(tj_p2_0, cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_18])), make_float2(to_o[2 * m_18], to_o[2 * m_18 + 1]));
                to_o[2 * m_18] = tj_o_0.x;
                to_o[2 * m_18 + 1] = tj_o_0.y;
            }
        }
        if (1 < n) {
            float tj_p_1 = __shfl_sync(0xFFFFFFFF, tk_p, 1);
            float2 tj_p2_1 = make_float2(tj_p_1, tj_p_1);
            #pragma unroll
            for (int m_19 = 0; m_19 < 4; m_19++) {
                float2 tj_o_1 = fma_f32x2_rn_noftz(tj_p2_1, cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_19])), make_float2(to_o[2 * m_19], to_o[2 * m_19 + 1]));
                to_o[2 * m_19] = tj_o_1.x;
                to_o[2 * m_19 + 1] = tj_o_1.y;
            }
        }
        if (2 < n) {
            float tj_p_2 = __shfl_sync(0xFFFFFFFF, tk_p, 2);
            float2 tj_p2_2 = make_float2(tj_p_2, tj_p_2);
            #pragma unroll
            for (int m_20 = 0; m_20 < 4; m_20++) {
                float2 tj_o_2 = fma_f32x2_rn_noftz(tj_p2_2, cake_bf16x2_to_f32x2(__as_bf16x2(vw[8 + m_20])), make_float2(to_o[2 * m_20], to_o[2 * m_20 + 1]));
                to_o[2 * m_20] = tj_o_2.x;
                to_o[2 * m_20 + 1] = tj_o_2.y;
            }
        }
        if (3 < n) {
            float tj_p_3 = __shfl_sync(0xFFFFFFFF, tk_p, 3);
            float2 tj_p2_3 = make_float2(tj_p_3, tj_p_3);
            #pragma unroll
            for (int m_21 = 0; m_21 < 4; m_21++) {
                float2 tj_o_3 = fma_f32x2_rn_noftz(tj_p2_3, cake_bf16x2_to_f32x2(__as_bf16x2(vw[12 + m_21])), make_float2(to_o[2 * m_21], to_o[2 * m_21 + 1]));
                to_o[2 * m_21] = tj_o_3.x;
                to_o[2 * m_21 + 1] = tj_o_3.y;
            }
        }
        if (4 < n) {
            float tj_p_4 = __shfl_sync(0xFFFFFFFF, tk_p, 4);
            float2 tj_p2_4 = make_float2(tj_p_4, tj_p_4);
            #pragma unroll
            for (int m_22 = 0; m_22 < 4; m_22++) {
                float2 tj_o_4 = fma_f32x2_rn_noftz(tj_p2_4, cake_bf16x2_to_f32x2(__as_bf16x2(vw[16 + m_22])), make_float2(to_o[2 * m_22], to_o[2 * m_22 + 1]));
                to_o[2 * m_22] = tj_o_4.x;
                to_o[2 * m_22 + 1] = tj_o_4.y;
            }
        }
        if (5 < n) {
            float tj_p_5 = __shfl_sync(0xFFFFFFFF, tk_p, 5);
            float2 tj_p2_5 = make_float2(tj_p_5, tj_p_5);
            #pragma unroll
            for (int m_23 = 0; m_23 < 4; m_23++) {
                float2 tj_o_5 = fma_f32x2_rn_noftz(tj_p2_5, cake_bf16x2_to_f32x2(__as_bf16x2(vw[20 + m_23])), make_float2(to_o[2 * m_23], to_o[2 * m_23 + 1]));
                to_o[2 * m_23] = tj_o_5.x;
                to_o[2 * m_23 + 1] = tj_o_5.y;
            }
        }
        if (6 < n) {
            float tj_p_6 = __shfl_sync(0xFFFFFFFF, tk_p, 6);
            float2 tj_p2_6 = make_float2(tj_p_6, tj_p_6);
            #pragma unroll
            for (int m_24 = 0; m_24 < 4; m_24++) {
                float2 tj_o_6 = fma_f32x2_rn_noftz(tj_p2_6, cake_bf16x2_to_f32x2(__as_bf16x2(vw[24 + m_24])), make_float2(to_o[2 * m_24], to_o[2 * m_24 + 1]));
                to_o[2 * m_24] = tj_o_6.x;
                to_o[2 * m_24 + 1] = tj_o_6.y;
            }
        }
        if (7 < n) {
            float tj_p_7 = __shfl_sync(0xFFFFFFFF, tk_p, 7);
            float2 tj_p2_7 = make_float2(tj_p_7, tj_p_7);
            #pragma unroll
            for (int m_25 = 0; m_25 < 4; m_25++) {
                float2 tj_o_7 = fma_f32x2_rn_noftz(tj_p2_7, cake_bf16x2_to_f32x2(__as_bf16x2(vw[28 + m_25])), make_float2(to_o[2 * m_25], to_o[2 * m_25 + 1]));
                to_o[2 * m_25] = tj_o_7.x;
                to_o[2 * m_25 + 1] = tj_o_7.y;
            }
        }
        if (8 < n) {
            float tj_p_8 = __shfl_sync(0xFFFFFFFF, tk_p, 8);
            float2 tj_p2_8 = make_float2(tj_p_8, tj_p_8);
            #pragma unroll
            for (int m_26 = 0; m_26 < 4; m_26++) {
                float2 tj_o_8 = fma_f32x2_rn_noftz(tj_p2_8, cake_bf16x2_to_f32x2(__as_bf16x2(vw[32 + m_26])), make_float2(to_o[2 * m_26], to_o[2 * m_26 + 1]));
                to_o[2 * m_26] = tj_o_8.x;
                to_o[2 * m_26 + 1] = tj_o_8.y;
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
            if (4 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[16])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (896 + tid))));
            }
            if (5 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[20])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (1120 + tid))));
            }
            if (6 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[24])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (1344 + tid))));
            }
            if (7 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[28])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (1568 + tid))));
            }
            if (8 < n) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vw[32])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (1792 + tid))));
            }
        }
    }
}

} // extern "C"
