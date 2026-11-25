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

#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

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
kernel_cake_attn_res_train_cdd21e6bc5cb6ddca3a7(__nv_bfloat16* __restrict__ v0, __nv_bfloat16* __restrict__ v1, __nv_bfloat16* __restrict__ v2, __nv_bfloat16* __restrict__ v3, __nv_bfloat16* __restrict__ v4, __nv_bfloat16* __restrict__ v5, __nv_bfloat16* __restrict__ v6, __nv_bfloat16* __restrict__ v7, __nv_bfloat16* __restrict__ v8, float* __restrict__ w, __nv_bfloat16* __restrict__ out, float* __restrict__ p, float* __restrict__ r, float* __restrict__ d, int T, int n, float eps)
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
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* wtab = reinterpret_cast<float*>(smem_raw + 0);
    const int wtab_addr = smem + 0;
    unsigned int* stg = reinterpret_cast<unsigned int*>(smem_raw + 448);
    const int stage_addr = smem + 448;

    // === Task calls (declaration order) ===
    int first = (int)cluster_id * 4 + cta_rank;
    int step = (int)num_clusters * 4;
    int base = tid * 8;
    int jl = ((lane) < (3) ? (lane) : (3));
    int act = ((lane < n) ? 1 : 0);
    float wf[32];
    #pragma unroll
    for (int half_0 = 0; half_0 < 2; half_0++) {
        {
            unsigned _v4_0_0;
            unsigned _v4_0_1;
            unsigned _v4_0_2;
            unsigned _v4_0_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(w + (base + 4 * half_0) + (0))) : "memory");
            wf[4 * half_0 + 0] = __uint_as_float(_v4_0_0);
            wf[4 * half_0 + 1] = __uint_as_float(_v4_0_1);
            wf[4 * half_0 + 2] = __uint_as_float(_v4_0_2);
            wf[4 * half_0 + 3] = __uint_as_float(_v4_0_3);
        }
    }
    #pragma unroll
    for (int half_1 = 0; half_1 < 2; half_1++) {
        {
            unsigned _v4_1_0;
            unsigned _v4_1_1;
            unsigned _v4_1_2;
            unsigned _v4_1_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(w + (base + (1792 + 4 * half_1)) + (0))) : "memory");
            wf[8 + 4 * half_1 + 0] = __uint_as_float(_v4_1_0);
            wf[8 + 4 * half_1 + 1] = __uint_as_float(_v4_1_1);
            wf[8 + 4 * half_1 + 2] = __uint_as_float(_v4_1_2);
            wf[8 + 4 * half_1 + 3] = __uint_as_float(_v4_1_3);
        }
    }
    #pragma unroll
    for (int half_2 = 0; half_2 < 2; half_2++) {
        {
            unsigned _v4_2_0;
            unsigned _v4_2_1;
            unsigned _v4_2_2;
            unsigned _v4_2_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(w + (base + (3584 + 4 * half_2)) + (0))) : "memory");
            wf[16 + 4 * half_2 + 0] = __uint_as_float(_v4_2_0);
            wf[16 + 4 * half_2 + 1] = __uint_as_float(_v4_2_1);
            wf[16 + 4 * half_2 + 2] = __uint_as_float(_v4_2_2);
            wf[16 + 4 * half_2 + 3] = __uint_as_float(_v4_2_3);
        }
    }
    #pragma unroll
    for (int half_3 = 0; half_3 < 2; half_3++) {
        {
            unsigned _v4_3_0;
            unsigned _v4_3_1;
            unsigned _v4_3_2;
            unsigned _v4_3_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(w + (base + (5376 + 4 * half_3)) + (0))) : "memory");
            wf[24 + 4 * half_3 + 0] = __uint_as_float(_v4_3_0);
            wf[24 + 4 * half_3 + 1] = __uint_as_float(_v4_3_1);
            wf[24 + 4 * half_3 + 2] = __uint_as_float(_v4_3_2);
            wf[24 + 4 * half_3 + 3] = __uint_as_float(_v4_3_3);
        }
    }
    unsigned int vw[64];
    float pj[4];
    float part[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        part[i] = 0.0f;
    }
    int xs = 0;
    if (first < T) {
        if (0 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * tid)), "l"(v0 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (224 + tid))), "l"(v0 + ((long long)first * 7168ll + (long long)(base + 1792))));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (448 + tid))), "l"(v0 + ((long long)first * 7168ll + (long long)(base + 3584))));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (672 + tid))), "l"(v0 + ((long long)first * 7168ll + (long long)(base + 5376))));
        }
        if (1 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (896 + tid))), "l"(v1 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (1120 + tid))), "l"(v1 + ((long long)first * 7168ll + (long long)(base + 1792))));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (1344 + tid))), "l"(v1 + ((long long)first * 7168ll + (long long)(base + 3584))));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (1568 + tid))), "l"(v1 + ((long long)first * 7168ll + (long long)(base + 5376))));
        }
        if (2 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (1792 + tid))), "l"(v2 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (2016 + tid))), "l"(v2 + ((long long)first * 7168ll + (long long)(base + 1792))));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (2240 + tid))), "l"(v2 + ((long long)first * 7168ll + (long long)(base + 3584))));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (2464 + tid))), "l"(v2 + ((long long)first * 7168ll + (long long)(base + 5376))));
        }
        if (3 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (2688 + tid))), "l"(v3 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (2912 + tid))), "l"(v3 + ((long long)first * 7168ll + (long long)(base + 1792))));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (3136 + tid))), "l"(v3 + ((long long)first * 7168ll + (long long)(base + 3584))));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (3360 + tid))), "l"(v3 + ((long long)first * 7168ll + (long long)(base + 5376))));
        }
        asm volatile("cp.async.commit_group;");
    }
    #pragma unroll 1
    for (int t = first; t < T; t += step) {
        asm volatile("cp.async.wait_group 0;");
        if (0 < n) {
            int ut_piece_0 = tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_0)));
            int ut_piece_1 = 224 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[4])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_1)));
            int ut_piece_2 = 448 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[8])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_2)));
            int ut_piece_3 = 672 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[12])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_3)));
        }
        if (1 < n) {
            int ut_piece_4 = 896 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[16])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_4)));
            int ut_piece_5 = 1120 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[20])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_5)));
            int ut_piece_6 = 1344 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[24])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_6)));
            int ut_piece_7 = 1568 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[28])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_7)));
        }
        if (2 < n) {
            int ut_piece_8 = 1792 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[32])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_8)));
            int ut_piece_9 = 2016 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[36])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(36) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_9)));
            int ut_piece_10 = 2240 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[40])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(40) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_10)));
            int ut_piece_11 = 2464 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[44])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(44) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_11)));
        }
        if (3 < n) {
            int ut_piece_12 = 2688 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[48])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(48) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_12)));
            int ut_piece_13 = 2912 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[52])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(52) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_13)));
            int ut_piece_14 = 3136 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[56])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(56) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_14)));
            int ut_piece_15 = 3360 + tid;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[60])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(60) + 3]))
                : "r"(stage_addr + (unsigned int)(16 * ut_piece_15)));
        }
        part[0] = 0.0f;
        part[4] = 0.0f;
        if (0 < n) {
            float ts_s_0 = 0.0f;
            float2 ts_f_0 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_0 = 0; m_0 < 4; m_0++) {
                float _dot_4;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_4) : "r"(vw[m_0]), "r"(vw[m_0]), "f"(ts_s_0));
                ts_s_0 = _dot_4;
            }
            #pragma unroll
            for (int m_1 = 0; m_1 < 4; m_1++) {
                ts_f_0 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_1], wf[2 * m_1 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_1])), ts_f_0);
            }
            #pragma unroll
            for (int m_2 = 0; m_2 < 4; m_2++) {
                float _dot_5;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_5) : "r"(vw[4 + m_2]), "r"(vw[4 + m_2]), "f"(ts_s_0));
                ts_s_0 = _dot_5;
            }
            #pragma unroll
            for (int m_3 = 0; m_3 < 4; m_3++) {
                ts_f_0 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_3], wf[8 + 2 * m_3 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_3])), ts_f_0);
            }
            #pragma unroll
            for (int m_4 = 0; m_4 < 4; m_4++) {
                float _dot_6;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_6) : "r"(vw[8 + m_4]), "r"(vw[8 + m_4]), "f"(ts_s_0));
                ts_s_0 = _dot_6;
            }
            #pragma unroll
            for (int m_5 = 0; m_5 < 4; m_5++) {
                ts_f_0 = fma_f32x2_rn_noftz(make_float2(wf[16 + 2 * m_5], wf[16 + 2 * m_5 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[8 + m_5])), ts_f_0);
            }
            #pragma unroll
            for (int m_6 = 0; m_6 < 4; m_6++) {
                float _dot_7;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_7) : "r"(vw[12 + m_6]), "r"(vw[12 + m_6]), "f"(ts_s_0));
                ts_s_0 = _dot_7;
            }
            #pragma unroll
            for (int m_7 = 0; m_7 < 4; m_7++) {
                ts_f_0 = fma_f32x2_rn_noftz(make_float2(wf[24 + 2 * m_7], wf[24 + 2 * m_7 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[12 + m_7])), ts_f_0);
            }
            part[0] = ts_s_0;
            float _add_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(ts_f_0.x), "f"(ts_f_0.y));
            part[4] = _add_8;
        }
        part[1] = 0.0f;
        part[5] = 0.0f;
        if (1 < n) {
            float ts_s_1 = 0.0f;
            float2 ts_f_1 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_8 = 0; m_8 < 4; m_8++) {
                float _dot_9;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_9) : "r"(vw[16 + m_8]), "r"(vw[16 + m_8]), "f"(ts_s_1));
                ts_s_1 = _dot_9;
            }
            #pragma unroll
            for (int m_9 = 0; m_9 < 4; m_9++) {
                ts_f_1 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_9], wf[2 * m_9 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[16 + m_9])), ts_f_1);
            }
            #pragma unroll
            for (int m_10 = 0; m_10 < 4; m_10++) {
                float _dot_10;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_10) : "r"(vw[20 + m_10]), "r"(vw[20 + m_10]), "f"(ts_s_1));
                ts_s_1 = _dot_10;
            }
            #pragma unroll
            for (int m_11 = 0; m_11 < 4; m_11++) {
                ts_f_1 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_11], wf[8 + 2 * m_11 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[20 + m_11])), ts_f_1);
            }
            #pragma unroll
            for (int m_12 = 0; m_12 < 4; m_12++) {
                float _dot_11;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_11) : "r"(vw[24 + m_12]), "r"(vw[24 + m_12]), "f"(ts_s_1));
                ts_s_1 = _dot_11;
            }
            #pragma unroll
            for (int m_13 = 0; m_13 < 4; m_13++) {
                ts_f_1 = fma_f32x2_rn_noftz(make_float2(wf[16 + 2 * m_13], wf[16 + 2 * m_13 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[24 + m_13])), ts_f_1);
            }
            #pragma unroll
            for (int m_14 = 0; m_14 < 4; m_14++) {
                float _dot_12;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_12) : "r"(vw[28 + m_14]), "r"(vw[28 + m_14]), "f"(ts_s_1));
                ts_s_1 = _dot_12;
            }
            #pragma unroll
            for (int m_15 = 0; m_15 < 4; m_15++) {
                ts_f_1 = fma_f32x2_rn_noftz(make_float2(wf[24 + 2 * m_15], wf[24 + 2 * m_15 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[28 + m_15])), ts_f_1);
            }
            part[1] = ts_s_1;
            float _add_13;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(ts_f_1.x), "f"(ts_f_1.y));
            part[5] = _add_13;
        }
        part[2] = 0.0f;
        part[6] = 0.0f;
        if (2 < n) {
            float ts_s_2 = 0.0f;
            float2 ts_f_2 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_16 = 0; m_16 < 4; m_16++) {
                float _dot_14;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_14) : "r"(vw[32 + m_16]), "r"(vw[32 + m_16]), "f"(ts_s_2));
                ts_s_2 = _dot_14;
            }
            #pragma unroll
            for (int m_17 = 0; m_17 < 4; m_17++) {
                ts_f_2 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_17], wf[2 * m_17 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[32 + m_17])), ts_f_2);
            }
            #pragma unroll
            for (int m_18 = 0; m_18 < 4; m_18++) {
                float _dot_15;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_15) : "r"(vw[36 + m_18]), "r"(vw[36 + m_18]), "f"(ts_s_2));
                ts_s_2 = _dot_15;
            }
            #pragma unroll
            for (int m_19 = 0; m_19 < 4; m_19++) {
                ts_f_2 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_19], wf[8 + 2 * m_19 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[36 + m_19])), ts_f_2);
            }
            #pragma unroll
            for (int m_20 = 0; m_20 < 4; m_20++) {
                float _dot_16;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_16) : "r"(vw[40 + m_20]), "r"(vw[40 + m_20]), "f"(ts_s_2));
                ts_s_2 = _dot_16;
            }
            #pragma unroll
            for (int m_21 = 0; m_21 < 4; m_21++) {
                ts_f_2 = fma_f32x2_rn_noftz(make_float2(wf[16 + 2 * m_21], wf[16 + 2 * m_21 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[40 + m_21])), ts_f_2);
            }
            #pragma unroll
            for (int m_22 = 0; m_22 < 4; m_22++) {
                float _dot_17;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_17) : "r"(vw[44 + m_22]), "r"(vw[44 + m_22]), "f"(ts_s_2));
                ts_s_2 = _dot_17;
            }
            #pragma unroll
            for (int m_23 = 0; m_23 < 4; m_23++) {
                ts_f_2 = fma_f32x2_rn_noftz(make_float2(wf[24 + 2 * m_23], wf[24 + 2 * m_23 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[44 + m_23])), ts_f_2);
            }
            part[2] = ts_s_2;
            float _add_18;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(ts_f_2.x), "f"(ts_f_2.y));
            part[6] = _add_18;
        }
        part[3] = 0.0f;
        part[7] = 0.0f;
        if (3 < n) {
            float ts_s_3 = 0.0f;
            float2 ts_f_3 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int m_24 = 0; m_24 < 4; m_24++) {
                float _dot_19;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_19) : "r"(vw[48 + m_24]), "r"(vw[48 + m_24]), "f"(ts_s_3));
                ts_s_3 = _dot_19;
            }
            #pragma unroll
            for (int m_25 = 0; m_25 < 4; m_25++) {
                ts_f_3 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_25], wf[2 * m_25 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[48 + m_25])), ts_f_3);
            }
            #pragma unroll
            for (int m_26 = 0; m_26 < 4; m_26++) {
                float _dot_20;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_20) : "r"(vw[52 + m_26]), "r"(vw[52 + m_26]), "f"(ts_s_3));
                ts_s_3 = _dot_20;
            }
            #pragma unroll
            for (int m_27 = 0; m_27 < 4; m_27++) {
                ts_f_3 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_27], wf[8 + 2 * m_27 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[52 + m_27])), ts_f_3);
            }
            #pragma unroll
            for (int m_28 = 0; m_28 < 4; m_28++) {
                float _dot_21;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_21) : "r"(vw[56 + m_28]), "r"(vw[56 + m_28]), "f"(ts_s_3));
                ts_s_3 = _dot_21;
            }
            #pragma unroll
            for (int m_29 = 0; m_29 < 4; m_29++) {
                ts_f_3 = fma_f32x2_rn_noftz(make_float2(wf[16 + 2 * m_29], wf[16 + 2 * m_29 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[56 + m_29])), ts_f_3);
            }
            #pragma unroll
            for (int m_30 = 0; m_30 < 4; m_30++) {
                float _dot_22;
                asm(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_dot_22) : "r"(vw[60 + m_30]), "r"(vw[60 + m_30]), "f"(ts_s_3));
                ts_s_3 = _dot_22;
            }
            #pragma unroll
            for (int m_31 = 0; m_31 < 4; m_31++) {
                ts_f_3 = fma_f32x2_rn_noftz(make_float2(wf[24 + 2 * m_31], wf[24 + 2 * m_31 + 1]), cake_bf16x2_to_f32x2(__as_bf16x2(vw[60 + m_31])), ts_f_3);
            }
            part[3] = ts_s_3;
            float _add_23;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(ts_f_3.x), "f"(ts_f_3.y));
            part[7] = _add_23;
        }
        if (t + step < T) {
            if (0 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * tid)), "l"(v0 + ((long long)(t + step) * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (224 + tid))), "l"(v0 + ((long long)(t + step) * 7168ll + (long long)(base + 1792))));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (448 + tid))), "l"(v0 + ((long long)(t + step) * 7168ll + (long long)(base + 3584))));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (672 + tid))), "l"(v0 + ((long long)(t + step) * 7168ll + (long long)(base + 5376))));
            }
            if (1 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (896 + tid))), "l"(v1 + ((long long)(t + step) * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (1120 + tid))), "l"(v1 + ((long long)(t + step) * 7168ll + (long long)(base + 1792))));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (1344 + tid))), "l"(v1 + ((long long)(t + step) * 7168ll + (long long)(base + 3584))));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (1568 + tid))), "l"(v1 + ((long long)(t + step) * 7168ll + (long long)(base + 5376))));
            }
            if (2 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (1792 + tid))), "l"(v2 + ((long long)(t + step) * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (2016 + tid))), "l"(v2 + ((long long)(t + step) * 7168ll + (long long)(base + 1792))));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (2240 + tid))), "l"(v2 + ((long long)(t + step) * 7168ll + (long long)(base + 3584))));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (2464 + tid))), "l"(v2 + ((long long)(t + step) * 7168ll + (long long)(base + 5376))));
            }
            if (3 < n) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (2688 + tid))), "l"(v3 + ((long long)(t + step) * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (2912 + tid))), "l"(v3 + ((long long)(t + step) * 7168ll + (long long)(base + 1792))));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (3136 + tid))), "l"(v3 + ((long long)(t + step) * 7168ll + (long long)(base + 3584))));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (3360 + tid))), "l"(v3 + ((long long)(t + step) * 7168ll + (long long)(base + 5376))));
            }
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (int i_0 = 0; i_0 < 4; i_0++) {
            int ws_upper_0 = lane & 16;
            float ws_keep_0 = ((ws_upper_0 != 0) ? part[4 + i_0] : part[i_0]);
            float ws_send_0 = ((ws_upper_0 != 0) ? part[i_0] : part[4 + i_0]);
            float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
            part[i_0] = ws_keep_0 + _shfl_xor_sync_24;
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; i_1++) {
            int ws_upper_1 = lane & 8;
            float ws_keep_1 = ((ws_upper_1 != 0) ? part[2 + i_1] : part[i_1]);
            float ws_send_1 = ((ws_upper_1 != 0) ? part[i_1] : part[2 + i_1]);
            float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
            part[i_1] = ws_keep_1 + _shfl_xor_sync_25;
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 1; i_2++) {
            int ws_upper_2 = lane & 4;
            float ws_keep_2 = ((ws_upper_2 != 0) ? part[1 + i_2] : part[i_2]);
            float ws_send_2 = ((ws_upper_2 != 0) ? part[i_2] : part[1 + i_2]);
            float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
            part[i_2] = ws_keep_2 + _shfl_xor_sync_26;
        }
        float ws_own = part[0];
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 2);
        ws_own = ws_own + _shfl_xor_sync_27;
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
        ws_own = ws_own + _shfl_xor_sync_28;
        float tx_word = ws_own;
        int tx_row = xs * 56;
        if ((lane & 3) == 0) {
            if (lane >> 2 < 8) {
                wtab[tx_row + warp * 8 + (lane >> 2)] = tx_word;
            }
        }
        __syncthreads();
        float tk_ss = wtab[tx_row + jl];
        float tk_d = wtab[tx_row + 4 + jl];
        #pragma unroll
        for (int wi = 1; wi < 7; wi++) {
            tk_ss = tk_ss + wtab[tx_row + wi * 8 + jl];
            tk_d = tk_d + wtab[tx_row + wi * 8 + 4 + jl];
        }
        xs = 1 - xs;
        float _fdiv_30 = __fdiv_rn(tk_ss, 7168.0f);
        float _sqrt_29;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_29) : "f"(_fdiv_30 + eps));
        float tk_r = __frcp_rn(_sqrt_29);
        float tk_s = tk_r * tk_d;
        float tk_m = ((act != 0) ? tk_s : -3.4e+38f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tk_m = max_noftz(tk_m, __shfl_xor_sync(0xFFFFFFFF, tk_m, offset));
        float _exp_31 = __nv_expf(tk_s - tk_m);
        float tk_e = ((act != 0) ? _exp_31 : 0.0f);
        float tk_z = tk_e;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tk_z += __shfl_xor_sync(0xFFFFFFFF, tk_z, offset);
        float tk_p = __fdiv_rn(tk_e, tk_z);
        if (warp == 0) {
            if (act != 0) {
                long long tk_at = (long long)t * (long long)n + (long long)lane;
                p[tk_at] = tk_p;
                r[tk_at] = tk_r;
                d[tk_at] = tk_d;
            }
        }
        pj[0] = 0.0f;
        if (0 < n) {
            float _shfl_sync_32 = __shfl_sync(0xFFFFFFFF, tk_p, 0);
            pj[0] = _shfl_sync_32;
        }
        pj[1] = 0.0f;
        if (1 < n) {
            float _shfl_sync_33 = __shfl_sync(0xFFFFFFFF, tk_p, 1);
            pj[1] = _shfl_sync_33;
        }
        pj[2] = 0.0f;
        if (2 < n) {
            float _shfl_sync_34 = __shfl_sync(0xFFFFFFFF, tk_p, 2);
            pj[2] = _shfl_sync_34;
        }
        pj[3] = 0.0f;
        if (3 < n) {
            float _shfl_sync_35 = __shfl_sync(0xFFFFFFFF, tk_p, 3);
            pj[3] = _shfl_sync_35;
        }
        float to_o_0[8];
        #pragma unroll
        for (int i_3 = 0; i_3 < 8; i_3++) {
            to_o_0[i_3] = 0.0f;
        }
        if (0 < n) {
            float2 tj_p2_0 = make_float2(pj[0], pj[0]);
            #pragma unroll
            for (int m_32 = 0; m_32 < 4; m_32++) {
                float2 tj_o_0 = fma_f32x2_rn_noftz(tj_p2_0, cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_32])), make_float2(to_o_0[2 * m_32], to_o_0[2 * m_32 + 1]));
                to_o_0[2 * m_32] = tj_o_0.x;
                to_o_0[2 * m_32 + 1] = tj_o_0.y;
            }
        }
        if (1 < n) {
            float2 tj_p2_1 = make_float2(pj[1], pj[1]);
            #pragma unroll
            for (int m_33 = 0; m_33 < 4; m_33++) {
                float2 tj_o_1 = fma_f32x2_rn_noftz(tj_p2_1, cake_bf16x2_to_f32x2(__as_bf16x2(vw[16 + m_33])), make_float2(to_o_0[2 * m_33], to_o_0[2 * m_33 + 1]));
                to_o_0[2 * m_33] = tj_o_1.x;
                to_o_0[2 * m_33 + 1] = tj_o_1.y;
            }
        }
        if (2 < n) {
            float2 tj_p2_2 = make_float2(pj[2], pj[2]);
            #pragma unroll
            for (int m_34 = 0; m_34 < 4; m_34++) {
                float2 tj_o_2 = fma_f32x2_rn_noftz(tj_p2_2, cake_bf16x2_to_f32x2(__as_bf16x2(vw[32 + m_34])), make_float2(to_o_0[2 * m_34], to_o_0[2 * m_34 + 1]));
                to_o_0[2 * m_34] = tj_o_2.x;
                to_o_0[2 * m_34 + 1] = tj_o_2.y;
            }
        }
        if (3 < n) {
            float2 tj_p2_3 = make_float2(pj[3], pj[3]);
            #pragma unroll
            for (int m_35 = 0; m_35 < 4; m_35++) {
                float2 tj_o_3 = fma_f32x2_rn_noftz(tj_p2_3, cake_bf16x2_to_f32x2(__as_bf16x2(vw[48 + m_35])), make_float2(to_o_0[2 * m_35], to_o_0[2 * m_35 + 1]));
                to_o_0[2 * m_35] = tj_o_3.x;
                to_o_0[2 * m_35 + 1] = tj_o_3.y;
            }
        }
        unsigned int ss_packed[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_o_0[_lp*2 + 0], to_o_0[_lp*2+1 + 0]));
            ss_packed[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + ((long long)t * 7168ll + (long long)base))[0] = reinterpret_cast<int4*>(ss_packed)[0];
        float to_o_1[8];
        #pragma unroll
        for (int i_4 = 0; i_4 < 8; i_4++) {
            to_o_1[i_4] = 0.0f;
        }
        if (0 < n) {
            float2 tj_p2_4 = make_float2(pj[0], pj[0]);
            #pragma unroll
            for (int m_36 = 0; m_36 < 4; m_36++) {
                float2 tj_o_4 = fma_f32x2_rn_noftz(tj_p2_4, cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_36])), make_float2(to_o_1[2 * m_36], to_o_1[2 * m_36 + 1]));
                to_o_1[2 * m_36] = tj_o_4.x;
                to_o_1[2 * m_36 + 1] = tj_o_4.y;
            }
        }
        if (1 < n) {
            float2 tj_p2_5 = make_float2(pj[1], pj[1]);
            #pragma unroll
            for (int m_37 = 0; m_37 < 4; m_37++) {
                float2 tj_o_5 = fma_f32x2_rn_noftz(tj_p2_5, cake_bf16x2_to_f32x2(__as_bf16x2(vw[20 + m_37])), make_float2(to_o_1[2 * m_37], to_o_1[2 * m_37 + 1]));
                to_o_1[2 * m_37] = tj_o_5.x;
                to_o_1[2 * m_37 + 1] = tj_o_5.y;
            }
        }
        if (2 < n) {
            float2 tj_p2_6 = make_float2(pj[2], pj[2]);
            #pragma unroll
            for (int m_38 = 0; m_38 < 4; m_38++) {
                float2 tj_o_6 = fma_f32x2_rn_noftz(tj_p2_6, cake_bf16x2_to_f32x2(__as_bf16x2(vw[36 + m_38])), make_float2(to_o_1[2 * m_38], to_o_1[2 * m_38 + 1]));
                to_o_1[2 * m_38] = tj_o_6.x;
                to_o_1[2 * m_38 + 1] = tj_o_6.y;
            }
        }
        if (3 < n) {
            float2 tj_p2_7 = make_float2(pj[3], pj[3]);
            #pragma unroll
            for (int m_39 = 0; m_39 < 4; m_39++) {
                float2 tj_o_7 = fma_f32x2_rn_noftz(tj_p2_7, cake_bf16x2_to_f32x2(__as_bf16x2(vw[52 + m_39])), make_float2(to_o_1[2 * m_39], to_o_1[2 * m_39 + 1]));
                to_o_1[2 * m_39] = tj_o_7.x;
                to_o_1[2 * m_39 + 1] = tj_o_7.y;
            }
        }
        unsigned int ss_packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_o_1[_lp*2 + 0], to_o_1[_lp*2+1 + 0]));
            ss_packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + ((long long)t * 7168ll + (long long)(base + 1792)))[0] = reinterpret_cast<int4*>(ss_packed_0)[0];
        float to_o_2[8];
        #pragma unroll
        for (int i_5 = 0; i_5 < 8; i_5++) {
            to_o_2[i_5] = 0.0f;
        }
        if (0 < n) {
            float2 tj_p2_8 = make_float2(pj[0], pj[0]);
            #pragma unroll
            for (int m_40 = 0; m_40 < 4; m_40++) {
                float2 tj_o_8 = fma_f32x2_rn_noftz(tj_p2_8, cake_bf16x2_to_f32x2(__as_bf16x2(vw[8 + m_40])), make_float2(to_o_2[2 * m_40], to_o_2[2 * m_40 + 1]));
                to_o_2[2 * m_40] = tj_o_8.x;
                to_o_2[2 * m_40 + 1] = tj_o_8.y;
            }
        }
        if (1 < n) {
            float2 tj_p2_9 = make_float2(pj[1], pj[1]);
            #pragma unroll
            for (int m_41 = 0; m_41 < 4; m_41++) {
                float2 tj_o_9 = fma_f32x2_rn_noftz(tj_p2_9, cake_bf16x2_to_f32x2(__as_bf16x2(vw[24 + m_41])), make_float2(to_o_2[2 * m_41], to_o_2[2 * m_41 + 1]));
                to_o_2[2 * m_41] = tj_o_9.x;
                to_o_2[2 * m_41 + 1] = tj_o_9.y;
            }
        }
        if (2 < n) {
            float2 tj_p2_10 = make_float2(pj[2], pj[2]);
            #pragma unroll
            for (int m_42 = 0; m_42 < 4; m_42++) {
                float2 tj_o_10 = fma_f32x2_rn_noftz(tj_p2_10, cake_bf16x2_to_f32x2(__as_bf16x2(vw[40 + m_42])), make_float2(to_o_2[2 * m_42], to_o_2[2 * m_42 + 1]));
                to_o_2[2 * m_42] = tj_o_10.x;
                to_o_2[2 * m_42 + 1] = tj_o_10.y;
            }
        }
        if (3 < n) {
            float2 tj_p2_11 = make_float2(pj[3], pj[3]);
            #pragma unroll
            for (int m_43 = 0; m_43 < 4; m_43++) {
                float2 tj_o_11 = fma_f32x2_rn_noftz(tj_p2_11, cake_bf16x2_to_f32x2(__as_bf16x2(vw[56 + m_43])), make_float2(to_o_2[2 * m_43], to_o_2[2 * m_43 + 1]));
                to_o_2[2 * m_43] = tj_o_11.x;
                to_o_2[2 * m_43 + 1] = tj_o_11.y;
            }
        }
        unsigned int ss_packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_o_2[_lp*2 + 0], to_o_2[_lp*2+1 + 0]));
            ss_packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + ((long long)t * 7168ll + (long long)(base + 3584)))[0] = reinterpret_cast<int4*>(ss_packed_1)[0];
        float to_o_3[8];
        #pragma unroll
        for (int i_6 = 0; i_6 < 8; i_6++) {
            to_o_3[i_6] = 0.0f;
        }
        if (0 < n) {
            float2 tj_p2_12 = make_float2(pj[0], pj[0]);
            #pragma unroll
            for (int m_44 = 0; m_44 < 4; m_44++) {
                float2 tj_o_12 = fma_f32x2_rn_noftz(tj_p2_12, cake_bf16x2_to_f32x2(__as_bf16x2(vw[12 + m_44])), make_float2(to_o_3[2 * m_44], to_o_3[2 * m_44 + 1]));
                to_o_3[2 * m_44] = tj_o_12.x;
                to_o_3[2 * m_44 + 1] = tj_o_12.y;
            }
        }
        if (1 < n) {
            float2 tj_p2_13 = make_float2(pj[1], pj[1]);
            #pragma unroll
            for (int m_45 = 0; m_45 < 4; m_45++) {
                float2 tj_o_13 = fma_f32x2_rn_noftz(tj_p2_13, cake_bf16x2_to_f32x2(__as_bf16x2(vw[28 + m_45])), make_float2(to_o_3[2 * m_45], to_o_3[2 * m_45 + 1]));
                to_o_3[2 * m_45] = tj_o_13.x;
                to_o_3[2 * m_45 + 1] = tj_o_13.y;
            }
        }
        if (2 < n) {
            float2 tj_p2_14 = make_float2(pj[2], pj[2]);
            #pragma unroll
            for (int m_46 = 0; m_46 < 4; m_46++) {
                float2 tj_o_14 = fma_f32x2_rn_noftz(tj_p2_14, cake_bf16x2_to_f32x2(__as_bf16x2(vw[44 + m_46])), make_float2(to_o_3[2 * m_46], to_o_3[2 * m_46 + 1]));
                to_o_3[2 * m_46] = tj_o_14.x;
                to_o_3[2 * m_46 + 1] = tj_o_14.y;
            }
        }
        if (3 < n) {
            float2 tj_p2_15 = make_float2(pj[3], pj[3]);
            #pragma unroll
            for (int m_47 = 0; m_47 < 4; m_47++) {
                float2 tj_o_15 = fma_f32x2_rn_noftz(tj_p2_15, cake_bf16x2_to_f32x2(__as_bf16x2(vw[60 + m_47])), make_float2(to_o_3[2 * m_47], to_o_3[2 * m_47 + 1]));
                to_o_3[2 * m_47] = tj_o_15.x;
                to_o_3[2 * m_47 + 1] = tj_o_15.y;
            }
        }
        unsigned int ss_packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_o_3[_lp*2 + 0], to_o_3[_lp*2+1 + 0]));
            ss_packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + ((long long)t * 7168ll + (long long)(base + 5376)))[0] = reinterpret_cast<int4*>(ss_packed_2)[0];
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
}

} // extern "C"
