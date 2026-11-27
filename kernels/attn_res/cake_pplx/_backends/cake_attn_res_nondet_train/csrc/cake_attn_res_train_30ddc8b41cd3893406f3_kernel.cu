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

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


extern "C" __device__ float __nv_expf(float);


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

__global__ __launch_bounds__(448, LAUNCH_MIN_BLOCKS) void
kernel_cake_attn_res_train_30ddc8b41cd3893406f3(__nv_bfloat16* __restrict__ v0, __nv_bfloat16* __restrict__ v1, __nv_bfloat16* __restrict__ v2, __nv_bfloat16* __restrict__ v3, __nv_bfloat16* __restrict__ v4, __nv_bfloat16* __restrict__ v5, __nv_bfloat16* __restrict__ v6, __nv_bfloat16* __restrict__ v7, __nv_bfloat16* __restrict__ v8, float* __restrict__ w, __nv_bfloat16* __restrict__ out, float* __restrict__ p, float* __restrict__ r, float* __restrict__ d, int T, int n, float eps)
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

    const int cta_rank = 0;

    // Prologue
    float* wtab = reinterpret_cast<float*>(smem_raw + 0);
    const int wtab_addr = smem + 0;
    unsigned int* stg = reinterpret_cast<unsigned int*>(smem_raw + 1792);
    const int stage_addr = smem + 1792;

    // === Task calls (declaration order) ===
    int first = bid;
    int step = num_bids;
    int base = tid * 8;
    int jl = ((lane) < (7) ? (lane) : (7));
    int act = ((lane < n) ? 1 : 0);
    float wf[16];
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
                : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(w + (base + (3584 + 4 * half_1)) + (0))) : "memory");
            wf[8 + 4 * half_1 + 0] = __uint_as_float(_v4_1_0);
            wf[8 + 4 * half_1 + 1] = __uint_as_float(_v4_1_1);
            wf[8 + 4 * half_1 + 2] = __uint_as_float(_v4_1_2);
            wf[8 + 4 * half_1 + 3] = __uint_as_float(_v4_1_3);
        }
    }
    unsigned int vw[64];
    float pj[8];
    float q[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        q[i] = 0.0f;
    }
    float ch[4];
    float tot[4];
    int xs = 0;
    if (first < T) {
        if (0 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * tid)), "l"(v0 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (448 + tid))), "l"(v0 + ((long long)first * 7168ll + (long long)(base + 3584))));
        }
        asm volatile("cp.async.commit_group;");
        if (1 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (896 + tid))), "l"(v1 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (1344 + tid))), "l"(v1 + ((long long)first * 7168ll + (long long)(base + 3584))));
        }
        asm volatile("cp.async.commit_group;");
        if (2 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (1792 + tid))), "l"(v2 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (2240 + tid))), "l"(v2 + ((long long)first * 7168ll + (long long)(base + 3584))));
        }
        asm volatile("cp.async.commit_group;");
        if (3 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (2688 + tid))), "l"(v3 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (3136 + tid))), "l"(v3 + ((long long)first * 7168ll + (long long)(base + 3584))));
        }
        asm volatile("cp.async.commit_group;");
        if (4 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (3584 + tid))), "l"(v4 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (4032 + tid))), "l"(v4 + ((long long)first * 7168ll + (long long)(base + 3584))));
        }
        asm volatile("cp.async.commit_group;");
        if (5 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (4480 + tid))), "l"(v5 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (4928 + tid))), "l"(v5 + ((long long)first * 7168ll + (long long)(base + 3584))));
        }
        asm volatile("cp.async.commit_group;");
        if (6 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (5376 + tid))), "l"(v6 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (5824 + tid))), "l"(v6 + ((long long)first * 7168ll + (long long)(base + 3584))));
        }
        asm volatile("cp.async.commit_group;");
        if (7 < n) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (6272 + tid))), "l"(v7 + ((long long)first * 7168ll + (long long)base)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(stage_addr + (unsigned int)(16 * (6720 + tid))), "l"(v7 + ((long long)first * 7168ll + (long long)(base + 3584))));
        }
        asm volatile("cp.async.commit_group;");
    }
    #pragma unroll 1
    for (int t = first; t < T; t += step) {
        unsigned int tv_at = stage_addr;
        int tv_nxt = t + step;
        q[0] = 0.0f;
        if (0 < n) {
            asm volatile("cp.async.wait_group 7;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[0])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(0) + 3]))
                : "r"(tv_at + (unsigned int)(16 * tid)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[4])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(4) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (448 + tid))));
            #pragma unroll
            for (int i_1 = 0; i_1 < 4; i_1++) {
                tot[i_1] = 0.0f;
            }
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                float2 sl_v = cake_bf16x2_to_f32x2(__as_bf16x2(vw[m]));
                float2 sl_s = fma_f32x2_rn_noftz(sl_v, sl_v, make_float2(tot[0], tot[1]));
                tot[0] = sl_s.x;
                tot[1] = sl_s.y;
                float2 sl_d = fma_f32x2_rn_noftz(make_float2(wf[2 * m], wf[2 * m + 1]), sl_v, make_float2(tot[2], tot[3]));
                tot[2] = sl_d.x;
                tot[3] = sl_d.y;
            }
            #pragma unroll
            for (int i_2 = 0; i_2 < 4; i_2++) {
                ch[i_2] = 0.0f;
            }
            #pragma unroll
            for (int m_1 = 0; m_1 < 4; m_1++) {
                float2 sl_v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_1]));
                float2 sl_s_1 = fma_f32x2_rn_noftz(sl_v_1, sl_v_1, make_float2(ch[0], ch[1]));
                ch[0] = sl_s_1.x;
                ch[1] = sl_s_1.y;
                float2 sl_d_1 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_1], wf[8 + 2 * m_1 + 1]), sl_v_1, make_float2(ch[2], ch[3]));
                ch[2] = sl_d_1.x;
                ch[3] = sl_d_1.y;
            }
            #pragma unroll
            for (int h_0 = 0; h_0 < 2; h_0++) {
                float2 vs_a_0 = add_f32x2_noftz(make_float2(tot[2 * h_0], tot[2 * h_0 + 1]), make_float2(ch[2 * h_0], ch[2 * h_0 + 1]));
                tot[2 * h_0] = vs_a_0.x;
                tot[2 * h_0 + 1] = vs_a_0.y;
            }
            if (tv_nxt < T) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * tid)), "l"(v0 + ((long long)tv_nxt * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (448 + tid))), "l"(v0 + ((long long)tv_nxt * 7168ll + (long long)(base + 3584))));
            }
            float vs_ss;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_ss) : "f"(tot[0]), "f"(tot[1]));
            float vs_d;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_d) : "f"(tot[2]), "f"(tot[3]));
            int pf_upper = lane & 16;
            float pf_keep = ((pf_upper != 0) ? vs_d : vs_ss);
            float pf_send = ((pf_upper != 0) ? vs_ss : vs_d);
            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, pf_send, 16);
            q[0] = pf_keep + _shfl_xor_sync_2;
        }
        asm volatile("cp.async.commit_group;");
        q[1] = 0.0f;
        if (1 < n) {
            asm volatile("cp.async.wait_group 7;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[8])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(8) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (896 + tid))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[12])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(12) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (1344 + tid))));
            #pragma unroll
            for (int i_3 = 0; i_3 < 4; i_3++) {
                tot[i_3] = 0.0f;
            }
            #pragma unroll
            for (int m_2 = 0; m_2 < 4; m_2++) {
                float2 sl_v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[8 + m_2]));
                float2 sl_s_2 = fma_f32x2_rn_noftz(sl_v_2, sl_v_2, make_float2(tot[0], tot[1]));
                tot[0] = sl_s_2.x;
                tot[1] = sl_s_2.y;
                float2 sl_d_2 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_2], wf[2 * m_2 + 1]), sl_v_2, make_float2(tot[2], tot[3]));
                tot[2] = sl_d_2.x;
                tot[3] = sl_d_2.y;
            }
            #pragma unroll
            for (int i_4 = 0; i_4 < 4; i_4++) {
                ch[i_4] = 0.0f;
            }
            #pragma unroll
            for (int m_3 = 0; m_3 < 4; m_3++) {
                float2 sl_v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[12 + m_3]));
                float2 sl_s_3 = fma_f32x2_rn_noftz(sl_v_3, sl_v_3, make_float2(ch[0], ch[1]));
                ch[0] = sl_s_3.x;
                ch[1] = sl_s_3.y;
                float2 sl_d_3 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_3], wf[8 + 2 * m_3 + 1]), sl_v_3, make_float2(ch[2], ch[3]));
                ch[2] = sl_d_3.x;
                ch[3] = sl_d_3.y;
            }
            #pragma unroll
            for (int h_1 = 0; h_1 < 2; h_1++) {
                float2 vs_a_1 = add_f32x2_noftz(make_float2(tot[2 * h_1], tot[2 * h_1 + 1]), make_float2(ch[2 * h_1], ch[2 * h_1 + 1]));
                tot[2 * h_1] = vs_a_1.x;
                tot[2 * h_1 + 1] = vs_a_1.y;
            }
            if (tv_nxt < T) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (896 + tid))), "l"(v1 + ((long long)tv_nxt * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (1344 + tid))), "l"(v1 + ((long long)tv_nxt * 7168ll + (long long)(base + 3584))));
            }
            float vs_ss_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_ss_1) : "f"(tot[0]), "f"(tot[1]));
            float vs_d_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_d_1) : "f"(tot[2]), "f"(tot[3]));
            int pf_upper_1 = lane & 16;
            float pf_keep_1 = ((pf_upper_1 != 0) ? vs_d_1 : vs_ss_1);
            float pf_send_1 = ((pf_upper_1 != 0) ? vs_ss_1 : vs_d_1);
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, pf_send_1, 16);
            q[1] = pf_keep_1 + _shfl_xor_sync_3;
        }
        asm volatile("cp.async.commit_group;");
        q[2] = 0.0f;
        if (2 < n) {
            asm volatile("cp.async.wait_group 7;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[16])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(16) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (1792 + tid))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[20])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(20) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (2240 + tid))));
            #pragma unroll
            for (int i_5 = 0; i_5 < 4; i_5++) {
                tot[i_5] = 0.0f;
            }
            #pragma unroll
            for (int m_4 = 0; m_4 < 4; m_4++) {
                float2 sl_v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[16 + m_4]));
                float2 sl_s_4 = fma_f32x2_rn_noftz(sl_v_4, sl_v_4, make_float2(tot[0], tot[1]));
                tot[0] = sl_s_4.x;
                tot[1] = sl_s_4.y;
                float2 sl_d_4 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_4], wf[2 * m_4 + 1]), sl_v_4, make_float2(tot[2], tot[3]));
                tot[2] = sl_d_4.x;
                tot[3] = sl_d_4.y;
            }
            #pragma unroll
            for (int i_6 = 0; i_6 < 4; i_6++) {
                ch[i_6] = 0.0f;
            }
            #pragma unroll
            for (int m_5 = 0; m_5 < 4; m_5++) {
                float2 sl_v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[20 + m_5]));
                float2 sl_s_5 = fma_f32x2_rn_noftz(sl_v_5, sl_v_5, make_float2(ch[0], ch[1]));
                ch[0] = sl_s_5.x;
                ch[1] = sl_s_5.y;
                float2 sl_d_5 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_5], wf[8 + 2 * m_5 + 1]), sl_v_5, make_float2(ch[2], ch[3]));
                ch[2] = sl_d_5.x;
                ch[3] = sl_d_5.y;
            }
            #pragma unroll
            for (int h_2 = 0; h_2 < 2; h_2++) {
                float2 vs_a_2 = add_f32x2_noftz(make_float2(tot[2 * h_2], tot[2 * h_2 + 1]), make_float2(ch[2 * h_2], ch[2 * h_2 + 1]));
                tot[2 * h_2] = vs_a_2.x;
                tot[2 * h_2 + 1] = vs_a_2.y;
            }
            if (tv_nxt < T) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (1792 + tid))), "l"(v2 + ((long long)tv_nxt * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (2240 + tid))), "l"(v2 + ((long long)tv_nxt * 7168ll + (long long)(base + 3584))));
            }
            float vs_ss_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_ss_2) : "f"(tot[0]), "f"(tot[1]));
            float vs_d_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_d_2) : "f"(tot[2]), "f"(tot[3]));
            int pf_upper_2 = lane & 16;
            float pf_keep_2 = ((pf_upper_2 != 0) ? vs_d_2 : vs_ss_2);
            float pf_send_2 = ((pf_upper_2 != 0) ? vs_ss_2 : vs_d_2);
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, pf_send_2, 16);
            q[2] = pf_keep_2 + _shfl_xor_sync_4;
        }
        asm volatile("cp.async.commit_group;");
        q[3] = 0.0f;
        if (3 < n) {
            asm volatile("cp.async.wait_group 7;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[24])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(24) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (2688 + tid))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[28])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(28) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (3136 + tid))));
            #pragma unroll
            for (int i_7 = 0; i_7 < 4; i_7++) {
                tot[i_7] = 0.0f;
            }
            #pragma unroll
            for (int m_6 = 0; m_6 < 4; m_6++) {
                float2 sl_v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[24 + m_6]));
                float2 sl_s_6 = fma_f32x2_rn_noftz(sl_v_6, sl_v_6, make_float2(tot[0], tot[1]));
                tot[0] = sl_s_6.x;
                tot[1] = sl_s_6.y;
                float2 sl_d_6 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_6], wf[2 * m_6 + 1]), sl_v_6, make_float2(tot[2], tot[3]));
                tot[2] = sl_d_6.x;
                tot[3] = sl_d_6.y;
            }
            #pragma unroll
            for (int i_8 = 0; i_8 < 4; i_8++) {
                ch[i_8] = 0.0f;
            }
            #pragma unroll
            for (int m_7 = 0; m_7 < 4; m_7++) {
                float2 sl_v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[28 + m_7]));
                float2 sl_s_7 = fma_f32x2_rn_noftz(sl_v_7, sl_v_7, make_float2(ch[0], ch[1]));
                ch[0] = sl_s_7.x;
                ch[1] = sl_s_7.y;
                float2 sl_d_7 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_7], wf[8 + 2 * m_7 + 1]), sl_v_7, make_float2(ch[2], ch[3]));
                ch[2] = sl_d_7.x;
                ch[3] = sl_d_7.y;
            }
            #pragma unroll
            for (int h_3 = 0; h_3 < 2; h_3++) {
                float2 vs_a_3 = add_f32x2_noftz(make_float2(tot[2 * h_3], tot[2 * h_3 + 1]), make_float2(ch[2 * h_3], ch[2 * h_3 + 1]));
                tot[2 * h_3] = vs_a_3.x;
                tot[2 * h_3 + 1] = vs_a_3.y;
            }
            if (tv_nxt < T) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (2688 + tid))), "l"(v3 + ((long long)tv_nxt * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (3136 + tid))), "l"(v3 + ((long long)tv_nxt * 7168ll + (long long)(base + 3584))));
            }
            float vs_ss_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_ss_3) : "f"(tot[0]), "f"(tot[1]));
            float vs_d_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_d_3) : "f"(tot[2]), "f"(tot[3]));
            int pf_upper_3 = lane & 16;
            float pf_keep_3 = ((pf_upper_3 != 0) ? vs_d_3 : vs_ss_3);
            float pf_send_3 = ((pf_upper_3 != 0) ? vs_ss_3 : vs_d_3);
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, pf_send_3, 16);
            q[3] = pf_keep_3 + _shfl_xor_sync_5;
        }
        asm volatile("cp.async.commit_group;");
        q[4] = 0.0f;
        if (4 < n) {
            asm volatile("cp.async.wait_group 7;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[32])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(32) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (3584 + tid))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[36])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(36) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (4032 + tid))));
            #pragma unroll
            for (int i_9 = 0; i_9 < 4; i_9++) {
                tot[i_9] = 0.0f;
            }
            #pragma unroll
            for (int m_8 = 0; m_8 < 4; m_8++) {
                float2 sl_v_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[32 + m_8]));
                float2 sl_s_8 = fma_f32x2_rn_noftz(sl_v_8, sl_v_8, make_float2(tot[0], tot[1]));
                tot[0] = sl_s_8.x;
                tot[1] = sl_s_8.y;
                float2 sl_d_8 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_8], wf[2 * m_8 + 1]), sl_v_8, make_float2(tot[2], tot[3]));
                tot[2] = sl_d_8.x;
                tot[3] = sl_d_8.y;
            }
            #pragma unroll
            for (int i_10 = 0; i_10 < 4; i_10++) {
                ch[i_10] = 0.0f;
            }
            #pragma unroll
            for (int m_9 = 0; m_9 < 4; m_9++) {
                float2 sl_v_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[36 + m_9]));
                float2 sl_s_9 = fma_f32x2_rn_noftz(sl_v_9, sl_v_9, make_float2(ch[0], ch[1]));
                ch[0] = sl_s_9.x;
                ch[1] = sl_s_9.y;
                float2 sl_d_9 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_9], wf[8 + 2 * m_9 + 1]), sl_v_9, make_float2(ch[2], ch[3]));
                ch[2] = sl_d_9.x;
                ch[3] = sl_d_9.y;
            }
            #pragma unroll
            for (int h_4 = 0; h_4 < 2; h_4++) {
                float2 vs_a_4 = add_f32x2_noftz(make_float2(tot[2 * h_4], tot[2 * h_4 + 1]), make_float2(ch[2 * h_4], ch[2 * h_4 + 1]));
                tot[2 * h_4] = vs_a_4.x;
                tot[2 * h_4 + 1] = vs_a_4.y;
            }
            if (tv_nxt < T) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (3584 + tid))), "l"(v4 + ((long long)tv_nxt * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (4032 + tid))), "l"(v4 + ((long long)tv_nxt * 7168ll + (long long)(base + 3584))));
            }
            float vs_ss_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_ss_4) : "f"(tot[0]), "f"(tot[1]));
            float vs_d_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_d_4) : "f"(tot[2]), "f"(tot[3]));
            int pf_upper_4 = lane & 16;
            float pf_keep_4 = ((pf_upper_4 != 0) ? vs_d_4 : vs_ss_4);
            float pf_send_4 = ((pf_upper_4 != 0) ? vs_ss_4 : vs_d_4);
            float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, pf_send_4, 16);
            q[4] = pf_keep_4 + _shfl_xor_sync_6;
        }
        asm volatile("cp.async.commit_group;");
        q[5] = 0.0f;
        if (5 < n) {
            asm volatile("cp.async.wait_group 7;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[40])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(40) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (4480 + tid))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[44])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(44) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (4928 + tid))));
            #pragma unroll
            for (int i_11 = 0; i_11 < 4; i_11++) {
                tot[i_11] = 0.0f;
            }
            #pragma unroll
            for (int m_10 = 0; m_10 < 4; m_10++) {
                float2 sl_v_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[40 + m_10]));
                float2 sl_s_10 = fma_f32x2_rn_noftz(sl_v_10, sl_v_10, make_float2(tot[0], tot[1]));
                tot[0] = sl_s_10.x;
                tot[1] = sl_s_10.y;
                float2 sl_d_10 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_10], wf[2 * m_10 + 1]), sl_v_10, make_float2(tot[2], tot[3]));
                tot[2] = sl_d_10.x;
                tot[3] = sl_d_10.y;
            }
            #pragma unroll
            for (int i_12 = 0; i_12 < 4; i_12++) {
                ch[i_12] = 0.0f;
            }
            #pragma unroll
            for (int m_11 = 0; m_11 < 4; m_11++) {
                float2 sl_v_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[44 + m_11]));
                float2 sl_s_11 = fma_f32x2_rn_noftz(sl_v_11, sl_v_11, make_float2(ch[0], ch[1]));
                ch[0] = sl_s_11.x;
                ch[1] = sl_s_11.y;
                float2 sl_d_11 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_11], wf[8 + 2 * m_11 + 1]), sl_v_11, make_float2(ch[2], ch[3]));
                ch[2] = sl_d_11.x;
                ch[3] = sl_d_11.y;
            }
            #pragma unroll
            for (int h_5 = 0; h_5 < 2; h_5++) {
                float2 vs_a_5 = add_f32x2_noftz(make_float2(tot[2 * h_5], tot[2 * h_5 + 1]), make_float2(ch[2 * h_5], ch[2 * h_5 + 1]));
                tot[2 * h_5] = vs_a_5.x;
                tot[2 * h_5 + 1] = vs_a_5.y;
            }
            if (tv_nxt < T) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (4480 + tid))), "l"(v5 + ((long long)tv_nxt * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (4928 + tid))), "l"(v5 + ((long long)tv_nxt * 7168ll + (long long)(base + 3584))));
            }
            float vs_ss_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_ss_5) : "f"(tot[0]), "f"(tot[1]));
            float vs_d_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_d_5) : "f"(tot[2]), "f"(tot[3]));
            int pf_upper_5 = lane & 16;
            float pf_keep_5 = ((pf_upper_5 != 0) ? vs_d_5 : vs_ss_5);
            float pf_send_5 = ((pf_upper_5 != 0) ? vs_ss_5 : vs_d_5);
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, pf_send_5, 16);
            q[5] = pf_keep_5 + _shfl_xor_sync_7;
        }
        asm volatile("cp.async.commit_group;");
        q[6] = 0.0f;
        if (6 < n) {
            asm volatile("cp.async.wait_group 7;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[48])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(48) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (5376 + tid))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[52])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(52) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (5824 + tid))));
            #pragma unroll
            for (int i_13 = 0; i_13 < 4; i_13++) {
                tot[i_13] = 0.0f;
            }
            #pragma unroll
            for (int m_12 = 0; m_12 < 4; m_12++) {
                float2 sl_v_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[48 + m_12]));
                float2 sl_s_12 = fma_f32x2_rn_noftz(sl_v_12, sl_v_12, make_float2(tot[0], tot[1]));
                tot[0] = sl_s_12.x;
                tot[1] = sl_s_12.y;
                float2 sl_d_12 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_12], wf[2 * m_12 + 1]), sl_v_12, make_float2(tot[2], tot[3]));
                tot[2] = sl_d_12.x;
                tot[3] = sl_d_12.y;
            }
            #pragma unroll
            for (int i_14 = 0; i_14 < 4; i_14++) {
                ch[i_14] = 0.0f;
            }
            #pragma unroll
            for (int m_13 = 0; m_13 < 4; m_13++) {
                float2 sl_v_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[52 + m_13]));
                float2 sl_s_13 = fma_f32x2_rn_noftz(sl_v_13, sl_v_13, make_float2(ch[0], ch[1]));
                ch[0] = sl_s_13.x;
                ch[1] = sl_s_13.y;
                float2 sl_d_13 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_13], wf[8 + 2 * m_13 + 1]), sl_v_13, make_float2(ch[2], ch[3]));
                ch[2] = sl_d_13.x;
                ch[3] = sl_d_13.y;
            }
            #pragma unroll
            for (int h_6 = 0; h_6 < 2; h_6++) {
                float2 vs_a_6 = add_f32x2_noftz(make_float2(tot[2 * h_6], tot[2 * h_6 + 1]), make_float2(ch[2 * h_6], ch[2 * h_6 + 1]));
                tot[2 * h_6] = vs_a_6.x;
                tot[2 * h_6 + 1] = vs_a_6.y;
            }
            if (tv_nxt < T) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (5376 + tid))), "l"(v6 + ((long long)tv_nxt * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (5824 + tid))), "l"(v6 + ((long long)tv_nxt * 7168ll + (long long)(base + 3584))));
            }
            float vs_ss_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_ss_6) : "f"(tot[0]), "f"(tot[1]));
            float vs_d_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_d_6) : "f"(tot[2]), "f"(tot[3]));
            int pf_upper_6 = lane & 16;
            float pf_keep_6 = ((pf_upper_6 != 0) ? vs_d_6 : vs_ss_6);
            float pf_send_6 = ((pf_upper_6 != 0) ? vs_ss_6 : vs_d_6);
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, pf_send_6, 16);
            q[6] = pf_keep_6 + _shfl_xor_sync_8;
        }
        asm volatile("cp.async.commit_group;");
        q[7] = 0.0f;
        if (7 < n) {
            asm volatile("cp.async.wait_group 7;");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[56])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(56) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (6272 + tid))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&vw[60])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vw[(60) + 3]))
                : "r"(tv_at + (unsigned int)(16 * (6720 + tid))));
            #pragma unroll
            for (int i_15 = 0; i_15 < 4; i_15++) {
                tot[i_15] = 0.0f;
            }
            #pragma unroll
            for (int m_14 = 0; m_14 < 4; m_14++) {
                float2 sl_v_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[56 + m_14]));
                float2 sl_s_14 = fma_f32x2_rn_noftz(sl_v_14, sl_v_14, make_float2(tot[0], tot[1]));
                tot[0] = sl_s_14.x;
                tot[1] = sl_s_14.y;
                float2 sl_d_14 = fma_f32x2_rn_noftz(make_float2(wf[2 * m_14], wf[2 * m_14 + 1]), sl_v_14, make_float2(tot[2], tot[3]));
                tot[2] = sl_d_14.x;
                tot[3] = sl_d_14.y;
            }
            #pragma unroll
            for (int i_16 = 0; i_16 < 4; i_16++) {
                ch[i_16] = 0.0f;
            }
            #pragma unroll
            for (int m_15 = 0; m_15 < 4; m_15++) {
                float2 sl_v_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vw[60 + m_15]));
                float2 sl_s_15 = fma_f32x2_rn_noftz(sl_v_15, sl_v_15, make_float2(ch[0], ch[1]));
                ch[0] = sl_s_15.x;
                ch[1] = sl_s_15.y;
                float2 sl_d_15 = fma_f32x2_rn_noftz(make_float2(wf[8 + 2 * m_15], wf[8 + 2 * m_15 + 1]), sl_v_15, make_float2(ch[2], ch[3]));
                ch[2] = sl_d_15.x;
                ch[3] = sl_d_15.y;
            }
            #pragma unroll
            for (int h_7 = 0; h_7 < 2; h_7++) {
                float2 vs_a_7 = add_f32x2_noftz(make_float2(tot[2 * h_7], tot[2 * h_7 + 1]), make_float2(ch[2 * h_7], ch[2 * h_7 + 1]));
                tot[2 * h_7] = vs_a_7.x;
                tot[2 * h_7 + 1] = vs_a_7.y;
            }
            if (tv_nxt < T) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (6272 + tid))), "l"(v7 + ((long long)tv_nxt * 7168ll + (long long)base)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(tv_at + (unsigned int)(16 * (6720 + tid))), "l"(v7 + ((long long)tv_nxt * 7168ll + (long long)(base + 3584))));
            }
            float vs_ss_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_ss_7) : "f"(tot[0]), "f"(tot[1]));
            float vs_d_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vs_d_7) : "f"(tot[2]), "f"(tot[3]));
            int pf_upper_7 = lane & 16;
            float pf_keep_7 = ((pf_upper_7 != 0) ? vs_d_7 : vs_ss_7);
            float pf_send_7 = ((pf_upper_7 != 0) ? vs_ss_7 : vs_d_7);
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, pf_send_7, 16);
            q[7] = pf_keep_7 + _shfl_xor_sync_9;
        }
        asm volatile("cp.async.commit_group;");
        #pragma unroll
        for (int i_0 = 0; i_0 < 4; i_0++) {
            int ps_upper_0 = lane & 8;
            float ps_keep_0 = ((ps_upper_0 != 0) ? q[4 + i_0] : q[i_0]);
            float ps_send_0 = ((ps_upper_0 != 0) ? q[i_0] : q[4 + i_0]);
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, ps_send_0, 8);
            q[i_0] = ps_keep_0 + _shfl_xor_sync_10;
        }
        #pragma unroll
        for (int i_1_1 = 0; i_1_1 < 2; i_1_1++) {
            int ps_upper_1 = lane & 4;
            float ps_keep_1 = ((ps_upper_1 != 0) ? q[2 + i_1_1] : q[i_1_1]);
            float ps_send_1 = ((ps_upper_1 != 0) ? q[i_1_1] : q[2 + i_1_1]);
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, ps_send_1, 4);
            q[i_1_1] = ps_keep_1 + _shfl_xor_sync_11;
        }
        #pragma unroll
        for (int i_2_1 = 0; i_2_1 < 1; i_2_1++) {
            int ps_upper_2 = lane & 2;
            float ps_keep_2 = ((ps_upper_2 != 0) ? q[1 + i_2_1] : q[i_2_1]);
            float ps_send_2 = ((ps_upper_2 != 0) ? q[i_2_1] : q[1 + i_2_1]);
            float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, ps_send_2, 2);
            q[i_2_1] = ps_keep_2 + _shfl_xor_sync_12;
        }
        float ps_own = q[0];
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, ps_own, 1);
        ps_own = ps_own + _shfl_xor_sync_13;
        float tv_word = ps_own;
        int tv_row = xs * 224;
        if ((lane & 1) == 0) {
            wtab[tv_row + warp * 16 + (lane >> 4) * 8 + (lane >> 1 & 7)] = tv_word;
        }
        __syncthreads();
        float tr_w[14];
        #pragma unroll
        for (int wi = 0; wi < 14; wi++) {
            tr_w[wi] = wtab[tv_row + jl + wi * 16];
        }
        tr_w[0] = tr_w[0] + tr_w[1];
        tr_w[2] = tr_w[2] + tr_w[3];
        tr_w[4] = tr_w[4] + tr_w[5];
        tr_w[6] = tr_w[6] + tr_w[7];
        tr_w[8] = tr_w[8] + tr_w[9];
        tr_w[10] = tr_w[10] + tr_w[11];
        tr_w[12] = tr_w[12] + tr_w[13];
        tr_w[0] = tr_w[0] + tr_w[2];
        tr_w[4] = tr_w[4] + tr_w[6];
        tr_w[8] = tr_w[8] + tr_w[10];
        tr_w[0] = tr_w[0] + tr_w[4];
        tr_w[8] = tr_w[8] + tr_w[12];
        tr_w[0] = tr_w[0] + tr_w[8];
        float tv_ss = tr_w[0];
        float tr_w_0[14];
        #pragma unroll
        for (int wi_1 = 0; wi_1 < 14; wi_1++) {
            tr_w_0[wi_1] = wtab[tv_row + 8 + jl + wi_1 * 16];
        }
        tr_w_0[0] = tr_w_0[0] + tr_w_0[1];
        tr_w_0[2] = tr_w_0[2] + tr_w_0[3];
        tr_w_0[4] = tr_w_0[4] + tr_w_0[5];
        tr_w_0[6] = tr_w_0[6] + tr_w_0[7];
        tr_w_0[8] = tr_w_0[8] + tr_w_0[9];
        tr_w_0[10] = tr_w_0[10] + tr_w_0[11];
        tr_w_0[12] = tr_w_0[12] + tr_w_0[13];
        tr_w_0[0] = tr_w_0[0] + tr_w_0[2];
        tr_w_0[4] = tr_w_0[4] + tr_w_0[6];
        tr_w_0[8] = tr_w_0[8] + tr_w_0[10];
        tr_w_0[0] = tr_w_0[0] + tr_w_0[4];
        tr_w_0[8] = tr_w_0[8] + tr_w_0[12];
        tr_w_0[0] = tr_w_0[0] + tr_w_0[8];
        float tv_d = tr_w_0[0];
        xs = 1 - xs;
        float _fdiv_15 = __fdiv_rn(tv_ss, 7168.0f);
        float _sqrt_14;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_14) : "f"(_fdiv_15 + eps));
        float tk_r = __frcp_rn(_sqrt_14);
        float tk_s = tk_r * tv_d;
        float tk_m = ((act != 0) ? tk_s : -3.4e+38f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tk_m = max_noftz(tk_m, __shfl_xor_sync(0xFFFFFFFF, tk_m, offset));
        float _exp_16 = __nv_expf(tk_s - tk_m);
        float tk_e = ((act != 0) ? _exp_16 : 0.0f);
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
                d[tk_at] = tv_d;
            }
        }
        pj[0] = 0.0f;
        if (0 < n) {
            float _shfl_sync_17 = __shfl_sync(0xFFFFFFFF, tk_p, 0);
            pj[0] = _shfl_sync_17;
        }
        pj[1] = 0.0f;
        if (1 < n) {
            float _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, tk_p, 1);
            pj[1] = _shfl_sync_18;
        }
        pj[2] = 0.0f;
        if (2 < n) {
            float _shfl_sync_19 = __shfl_sync(0xFFFFFFFF, tk_p, 2);
            pj[2] = _shfl_sync_19;
        }
        pj[3] = 0.0f;
        if (3 < n) {
            float _shfl_sync_20 = __shfl_sync(0xFFFFFFFF, tk_p, 3);
            pj[3] = _shfl_sync_20;
        }
        pj[4] = 0.0f;
        if (4 < n) {
            float _shfl_sync_21 = __shfl_sync(0xFFFFFFFF, tk_p, 4);
            pj[4] = _shfl_sync_21;
        }
        pj[5] = 0.0f;
        if (5 < n) {
            float _shfl_sync_22 = __shfl_sync(0xFFFFFFFF, tk_p, 5);
            pj[5] = _shfl_sync_22;
        }
        pj[6] = 0.0f;
        if (6 < n) {
            float _shfl_sync_23 = __shfl_sync(0xFFFFFFFF, tk_p, 6);
            pj[6] = _shfl_sync_23;
        }
        pj[7] = 0.0f;
        if (7 < n) {
            float _shfl_sync_24 = __shfl_sync(0xFFFFFFFF, tk_p, 7);
            pj[7] = _shfl_sync_24;
        }
        float to_o_0[8];
        #pragma unroll
        for (int i_3_1 = 0; i_3_1 < 8; i_3_1++) {
            to_o_0[i_3_1] = 0.0f;
        }
        if (0 < n) {
            float2 tj_p2_0 = make_float2(pj[0], pj[0]);
            #pragma unroll
            for (int m_0 = 0; m_0 < 4; m_0++) {
                float2 tj_o_0 = fma_f32x2_rn_noftz(tj_p2_0, cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_0])), make_float2(to_o_0[2 * m_0], to_o_0[2 * m_0 + 1]));
                to_o_0[2 * m_0] = tj_o_0.x;
                to_o_0[2 * m_0 + 1] = tj_o_0.y;
            }
        }
        if (1 < n) {
            float2 tj_p2_1 = make_float2(pj[1], pj[1]);
            #pragma unroll
            for (int m_1_1 = 0; m_1_1 < 4; m_1_1++) {
                float2 tj_o_1 = fma_f32x2_rn_noftz(tj_p2_1, cake_bf16x2_to_f32x2(__as_bf16x2(vw[8 + m_1_1])), make_float2(to_o_0[2 * m_1_1], to_o_0[2 * m_1_1 + 1]));
                to_o_0[2 * m_1_1] = tj_o_1.x;
                to_o_0[2 * m_1_1 + 1] = tj_o_1.y;
            }
        }
        if (2 < n) {
            float2 tj_p2_2 = make_float2(pj[2], pj[2]);
            #pragma unroll
            for (int m_2_1 = 0; m_2_1 < 4; m_2_1++) {
                float2 tj_o_2 = fma_f32x2_rn_noftz(tj_p2_2, cake_bf16x2_to_f32x2(__as_bf16x2(vw[16 + m_2_1])), make_float2(to_o_0[2 * m_2_1], to_o_0[2 * m_2_1 + 1]));
                to_o_0[2 * m_2_1] = tj_o_2.x;
                to_o_0[2 * m_2_1 + 1] = tj_o_2.y;
            }
        }
        if (3 < n) {
            float2 tj_p2_3 = make_float2(pj[3], pj[3]);
            #pragma unroll
            for (int m_3_1 = 0; m_3_1 < 4; m_3_1++) {
                float2 tj_o_3 = fma_f32x2_rn_noftz(tj_p2_3, cake_bf16x2_to_f32x2(__as_bf16x2(vw[24 + m_3_1])), make_float2(to_o_0[2 * m_3_1], to_o_0[2 * m_3_1 + 1]));
                to_o_0[2 * m_3_1] = tj_o_3.x;
                to_o_0[2 * m_3_1 + 1] = tj_o_3.y;
            }
        }
        if (4 < n) {
            float2 tj_p2_4 = make_float2(pj[4], pj[4]);
            #pragma unroll
            for (int m_4_1 = 0; m_4_1 < 4; m_4_1++) {
                float2 tj_o_4 = fma_f32x2_rn_noftz(tj_p2_4, cake_bf16x2_to_f32x2(__as_bf16x2(vw[32 + m_4_1])), make_float2(to_o_0[2 * m_4_1], to_o_0[2 * m_4_1 + 1]));
                to_o_0[2 * m_4_1] = tj_o_4.x;
                to_o_0[2 * m_4_1 + 1] = tj_o_4.y;
            }
        }
        if (5 < n) {
            float2 tj_p2_5 = make_float2(pj[5], pj[5]);
            #pragma unroll
            for (int m_5_1 = 0; m_5_1 < 4; m_5_1++) {
                float2 tj_o_5 = fma_f32x2_rn_noftz(tj_p2_5, cake_bf16x2_to_f32x2(__as_bf16x2(vw[40 + m_5_1])), make_float2(to_o_0[2 * m_5_1], to_o_0[2 * m_5_1 + 1]));
                to_o_0[2 * m_5_1] = tj_o_5.x;
                to_o_0[2 * m_5_1 + 1] = tj_o_5.y;
            }
        }
        if (6 < n) {
            float2 tj_p2_6 = make_float2(pj[6], pj[6]);
            #pragma unroll
            for (int m_6_1 = 0; m_6_1 < 4; m_6_1++) {
                float2 tj_o_6 = fma_f32x2_rn_noftz(tj_p2_6, cake_bf16x2_to_f32x2(__as_bf16x2(vw[48 + m_6_1])), make_float2(to_o_0[2 * m_6_1], to_o_0[2 * m_6_1 + 1]));
                to_o_0[2 * m_6_1] = tj_o_6.x;
                to_o_0[2 * m_6_1 + 1] = tj_o_6.y;
            }
        }
        if (7 < n) {
            float2 tj_p2_7 = make_float2(pj[7], pj[7]);
            #pragma unroll
            for (int m_7_1 = 0; m_7_1 < 4; m_7_1++) {
                float2 tj_o_7 = fma_f32x2_rn_noftz(tj_p2_7, cake_bf16x2_to_f32x2(__as_bf16x2(vw[56 + m_7_1])), make_float2(to_o_0[2 * m_7_1], to_o_0[2 * m_7_1 + 1]));
                to_o_0[2 * m_7_1] = tj_o_7.x;
                to_o_0[2 * m_7_1 + 1] = tj_o_7.y;
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
        for (int i_4_1 = 0; i_4_1 < 8; i_4_1++) {
            to_o_1[i_4_1] = 0.0f;
        }
        if (0 < n) {
            float2 tj_p2_8 = make_float2(pj[0], pj[0]);
            #pragma unroll
            for (int m_8_1 = 0; m_8_1 < 4; m_8_1++) {
                float2 tj_o_8 = fma_f32x2_rn_noftz(tj_p2_8, cake_bf16x2_to_f32x2(__as_bf16x2(vw[4 + m_8_1])), make_float2(to_o_1[2 * m_8_1], to_o_1[2 * m_8_1 + 1]));
                to_o_1[2 * m_8_1] = tj_o_8.x;
                to_o_1[2 * m_8_1 + 1] = tj_o_8.y;
            }
        }
        if (1 < n) {
            float2 tj_p2_9 = make_float2(pj[1], pj[1]);
            #pragma unroll
            for (int m_9_1 = 0; m_9_1 < 4; m_9_1++) {
                float2 tj_o_9 = fma_f32x2_rn_noftz(tj_p2_9, cake_bf16x2_to_f32x2(__as_bf16x2(vw[12 + m_9_1])), make_float2(to_o_1[2 * m_9_1], to_o_1[2 * m_9_1 + 1]));
                to_o_1[2 * m_9_1] = tj_o_9.x;
                to_o_1[2 * m_9_1 + 1] = tj_o_9.y;
            }
        }
        if (2 < n) {
            float2 tj_p2_10 = make_float2(pj[2], pj[2]);
            #pragma unroll
            for (int m_10_1 = 0; m_10_1 < 4; m_10_1++) {
                float2 tj_o_10 = fma_f32x2_rn_noftz(tj_p2_10, cake_bf16x2_to_f32x2(__as_bf16x2(vw[20 + m_10_1])), make_float2(to_o_1[2 * m_10_1], to_o_1[2 * m_10_1 + 1]));
                to_o_1[2 * m_10_1] = tj_o_10.x;
                to_o_1[2 * m_10_1 + 1] = tj_o_10.y;
            }
        }
        if (3 < n) {
            float2 tj_p2_11 = make_float2(pj[3], pj[3]);
            #pragma unroll
            for (int m_11_1 = 0; m_11_1 < 4; m_11_1++) {
                float2 tj_o_11 = fma_f32x2_rn_noftz(tj_p2_11, cake_bf16x2_to_f32x2(__as_bf16x2(vw[28 + m_11_1])), make_float2(to_o_1[2 * m_11_1], to_o_1[2 * m_11_1 + 1]));
                to_o_1[2 * m_11_1] = tj_o_11.x;
                to_o_1[2 * m_11_1 + 1] = tj_o_11.y;
            }
        }
        if (4 < n) {
            float2 tj_p2_12 = make_float2(pj[4], pj[4]);
            #pragma unroll
            for (int m_12_1 = 0; m_12_1 < 4; m_12_1++) {
                float2 tj_o_12 = fma_f32x2_rn_noftz(tj_p2_12, cake_bf16x2_to_f32x2(__as_bf16x2(vw[36 + m_12_1])), make_float2(to_o_1[2 * m_12_1], to_o_1[2 * m_12_1 + 1]));
                to_o_1[2 * m_12_1] = tj_o_12.x;
                to_o_1[2 * m_12_1 + 1] = tj_o_12.y;
            }
        }
        if (5 < n) {
            float2 tj_p2_13 = make_float2(pj[5], pj[5]);
            #pragma unroll
            for (int m_13_1 = 0; m_13_1 < 4; m_13_1++) {
                float2 tj_o_13 = fma_f32x2_rn_noftz(tj_p2_13, cake_bf16x2_to_f32x2(__as_bf16x2(vw[44 + m_13_1])), make_float2(to_o_1[2 * m_13_1], to_o_1[2 * m_13_1 + 1]));
                to_o_1[2 * m_13_1] = tj_o_13.x;
                to_o_1[2 * m_13_1 + 1] = tj_o_13.y;
            }
        }
        if (6 < n) {
            float2 tj_p2_14 = make_float2(pj[6], pj[6]);
            #pragma unroll
            for (int m_14_1 = 0; m_14_1 < 4; m_14_1++) {
                float2 tj_o_14 = fma_f32x2_rn_noftz(tj_p2_14, cake_bf16x2_to_f32x2(__as_bf16x2(vw[52 + m_14_1])), make_float2(to_o_1[2 * m_14_1], to_o_1[2 * m_14_1 + 1]));
                to_o_1[2 * m_14_1] = tj_o_14.x;
                to_o_1[2 * m_14_1 + 1] = tj_o_14.y;
            }
        }
        if (7 < n) {
            float2 tj_p2_15 = make_float2(pj[7], pj[7]);
            #pragma unroll
            for (int m_15_1 = 0; m_15_1 < 4; m_15_1++) {
                float2 tj_o_15 = fma_f32x2_rn_noftz(tj_p2_15, cake_bf16x2_to_f32x2(__as_bf16x2(vw[60 + m_15_1])), make_float2(to_o_1[2 * m_15_1], to_o_1[2 * m_15_1 + 1]));
                to_o_1[2 * m_15_1] = tj_o_15.x;
                to_o_1[2 * m_15_1 + 1] = tj_o_15.y;
            }
        }
        unsigned int ss_packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_o_1[_lp*2 + 0], to_o_1[_lp*2+1 + 0]));
            ss_packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + ((long long)t * 7168ll + (long long)(base + 3584)))[0] = reinterpret_cast<int4*>(ss_packed_0)[0];
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
}

} // extern "C"
