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
kernel_cake_engram_gate_train_d025b699fdc9fd358684(__nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, int* __restrict__ q_weight, int* __restrict__ k_weight, bool* __restrict__ mask, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dkey, __nv_bfloat16* __restrict__ dvalue, float* __restrict__ dw_part, int T, float eps, float clamp, int wmode, long long q_weight_size_p0, long long q_weight_size_p1, long long k_weight_size_p0, long long k_weight_size_p1, long long key_stride_p0, long long key_stride_p1, long long value_stride_p0, long long dkey_stride_p0, long long dkey_stride_p1, long long dvalue_stride_p0, long long mask_size_p0)
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
    float* fac = reinterpret_cast<float*>(smem_raw + 43424);
    const int fac_addr = smem + 43424;
    unsigned int* stg = reinterpret_cast<unsigned int*>(smem_raw + 43488);
    const int stage_addr = smem + 43488;
    unsigned int* fsw = reinterpret_cast<unsigned int*>(smem_raw + 74208);
    const int fs_addr = smem + 74208;
    unsigned int* nxt = reinterpret_cast<unsigned int*>(smem_raw + 74848);
    const int nxt_addr = smem + 74848;

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
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(fs_addr + (unsigned int)(4 * tid)), "r"((unsigned int)first) : "memory");
    float wf[32];
    unsigned int gw[16];
    unsigned int xw[16];
    unsigned int kw[16];
    unsigned int vw[4];
    float dwa[32];
    if (wmode == 0) {
        unsigned int lp_q[32];
        unsigned int lp_k[32];
        {
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    {
                        uint32_t _uv4_0_0;
                        uint32_t _uv4_0_1;
                        uint32_t _uv4_0_2;
                        uint32_t _uv4_0_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s * 5120 + col + 4 * half)) + (0))) : "memory");
                        lp_q[8 * s + 4 * half + 0] = _uv4_0_0;
                        lp_q[8 * s + 4 * half + 1] = _uv4_0_1;
                        lp_q[8 * s + 4 * half + 2] = _uv4_0_2;
                        lp_q[8 * s + 4 * half + 3] = _uv4_0_3;
                    }
                }
            }
        }
        {
            #pragma unroll
            for (int s_1 = 0; s_1 < 4; s_1++) {
                #pragma unroll
                for (int half_1 = 0; half_1 < 2; half_1++) {
                    {
                        uint32_t _uv4_1_0;
                        uint32_t _uv4_1_1;
                        uint32_t _uv4_1_2;
                        uint32_t _uv4_1_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_1 * 5120 + col + 4 * half_1)) + (0))) : "memory");
                        lp_k[8 * s_1 + 4 * half_1 + 0] = _uv4_1_0;
                        lp_k[8 * s_1 + 4 * half_1 + 1] = _uv4_1_1;
                        lp_k[8 * s_1 + 4 * half_1 + 2] = _uv4_1_2;
                        lp_k[8 * s_1 + 4 * half_1 + 3] = _uv4_1_3;
                    }
                }
            }
        }
        float lp_qf[32];
        {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                lp_qf[i] = __uint_as_float(lp_q[i]);
            }
        }
        {
            #pragma unroll
            for (int i_1 = 0; i_1 < 32; i_1++) {
                wf[i_1] = __uint_as_float(lp_k[i_1]);
            }
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 32; i_2++) {
            wf[i_2] = mul_f32_rn_noftz(lp_qf[i_2], wf[i_2]);
        }
    } else if (wmode == 1) {
        unsigned int lp_q_1[32];
        unsigned int lp_k_1[32];
        {
            #pragma unroll
            for (int s_2 = 0; s_2 < 4; s_2++) {
                {
                    uint32_t _uv4_2_0;
                    uint32_t _uv4_2_1;
                    uint32_t _uv4_2_2;
                    uint32_t _uv4_2_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s_2 * 2560 + (col >> 1))) + (0))) : "memory");
                    lp_q_1[4 * s_2 + 0] = _uv4_2_0;
                    lp_q_1[4 * s_2 + 1] = _uv4_2_1;
                    lp_q_1[4 * s_2 + 2] = _uv4_2_2;
                    lp_q_1[4 * s_2 + 3] = _uv4_2_3;
                }
            }
        }
        {
            #pragma unroll
            for (int s_3 = 0; s_3 < 4; s_3++) {
                #pragma unroll
                for (int half_2 = 0; half_2 < 2; half_2++) {
                    {
                        uint32_t _uv4_3_0;
                        uint32_t _uv4_3_1;
                        uint32_t _uv4_3_2;
                        uint32_t _uv4_3_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_3_0), "=r"(_uv4_3_1), "=r"(_uv4_3_2), "=r"(_uv4_3_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_3 * 5120 + col + 4 * half_2)) + (0))) : "memory");
                        lp_k_1[8 * s_3 + 4 * half_2 + 0] = _uv4_3_0;
                        lp_k_1[8 * s_3 + 4 * half_2 + 1] = _uv4_3_1;
                        lp_k_1[8 * s_3 + 4 * half_2 + 2] = _uv4_3_2;
                        lp_k_1[8 * s_3 + 4 * half_2 + 3] = _uv4_3_3;
                    }
                }
            }
        }
        float lp_qf_1[32];
        {
            #pragma unroll
            for (int s_4 = 0; s_4 < 4; s_4++) {
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    float2 ww_v = cake_bf16x2_to_f32x2(__as_bf16x2(lp_q_1[4 * s_4 + m]));
                    lp_qf_1[8 * s_4 + 2 * m] = ww_v.x;
                    lp_qf_1[8 * s_4 + 2 * m + 1] = ww_v.y;
                }
            }
        }
        {
            #pragma unroll
            for (int i_3 = 0; i_3 < 32; i_3++) {
                wf[i_3] = __uint_as_float(lp_k_1[i_3]);
            }
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 32; i_4++) {
            wf[i_4] = mul_f32_rn_noftz(lp_qf_1[i_4], wf[i_4]);
        }
    } else {
        if (wmode == 2) {
            unsigned int lp_q_2[32];
            unsigned int lp_k_2[32];
            {
                #pragma unroll
                for (int s_5 = 0; s_5 < 4; s_5++) {
                    #pragma unroll
                    for (int half_3 = 0; half_3 < 2; half_3++) {
                        {
                            uint32_t _uv4_4_0;
                            uint32_t _uv4_4_1;
                            uint32_t _uv4_4_2;
                            uint32_t _uv4_4_3;
                            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_uv4_4_0), "=r"(_uv4_4_1), "=r"(_uv4_4_2), "=r"(_uv4_4_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s_5 * 5120 + col + 4 * half_3)) + (0))) : "memory");
                            lp_q_2[8 * s_5 + 4 * half_3 + 0] = _uv4_4_0;
                            lp_q_2[8 * s_5 + 4 * half_3 + 1] = _uv4_4_1;
                            lp_q_2[8 * s_5 + 4 * half_3 + 2] = _uv4_4_2;
                            lp_q_2[8 * s_5 + 4 * half_3 + 3] = _uv4_4_3;
                        }
                    }
                }
            }
            {
                #pragma unroll
                for (int s_6 = 0; s_6 < 4; s_6++) {
                    {
                        uint32_t _uv4_5_0;
                        uint32_t _uv4_5_1;
                        uint32_t _uv4_5_2;
                        uint32_t _uv4_5_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_5_0), "=r"(_uv4_5_1), "=r"(_uv4_5_2), "=r"(_uv4_5_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_6 * 2560 + (col >> 1))) + (0))) : "memory");
                        lp_k_2[4 * s_6 + 0] = _uv4_5_0;
                        lp_k_2[4 * s_6 + 1] = _uv4_5_1;
                        lp_k_2[4 * s_6 + 2] = _uv4_5_2;
                        lp_k_2[4 * s_6 + 3] = _uv4_5_3;
                    }
                }
            }
            float lp_qf_2[32];
            {
                #pragma unroll
                for (int i_5 = 0; i_5 < 32; i_5++) {
                    lp_qf_2[i_5] = __uint_as_float(lp_q_2[i_5]);
                }
            }
            {
                #pragma unroll
                for (int s_7 = 0; s_7 < 4; s_7++) {
                    #pragma unroll
                    for (int m_1 = 0; m_1 < 4; m_1++) {
                        float2 ww_v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(lp_k_2[4 * s_7 + m_1]));
                        wf[8 * s_7 + 2 * m_1] = ww_v_1.x;
                        wf[8 * s_7 + 2 * m_1 + 1] = ww_v_1.y;
                    }
                }
            }
            #pragma unroll
            for (int i_6 = 0; i_6 < 32; i_6++) {
                wf[i_6] = mul_f32_rn_noftz(lp_qf_2[i_6], wf[i_6]);
            }
        } else {
            unsigned int lp_q_3[32];
            unsigned int lp_k_3[32];
            {
                #pragma unroll
                for (int s_8 = 0; s_8 < 4; s_8++) {
                    {
                        uint32_t _uv4_6_0;
                        uint32_t _uv4_6_1;
                        uint32_t _uv4_6_2;
                        uint32_t _uv4_6_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_6_0), "=r"(_uv4_6_1), "=r"(_uv4_6_2), "=r"(_uv4_6_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s_8 * 2560 + (col >> 1))) + (0))) : "memory");
                        lp_q_3[4 * s_8 + 0] = _uv4_6_0;
                        lp_q_3[4 * s_8 + 1] = _uv4_6_1;
                        lp_q_3[4 * s_8 + 2] = _uv4_6_2;
                        lp_q_3[4 * s_8 + 3] = _uv4_6_3;
                    }
                }
            }
            {
                #pragma unroll
                for (int s_9 = 0; s_9 < 4; s_9++) {
                    {
                        uint32_t _uv4_7_0;
                        uint32_t _uv4_7_1;
                        uint32_t _uv4_7_2;
                        uint32_t _uv4_7_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_7_0), "=r"(_uv4_7_1), "=r"(_uv4_7_2), "=r"(_uv4_7_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_9 * 2560 + (col >> 1))) + (0))) : "memory");
                        lp_k_3[4 * s_9 + 0] = _uv4_7_0;
                        lp_k_3[4 * s_9 + 1] = _uv4_7_1;
                        lp_k_3[4 * s_9 + 2] = _uv4_7_2;
                        lp_k_3[4 * s_9 + 3] = _uv4_7_3;
                    }
                }
            }
            float lp_qf_3[32];
            {
                #pragma unroll
                for (int s_10 = 0; s_10 < 4; s_10++) {
                    #pragma unroll
                    for (int m_2 = 0; m_2 < 4; m_2++) {
                        float2 ww_v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(lp_q_3[4 * s_10 + m_2]));
                        lp_qf_3[8 * s_10 + 2 * m_2] = ww_v_2.x;
                        lp_qf_3[8 * s_10 + 2 * m_2 + 1] = ww_v_2.y;
                    }
                }
            }
            {
                #pragma unroll
                for (int s_11 = 0; s_11 < 4; s_11++) {
                    #pragma unroll
                    for (int m_3 = 0; m_3 < 4; m_3++) {
                        float2 ww_v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(lp_k_3[4 * s_11 + m_3]));
                        wf[8 * s_11 + 2 * m_3] = ww_v_3.x;
                        wf[8 * s_11 + 2 * m_3 + 1] = ww_v_3.y;
                    }
                }
            }
            #pragma unroll
            for (int i_7 = 0; i_7 < 32; i_7++) {
                wf[i_7] = mul_f32_rn_noftz(lp_qf_3[i_7], wf[i_7]);
            }
        }
    }
    #pragma unroll
    for (int s_12 = 0; s_12 < 4; s_12++) {
        #pragma unroll
        for (int half_4 = 0; half_4 < 2; half_4++) {
            unsigned int bw_at = (unsigned int)(16 * ((2 * s_12 + half_4) * 160 + tid));
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(wslots_addr + bw_at), "f"(wf[8 * s_12 + 4 * half_4]), "f"(wf[8 * s_12 + 4 * half_4 + 1]), "f"(wf[8 * s_12 + 4 * half_4 + 2]), "f"(wf[8 * s_12 + 4 * half_4 + 3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dslots_addr + bw_at), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f) : "memory");
        }
    }
    float plain[16];
    #pragma unroll
    for (int i_8 = 12; i_8 < 16; i_8++) {
        plain[i_8] = 0.0f;
    }
    float ps[4];
    float pe[4];
    float tot[4];
    unsigned int zw[4];
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; i_9++) {
        zw[i_9] = 0;
    }
    int xk = 0;
    int keep = 1;
    unsigned int vn[4];
    int keep_n = 1;
    int ftok = ((first) < (T - 1) ? (first) : (T - 1));
    long long ls_at = (long long)ftok * 20480ll + (long long)col;
    #pragma unroll
    for (int s_13 = 0; s_13 < 4; s_13++) {
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(dout + (ls_at + (long long)(s_13 * 5120)) + 0);
            uint4* _vdst_8 = reinterpret_cast<uint4*>(&gw[4 * s_13]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_8[_blk] = _vptr_8[_blk];
            }
        }
    }
    long long ls_at_0 = (long long)ftok * 20480ll + (long long)col;
    #pragma unroll
    for (int s_14 = 0; s_14 < 4; s_14++) {
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + (ls_at_0 + (long long)(s_14 * 5120)) + 0);
            uint4* _vdst_9 = reinterpret_cast<uint4*>(&xw[4 * s_14]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_9[_blk] = _vptr_9[_blk];
            }
        }
    }
    long long lk_at = (long long)ftok * key_stride_p0 + (long long)col;
    #pragma unroll
    for (int s_15 = 0; s_15 < 4; s_15++) {
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(key + (lk_at + (long long)s_15 * key_stride_p1) + 0);
            uint4* _vdst_10 = reinterpret_cast<uint4*>(&kw[4 * s_15]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_10[_blk] = _vptr_10[_blk];
            }
        }
    }
    {
        const uint4* _vptr_11 = reinterpret_cast<const uint4*>(value + ((long long)ftok * value_stride_p0 + (long long)col) + 0);
        uint4* _vdst_11 = reinterpret_cast<uint4*>(&vw[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_11[_blk] = _vptr_11[_blk];
        }
    }
    int dn_head = (((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) < (T) ? ((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) : (T));
    int dn_words = (T - dn_head) / 16;
    int dn_tail0 = dn_head + 16 * dn_words;
    int dn_zero = 0;
    unsigned int dn_w[4];
    #pragma unroll 4
    for (int dn_j = tid; dn_j < dn_words; dn_j += 160) {
        {
            uint32_t _uv4_12_0;
            uint32_t _uv4_12_1;
            uint32_t _uv4_12_2;
            uint32_t _uv4_12_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_uv4_12_0), "=r"(_uv4_12_1), "=r"(_uv4_12_2), "=r"(_uv4_12_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(mask + (dn_head + 16 * dn_j)) + (0))) : "memory");
            dn_w[0 + 0] = _uv4_12_0;
            dn_w[0 + 1] = _uv4_12_1;
            dn_w[0 + 2] = _uv4_12_2;
            dn_w[0 + 3] = _uv4_12_3;
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 4; i_10++) {
            unsigned int dn_nz = ((dn_w[i_10] & 2139062143) + 2139062143 | dn_w[i_10]) & 2155905152u;
            dn_zero = dn_zero + 4 - (int)(unsigned int)__popc(dn_nz);
        }
    }
    int dn_at = tid;
    if (tid >= dn_head) {
        dn_at = dn_tail0 + (tid - dn_head);
    }
    if (dn_at < T) {
        if (tid < dn_head + (T - dn_tail0)) {
            int dn_byte = 0;
            if (mask[dn_at]) {
                dn_byte = 1;
            }
            if (dn_byte == 0) {
                dn_zero = dn_zero + 1;
            }
        }
    }
    int dn_wz = dn_zero;
    {
        uint32_t dn_wz_scan = (uint32_t)dn_wz;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, dn_wz_scan, 1, 32); if (lane >= 1) { dn_wz_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, dn_wz_scan, 2, 32); if (lane >= 2) { dn_wz_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, dn_wz_scan, 4, 32); if (lane >= 4) { dn_wz_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, dn_wz_scan, 8, 32); if (lane >= 8) { dn_wz_scan += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, dn_wz_scan, 16, 32); if (lane >= 16) { dn_wz_scan += peer; } }
        dn_wz = (int)dn_wz_scan;
    }
    if (lane == 31) {
        nxt[2 + warp] = (unsigned int)dn_wz;
    }
    if (warp == 1) {
        int pro[2];
        int sk_head = (((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) < (T) ? ((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) : (T));
        int sk_words = (T - sk_head) / 16;
        int sk_tail0 = sk_head + 16 * sk_words;
        unsigned int sk_w[4];
        pro[0] = -1;
        pro[1] = -1;
        int sk_base = 0;
        int sk_ch = sk_words + 1;
        if (0 < sk_tail0) {
            sk_ch = 1 + (-sk_head) / 16;
        }
        if (0 < sk_head) {
            sk_ch = 0;
        }
        #pragma unroll 1
        for (int sk_r = sk_ch; sk_r < sk_words + 2; sk_r += 32) {
            int sk_c = sk_r + lane;
            int sk_c0 = sk_head + 16 * (sk_c - 1);
            int sk_nb = 16;
            if (sk_c == 0) {
                sk_c0 = 0;
                sk_nb = sk_head;
            }
            if (sk_c >= sk_words + 1) {
                sk_c0 = sk_tail0;
                sk_nb = T - sk_tail0;
            }
            if (sk_c > sk_words + 1) {
                sk_nb = 0;
            }
            unsigned int sk_m = 0;
            if (sk_nb < 16) {
                #pragma unroll
                for (int b = 0; b < 15; b++) {
                    if (b < sk_nb) {
                        if (mask[sk_c0 + b]) {
                            sk_m = sk_m | 1 << (unsigned int)b;
                        }
                    }
                }
            } else {
                {
                    uint32_t _uv4_13_0;
                    uint32_t _uv4_13_1;
                    uint32_t _uv4_13_2;
                    uint32_t _uv4_13_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_13_0), "=r"(_uv4_13_1), "=r"(_uv4_13_2), "=r"(_uv4_13_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(mask + sk_c0) + (0))) : "memory");
                    sk_w[0 + 0] = _uv4_13_0;
                    sk_w[0 + 1] = _uv4_13_1;
                    sk_w[0 + 2] = _uv4_13_2;
                    sk_w[0 + 3] = _uv4_13_3;
                }
                #pragma unroll
                for (int i_11 = 0; i_11 < 4; i_11++) {
                    unsigned int sk_nz = ((sk_w[i_11] & 2139062143) + 2139062143 | sk_w[i_11]) & 2155905152u;
                    unsigned int sk_m4 = sk_nz >> 7 & 1 | sk_nz >> 14 & 2 | sk_nz >> 21 & 4 | sk_nz >> 28 & 8;
                    sk_m = sk_m | sk_m4 << (unsigned int)(4 * i_11);
                }
            }
            int _max_15 = ((-sk_c0) > (0) ? (-sk_c0) : (0));
            int _min_14 = ((_max_15) < (sk_nb) ? (_max_15) : (sk_nb));
            unsigned int sk_lo = (unsigned int)_min_14;
            unsigned int sk_span = (1 << (unsigned int)sk_nb) - (1 << sk_lo);
            unsigned int sk_sel = (sk_m ^ 0) & sk_span;
            int sk_cnt = (int)(unsigned int)__popc(sk_sel);
            int sk_incl = sk_cnt;
            {
                uint32_t sk_incl_scan = (uint32_t)sk_incl;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_scan, 1, 32); if (lane >= 1) { sk_incl_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_scan, 2, 32); if (lane >= 2) { sk_incl_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_scan, 4, 32); if (lane >= 4) { sk_incl_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_scan, 8, 32); if (lane >= 8) { sk_incl_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_scan, 16, 32); if (lane >= 16) { sk_incl_scan += peer; } }
                sk_incl = (int)sk_incl_scan;
            }
            int sk_t_0 = first + 1 - sk_base;
            unsigned int sk_hit_0 = __ballot_sync(0xFFFFFFFF, sk_incl >= sk_t_0);
            int sk_k_0 = sk_t_0 - (sk_incl - sk_cnt);
            unsigned int sk_s_0 = sk_sel;
            int sk_pos_0 = 0;
            #pragma unroll
            for (int sk_h_0 = 0; sk_h_0 < 4; sk_h_0++) {
                unsigned int sk_half_0 = 8 >> (unsigned int)sk_h_0;
                int sk_low_0 = (int)(unsigned int)__popc(sk_s_0 & (1 << sk_half_0) - 1);
                if (sk_k_0 > sk_low_0) {
                    sk_k_0 = sk_k_0 - sk_low_0;
                    sk_pos_0 = sk_pos_0 + (int)sk_half_0;
                    sk_s_0 = sk_s_0 >> sk_half_0;
                }
            }
            int _find_first_set_16 = __ffs(sk_hit_0);
            int sk_at_0 = __shfl_sync(0xFFFFFFFF, sk_c0 + sk_pos_0, _find_first_set_16 - 1);
            if (pro[0] < 0) {
                if (sk_hit_0 != 0) {
                    pro[0] = sk_at_0;
                }
            }
            int sk_t_1 = first + 1 + step - sk_base;
            unsigned int sk_hit_1 = __ballot_sync(0xFFFFFFFF, sk_incl >= sk_t_1);
            int sk_k_1 = sk_t_1 - (sk_incl - sk_cnt);
            unsigned int sk_s_1 = sk_sel;
            int sk_pos_1 = 0;
            #pragma unroll
            for (int sk_h_1 = 0; sk_h_1 < 4; sk_h_1++) {
                unsigned int sk_half_1 = 8 >> (unsigned int)sk_h_1;
                int sk_low_1 = (int)(unsigned int)__popc(sk_s_1 & (1 << sk_half_1) - 1);
                if (sk_k_1 > sk_low_1) {
                    sk_k_1 = sk_k_1 - sk_low_1;
                    sk_pos_1 = sk_pos_1 + (int)sk_half_1;
                    sk_s_1 = sk_s_1 >> sk_half_1;
                }
            }
            int _find_first_set_17 = __ffs(sk_hit_1);
            int sk_at_1 = __shfl_sync(0xFFFFFFFF, sk_c0 + sk_pos_1, _find_first_set_17 - 1);
            if (pro[1] < 0) {
                if (sk_hit_1 != 0) {
                    pro[1] = sk_at_1;
                }
            }
            if (pro[1] >= 0) {
                break;
            }
            int _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, sk_incl, 31);
            sk_base = sk_base + _shfl_sync_18;
        }
        if (lane == 0) {
            nxt[0] = (unsigned int)pro[0];
            nxt[1] = (unsigned int)pro[1];
        }
    }
    __syncthreads();
    uint32_t pw_w;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(pw_w) : "r"(static_cast<uint32_t>(nxt_addr)));
    int kept_0 = (int)pw_w;
    uint32_t pw_w_0;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(pw_w_0) : "r"(static_cast<uint32_t>(nxt_addr + 4)));
    keep_n = (int)pw_w_0;
    unsigned int pt_seen = 0;
    if (lane == 0) {
        #pragma unroll
        for (unsigned int pt_i = 0; pt_i < 5; pt_i++) {
            uint32_t pt_w;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(pt_w) : "r"(static_cast<uint32_t>(nxt_addr + 8 + 4 * pt_i)));
            pt_seen = pt_seen + pt_w;
        }
    }
    unsigned int _shfl_sync_19 = __shfl_sync(0xFFFFFFFF, pt_seen, 0);
    int zeros_n = (int)_shfl_sync_19;
    if (kept_0 >= 0) {
        if (kept_0 != ftok) {
            long long ls_at_1 = (long long)kept_0 * 20480ll + (long long)col;
            #pragma unroll
            for (int s_16 = 0; s_16 < 4; s_16++) {
                {
                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(dout + (ls_at_1 + (long long)(s_16 * 5120)) + 0);
                    uint4* _vdst_20 = reinterpret_cast<uint4*>(&gw[4 * s_16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_20[_blk] = _vptr_20[_blk];
                    }
                }
            }
            long long ls_at_2 = (long long)kept_0 * 20480ll + (long long)col;
            #pragma unroll
            for (int s_17 = 0; s_17 < 4; s_17++) {
                {
                    const uint4* _vptr_21 = reinterpret_cast<const uint4*>(x + (ls_at_2 + (long long)(s_17 * 5120)) + 0);
                    uint4* _vdst_21 = reinterpret_cast<uint4*>(&xw[4 * s_17]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_21[_blk] = _vptr_21[_blk];
                    }
                }
            }
            long long lk_at_0 = (long long)kept_0 * key_stride_p0 + (long long)col;
            #pragma unroll
            for (int s_18 = 0; s_18 < 4; s_18++) {
                {
                    const uint4* _vptr_22 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_18 * key_stride_p1) + 0);
                    uint4* _vdst_22 = reinterpret_cast<uint4*>(&kw[4 * s_18]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_22[_blk] = _vptr_22[_blk];
                    }
                }
            }
            {
                const uint4* _vptr_23 = reinterpret_cast<const uint4*>(value + ((long long)kept_0 * value_stride_p0 + (long long)col) + 0);
                uint4* _vdst_23 = reinterpret_cast<uint4*>(&vw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_23[_blk] = _vptr_23[_blk];
                }
            }
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    int tok_c = kept_0;
    #pragma unroll 1
    for (int pj = first; pj < T; pj += step) {
        if (tok_c < 0) {
            break;
        }
        long long tg_x = (long long)tok_c * 20480ll + (long long)col;
        long long tg_k = (long long)tok_c * dkey_stride_p0 + (long long)col;
        long long tg_v = (long long)tok_c * dvalue_stride_p0 + (long long)col;
        if (keep != 0) {
            #pragma unroll
            for (int s_19 = 0; s_19 < 4; s_19++) {
                #pragma unroll
                for (int half_5 = 0; half_5 < 2; half_5++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_19 + 4 * half_5])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_19 + 4 * half_5) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_19 + 4 * half_5) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_19 + 4 * half_5) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_19 + half_5) * 160 + tid))));
                }
            }
            #pragma unroll
            for (int s_20 = 0; s_20 < 4; s_20++) {
                float ss_h = 0.0f;
                float ss_k = 0.0f;
                #pragma unroll
                for (int m_4 = 0; m_4 < 4; m_4++) {
                    float _dot_24;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_24) : "r"(xw[4 * s_20 + m_4]), "r"(xw[4 * s_20 + m_4]), "f"(ss_h));
                    ss_h = _dot_24;
                    float _dot_25;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_25) : "r"(kw[4 * s_20 + m_4]), "r"(kw[4 * s_20 + m_4]), "f"(ss_k));
                    ss_k = _dot_25;
                }
                plain[s_20] = ss_h;
                plain[4 + s_20] = ss_k;
            }
            #pragma unroll
            for (int s_21 = 0; s_21 < 4; s_21++) {
                float sp_s = 0.0f;
                float sp_e = 0.0f;
                float sp_hk1_0[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_0[0]) : "h"(static_cast<uint16_t>((xw[4 * s_21]))), "h"(static_cast<uint16_t>((kw[4 * s_21]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_0[1]) : "h"(static_cast<uint16_t>((xw[4 * s_21]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_21]) >> 16)));
                float sp_w1_0 = wf[8 * s_21];
                float sp_p1_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_0) : "f"(sp_w1_0), "f"(sp_hk1_0[0]));
                float _neg_26;
                asm("neg.f32 %0, %1;" : "=f"(_neg_26) : "f"(sp_p1_0));
                float sp_q1_0;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_0) : "f"(sp_w1_0), "f"(sp_hk1_0[0]), "f"(_neg_26));
                float _add_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(0.0f), "f"(sp_p1_0));
                sp_s = _add_27;
                float _add_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(0.0f), "f"(sp_q1_0));
                sp_e = _add_28;
                float sp_w1_1 = wf[8 * s_21 + 1];
                float sp_p1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_1) : "f"(sp_w1_1), "f"(sp_hk1_0[1]));
                float _neg_29;
                asm("neg.f32 %0, %1;" : "=f"(_neg_29) : "f"(sp_p1_1));
                float sp_q1_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_1) : "f"(sp_w1_1), "f"(sp_hk1_0[1]), "f"(_neg_29));
                float ts_s;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(sp_s), "f"(sp_p1_1));
                float ts_bb;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(sp_s));
                float _sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(ts_s), "f"(ts_bb));
                float _sub_30;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(sp_s), "f"(_sub_31));
                float _sub_32;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(sp_p1_1), "f"(ts_bb));
                float ts_e;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_30), "f"(_sub_32));
                sp_s = ts_s;
                float _add_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(sp_q1_1), "f"(ts_e));
                float _add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(sp_e), "f"(_add_34));
                sp_e = _add_33;
                float sp_hk1_1[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_1[0]) : "h"(static_cast<uint16_t>((xw[4 * s_21 + 1]))), "h"(static_cast<uint16_t>((kw[4 * s_21 + 1]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_1[1]) : "h"(static_cast<uint16_t>((xw[4 * s_21 + 1]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_21 + 1]) >> 16)));
                float sp_w1_2 = wf[8 * s_21 + 2];
                float sp_p1_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_2) : "f"(sp_w1_2), "f"(sp_hk1_1[0]));
                float _neg_35;
                asm("neg.f32 %0, %1;" : "=f"(_neg_35) : "f"(sp_p1_2));
                float sp_q1_2;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_2) : "f"(sp_w1_2), "f"(sp_hk1_1[0]), "f"(_neg_35));
                float ts_s_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(sp_s), "f"(sp_p1_2));
                float ts_bb_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_0), "f"(sp_s));
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(ts_s_0), "f"(ts_bb_0));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(sp_s), "f"(_sub_37));
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(sp_p1_2), "f"(ts_bb_0));
                float ts_e_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_36), "f"(_sub_38));
                sp_s = ts_s_0;
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(sp_q1_2), "f"(ts_e_0));
                float _add_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(sp_e), "f"(_add_40));
                sp_e = _add_39;
                float sp_w1_3 = wf[8 * s_21 + 2 + 1];
                float sp_p1_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_3) : "f"(sp_w1_3), "f"(sp_hk1_1[1]));
                float _neg_41;
                asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(sp_p1_3));
                float sp_q1_3;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_3) : "f"(sp_w1_3), "f"(sp_hk1_1[1]), "f"(_neg_41));
                float ts_s_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(sp_s), "f"(sp_p1_3));
                float ts_bb_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_1), "f"(sp_s));
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(ts_s_1), "f"(ts_bb_1));
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(sp_s), "f"(_sub_43));
                float _sub_44;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(sp_p1_3), "f"(ts_bb_1));
                float ts_e_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_42), "f"(_sub_44));
                sp_s = ts_s_1;
                float _add_46;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(sp_q1_3), "f"(ts_e_1));
                float _add_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(sp_e), "f"(_add_46));
                sp_e = _add_45;
                float sp_hk1_2[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_2[0]) : "h"(static_cast<uint16_t>((xw[4 * s_21 + 2]))), "h"(static_cast<uint16_t>((kw[4 * s_21 + 2]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_2[1]) : "h"(static_cast<uint16_t>((xw[4 * s_21 + 2]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_21 + 2]) >> 16)));
                float sp_w1_4 = wf[8 * s_21 + 4];
                float sp_p1_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_4) : "f"(sp_w1_4), "f"(sp_hk1_2[0]));
                float _neg_47;
                asm("neg.f32 %0, %1;" : "=f"(_neg_47) : "f"(sp_p1_4));
                float sp_q1_4;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_4) : "f"(sp_w1_4), "f"(sp_hk1_2[0]), "f"(_neg_47));
                float ts_s_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(sp_s), "f"(sp_p1_4));
                float ts_bb_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_2), "f"(sp_s));
                float _sub_49;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(ts_s_2), "f"(ts_bb_2));
                float _sub_48;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(sp_s), "f"(_sub_49));
                float _sub_50;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(sp_p1_4), "f"(ts_bb_2));
                float ts_e_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_48), "f"(_sub_50));
                sp_s = ts_s_2;
                float _add_52;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(sp_q1_4), "f"(ts_e_2));
                float _add_51;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(sp_e), "f"(_add_52));
                sp_e = _add_51;
                float sp_w1_5 = wf[8 * s_21 + 4 + 1];
                float sp_p1_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_5) : "f"(sp_w1_5), "f"(sp_hk1_2[1]));
                float _neg_53;
                asm("neg.f32 %0, %1;" : "=f"(_neg_53) : "f"(sp_p1_5));
                float sp_q1_5;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_5) : "f"(sp_w1_5), "f"(sp_hk1_2[1]), "f"(_neg_53));
                float ts_s_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3) : "f"(sp_s), "f"(sp_p1_5));
                float ts_bb_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3) : "f"(ts_s_3), "f"(sp_s));
                float _sub_55;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(ts_s_3), "f"(ts_bb_3));
                float _sub_54;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(sp_s), "f"(_sub_55));
                float _sub_56;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(sp_p1_5), "f"(ts_bb_3));
                float ts_e_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_54), "f"(_sub_56));
                sp_s = ts_s_3;
                float _add_58;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(sp_q1_5), "f"(ts_e_3));
                float _add_57;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(sp_e), "f"(_add_58));
                sp_e = _add_57;
                float sp_hk1_3[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_3[0]) : "h"(static_cast<uint16_t>((xw[4 * s_21 + 3]))), "h"(static_cast<uint16_t>((kw[4 * s_21 + 3]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_3[1]) : "h"(static_cast<uint16_t>((xw[4 * s_21 + 3]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_21 + 3]) >> 16)));
                float sp_w1_6 = wf[8 * s_21 + 6];
                float sp_p1_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_6) : "f"(sp_w1_6), "f"(sp_hk1_3[0]));
                float _neg_59;
                asm("neg.f32 %0, %1;" : "=f"(_neg_59) : "f"(sp_p1_6));
                float sp_q1_6;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_6) : "f"(sp_w1_6), "f"(sp_hk1_3[0]), "f"(_neg_59));
                float ts_s_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_4) : "f"(sp_s), "f"(sp_p1_6));
                float ts_bb_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_4) : "f"(ts_s_4), "f"(sp_s));
                float _sub_61;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(ts_s_4), "f"(ts_bb_4));
                float _sub_60;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(sp_s), "f"(_sub_61));
                float _sub_62;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(sp_p1_6), "f"(ts_bb_4));
                float ts_e_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_60), "f"(_sub_62));
                sp_s = ts_s_4;
                float _add_64;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(sp_q1_6), "f"(ts_e_4));
                float _add_63;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(sp_e), "f"(_add_64));
                sp_e = _add_63;
                float sp_w1_7 = wf[8 * s_21 + 6 + 1];
                float sp_p1_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_7) : "f"(sp_w1_7), "f"(sp_hk1_3[1]));
                float _neg_65;
                asm("neg.f32 %0, %1;" : "=f"(_neg_65) : "f"(sp_p1_7));
                float sp_q1_7;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_7) : "f"(sp_w1_7), "f"(sp_hk1_3[1]), "f"(_neg_65));
                float ts_s_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_5) : "f"(sp_s), "f"(sp_p1_7));
                float ts_bb_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_5) : "f"(ts_s_5), "f"(sp_s));
                float _sub_67;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(ts_s_5), "f"(ts_bb_5));
                float _sub_66;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(sp_s), "f"(_sub_67));
                float _sub_68;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(sp_p1_7), "f"(ts_bb_5));
                float ts_e_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_5) : "f"(_sub_66), "f"(_sub_68));
                sp_s = ts_s_5;
                float _add_70;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(sp_q1_7), "f"(ts_e_5));
                float _add_69;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(sp_e), "f"(_add_70));
                sp_e = _add_69;
                ps[s_21] = sp_s;
                pe[s_21] = sp_e;
            }
            #pragma unroll
            for (int s_22 = 0; s_22 < 4; s_22++) {
                float tl_q = 0.0f;
                #pragma unroll
                for (int m_5 = 0; m_5 < 4; m_5++) {
                    float _dot_71;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_71) : "r"(gw[4 * s_22 + m_5]), "r"(vw[m_5]), "f"(tl_q));
                    tl_q = _dot_71;
                }
                plain[8 + s_22] = tl_q;
            }
            int pn_tok = keep_n;
            if (pn_tok >= 0) {
                long long st_at = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
                for (int s_23 = 0; s_23 < 4; s_23++) {
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(stage_addr + (unsigned int)(16 * (s_23 * 160 + tid))), "l"(dout + (st_at + (long long)s_23 * 5120ll)));
                }
                long long st_at_0 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
                for (int s_24 = 0; s_24 < 4; s_24++) {
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(stage_addr + (unsigned int)(16 * ((4 + s_24) * 160 + tid))), "l"(x + (st_at_0 + (long long)s_24 * 5120ll)));
                }
                long long st_at_1 = (long long)pn_tok * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_25 = 0; s_25 < 4; s_25++) {
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(stage_addr + (unsigned int)(16 * ((8 + s_25) * 160 + tid))), "l"(key + (st_at_1 + (long long)s_25 * key_stride_p1)));
                }
                asm volatile("cp.async.commit_group;");
                {
                    const uint4* _vptr_72 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_72 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_72[_blk] = _vptr_72[_blk];
                    }
                }
            }
            int xp_slot = xk % 2;
            int xp_row = xp_slot * 100 + warp * 20;
            #pragma unroll
            for (int i_0 = 0; i_0 < 8; i_0++) {
                int ws_upper_0 = lane & 16;
                float ws_keep_0 = ((ws_upper_0 != 0) ? plain[8 + i_0] : plain[i_0]);
                float ws_send_0 = ((ws_upper_0 != 0) ? plain[i_0] : plain[8 + i_0]);
                float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_73;
            }
            #pragma unroll
            for (int i_1_1 = 0; i_1_1 < 4; i_1_1++) {
                int ws_upper_1 = lane & 8;
                float ws_keep_1 = ((ws_upper_1 != 0) ? plain[4 + i_1_1] : plain[i_1_1]);
                float ws_send_1 = ((ws_upper_1 != 0) ? plain[i_1_1] : plain[4 + i_1_1]);
                float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1_1] = ws_keep_1 + _shfl_xor_sync_74;
            }
            #pragma unroll
            for (int i_2_1 = 0; i_2_1 < 2; i_2_1++) {
                int ws_upper_2 = lane & 4;
                float ws_keep_2 = ((ws_upper_2 != 0) ? plain[2 + i_2_1] : plain[i_2_1]);
                float ws_send_2 = ((ws_upper_2 != 0) ? plain[i_2_1] : plain[2 + i_2_1]);
                float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2_1] = ws_keep_2 + _shfl_xor_sync_75;
            }
            #pragma unroll
            for (int i_3_1 = 0; i_3_1 < 1; i_3_1++) {
                int ws_upper_3 = lane & 2;
                float ws_keep_3 = ((ws_upper_3 != 0) ? plain[1 + i_3_1] : plain[i_3_1]);
                float ws_send_3 = ((ws_upper_3 != 0) ? plain[i_3_1] : plain[1 + i_3_1]);
                float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
                plain[i_3_1] = ws_keep_3 + _shfl_xor_sync_76;
            }
            float ws_own = plain[0];
            float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_77;
            float xp_word = ws_own;
            if ((lane & 1) == 0) {
                if (lane >> 1 < 12) {
                    wtab[xp_row + (lane >> 1)] = xp_word;
                }
            }
            #pragma unroll
            for (int i_4_1 = 0; i_4_1 < 2; i_4_1++) {
                int wp_upper_0 = lane & 16;
                float wp_ks_0 = ((wp_upper_0 != 0) ? ps[2 + i_4_1] : ps[i_4_1]);
                float wp_ke_0 = ((wp_upper_0 != 0) ? pe[2 + i_4_1] : pe[i_4_1]);
                float wp_rs_0 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_0 != 0) ? ps[i_4_1] : ps[2 + i_4_1]), 16);
                float wp_re_0 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_0 != 0) ? pe[i_4_1] : pe[2 + i_4_1]), 16);
                float ts_s_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_6) : "f"(wp_ks_0), "f"(wp_rs_0));
                float ts_bb_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6) : "f"(ts_s_6), "f"(wp_ks_0));
                float _sub_79;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(ts_s_6), "f"(ts_bb_6));
                float _sub_78;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(wp_ks_0), "f"(_sub_79));
                float _sub_80;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(wp_rs_0), "f"(ts_bb_6));
                float ts_e_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6) : "f"(_sub_78), "f"(_sub_80));
                wp_ks_0 = ts_s_6;
                float _add_82;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(wp_ke_0), "f"(wp_re_0));
                float _add_81;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(_add_82), "f"(ts_e_6));
                wp_ke_0 = _add_81;
                ps[i_4_1] = wp_ks_0;
                pe[i_4_1] = wp_ke_0;
            }
            #pragma unroll
            for (int i_5_1 = 0; i_5_1 < 1; i_5_1++) {
                int wp_upper_1 = lane & 8;
                float wp_ks_1 = ((wp_upper_1 != 0) ? ps[1 + i_5_1] : ps[i_5_1]);
                float wp_ke_1 = ((wp_upper_1 != 0) ? pe[1 + i_5_1] : pe[i_5_1]);
                float wp_rs_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? ps[i_5_1] : ps[1 + i_5_1]), 8);
                float wp_re_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? pe[i_5_1] : pe[1 + i_5_1]), 8);
                float ts_s_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7) : "f"(wp_ks_1), "f"(wp_rs_1));
                float ts_bb_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7) : "f"(ts_s_7), "f"(wp_ks_1));
                float _sub_84;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(ts_s_7), "f"(ts_bb_7));
                float _sub_83;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(wp_ks_1), "f"(_sub_84));
                float _sub_85;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(wp_rs_1), "f"(ts_bb_7));
                float ts_e_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7) : "f"(_sub_83), "f"(_sub_85));
                wp_ks_1 = ts_s_7;
                float _add_87;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(wp_ke_1), "f"(wp_re_1));
                float _add_86;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(_add_87), "f"(ts_e_7));
                wp_ke_1 = _add_86;
                ps[i_5_1] = wp_ks_1;
                pe[i_5_1] = wp_ke_1;
            }
            float wp_s = ps[0];
            float wp_e = pe[0];
            float wp_bs_0 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 4);
            float wp_be_0 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 4);
            float ts_s_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8) : "f"(wp_s), "f"(wp_bs_0));
            float ts_bb_8;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8) : "f"(ts_s_8), "f"(wp_s));
            float _sub_89;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_89) : "f"(ts_s_8), "f"(ts_bb_8));
            float _sub_88;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(wp_s), "f"(_sub_89));
            float _sub_90;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(wp_bs_0), "f"(ts_bb_8));
            float ts_e_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8) : "f"(_sub_88), "f"(_sub_90));
            wp_s = ts_s_8;
            float _add_92;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(wp_e), "f"(wp_be_0));
            float _add_91;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(_add_92), "f"(ts_e_8));
            wp_e = _add_91;
            float wp_bs_1 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 2);
            float wp_be_1 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 2);
            float ts_s_6_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_6_1) : "f"(wp_s), "f"(wp_bs_1));
            float ts_bb_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6_1) : "f"(ts_s_6_1), "f"(wp_s));
            float _sub_94;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(ts_s_6_1), "f"(ts_bb_6_1));
            float _sub_93;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(wp_s), "f"(_sub_94));
            float _sub_95;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(wp_bs_1), "f"(ts_bb_6_1));
            float ts_e_6_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6_1) : "f"(_sub_93), "f"(_sub_95));
            wp_s = ts_s_6_1;
            float _add_97;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(wp_e), "f"(wp_be_1));
            float _add_96;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_add_97), "f"(ts_e_6_1));
            wp_e = _add_96;
            float wp_bs_2 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 1);
            float wp_be_2 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 1);
            float ts_s_7_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7_1) : "f"(wp_s), "f"(wp_bs_2));
            float ts_bb_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7_1) : "f"(ts_s_7_1), "f"(wp_s));
            float _sub_99;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_99) : "f"(ts_s_7_1), "f"(ts_bb_7_1));
            float _sub_98;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_98) : "f"(wp_s), "f"(_sub_99));
            float _sub_100;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(wp_bs_2), "f"(ts_bb_7_1));
            float ts_e_7_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7_1) : "f"(_sub_98), "f"(_sub_100));
            wp_s = ts_s_7_1;
            float _add_102;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(wp_e), "f"(wp_be_2));
            float _add_101;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_add_102), "f"(ts_e_7_1));
            wp_e = _add_101;
            if ((lane & 7) == 0) {
                wtab[xp_row + 12 + (lane >> 3)] = wp_s;
                wtab[xp_row + 16 + (lane >> 3)] = wp_e;
            }
            __syncthreads();
            if (warp == 0) {
                if (lane < 16) {
                    int xp_job = lane % 4;
                    int xp_dst = lane / 4;
                    int xp_base = xp_slot * 100 + 4 * xp_job;
                    unsigned int xp_to = recv_addr + (unsigned int)(4 * (xp_slot * 80 + rank * 20));
                    float xp_s[4];
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        xp_s[c] = wtab[xp_base + c];
                    }
                    if (xp_job < 3) {
                        #pragma unroll
                        for (int wi = 1; wi < 5; wi++) {
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 4; c_1++) {
                                xp_s[c_1] = xp_s[c_1] + wtab[xp_base + wi * 20 + c_1];
                            }
                        }
                    } else {
                        float xp_e[4];
                        #pragma unroll
                        for (int c_2 = 0; c_2 < 4; c_2++) {
                            float xp_fs = xp_s[c_2];
                            float xp_fe = wtab[xp_base + 4 + c_2];
                            #pragma unroll
                            for (int wi_1 = 1; wi_1 < 5; wi_1++) {
                                float ts_s_8_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8_1) : "f"(xp_fs), "f"(wtab[xp_base + wi_1 * 20 + c_2]));
                                float ts_bb_8_1;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8_1) : "f"(ts_s_8_1), "f"(xp_fs));
                                float _sub_104;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(ts_s_8_1), "f"(ts_bb_8_1));
                                float _sub_103;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_103) : "f"(xp_fs), "f"(_sub_104));
                                float _sub_105;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(wtab[xp_base + wi_1 * 20 + c_2]), "f"(ts_bb_8_1));
                                float ts_e_8_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8_1) : "f"(_sub_103), "f"(_sub_105));
                                xp_fs = ts_s_8_1;
                                float _add_107;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(xp_fe), "f"(wtab[xp_base + wi_1 * 20 + 4 + c_2]));
                                float _add_106;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(_add_107), "f"(ts_e_8_1));
                                xp_fe = _add_106;
                            }
                            xp_s[c_2] = xp_fs;
                            xp_e[c_2] = xp_fe;
                        }
                        uint32_t _cluster_mapa_108;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_108) : "r"(xp_to + (unsigned int)(16 * (xp_job + 1))), "r"(xp_dst));
                        uint32_t _mbar_peer_109;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_109) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_108), "r"(__float_as_uint(xp_e[0])), "r"(__float_as_uint(xp_e[1])), "r"(__float_as_uint(xp_e[2])), "r"(__float_as_uint(xp_e[3])), "r"(_mbar_peer_109) : "memory");
                    }
                    uint32_t _cluster_mapa_110;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_110) : "r"(xp_to + (unsigned int)(16 * xp_job)), "r"(xp_dst));
                    uint32_t _mbar_peer_111;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_111) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_110), "r"(__float_as_uint(xp_s[0])), "r"(__float_as_uint(xp_s[1])), "r"(__float_as_uint(xp_s[2])), "r"(__float_as_uint(xp_s[3])), "r"(_mbar_peer_111) : "memory");
                }
            }
            if (tid == 0) {
                mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 320);
            }
            if (warp == 1) {
                int tl_next[1];
                tl_next[0] = -1;
                if (keep_n >= 0) {
                    if (zeros_n != 0) {
                        int sk_head_1 = (((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) < (T) ? ((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) : (T));
                        int sk_words_1 = (T - sk_head_1) / 16;
                        int sk_tail0_1 = sk_head_1 + 16 * sk_words_1;
                        unsigned int sk_w_1[4];
                        tl_next[0] = -1;
                        int sk_base_1 = 0;
                        int sk_ch_1 = sk_words_1 + 1;
                        if (keep_n + 1 < sk_tail0_1) {
                            sk_ch_1 = 1 + (keep_n + 1 - sk_head_1) / 16;
                        }
                        if (keep_n + 1 < sk_head_1) {
                            sk_ch_1 = 0;
                        }
                        #pragma unroll 1
                        for (int sk_r_1 = sk_ch_1; sk_r_1 < sk_words_1 + 2; sk_r_1 += 32) {
                            int sk_c_1 = sk_r_1 + lane;
                            int sk_c0_1 = sk_head_1 + 16 * (sk_c_1 - 1);
                            int sk_nb_1 = 16;
                            if (sk_c_1 == 0) {
                                sk_c0_1 = 0;
                                sk_nb_1 = sk_head_1;
                            }
                            if (sk_c_1 >= sk_words_1 + 1) {
                                sk_c0_1 = sk_tail0_1;
                                sk_nb_1 = T - sk_tail0_1;
                            }
                            if (sk_c_1 > sk_words_1 + 1) {
                                sk_nb_1 = 0;
                            }
                            unsigned int sk_m_1 = 0;
                            if (sk_nb_1 < 16) {
                                #pragma unroll
                                for (int b_1 = 0; b_1 < 15; b_1++) {
                                    if (b_1 < sk_nb_1) {
                                        if (mask[sk_c0_1 + b_1]) {
                                            sk_m_1 = sk_m_1 | 1 << (unsigned int)b_1;
                                        }
                                    }
                                }
                            } else {
                                {
                                    uint32_t _uv4_112_0;
                                    uint32_t _uv4_112_1;
                                    uint32_t _uv4_112_2;
                                    uint32_t _uv4_112_3;
                                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                        : "=r"(_uv4_112_0), "=r"(_uv4_112_1), "=r"(_uv4_112_2), "=r"(_uv4_112_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(mask + sk_c0_1) + (0))) : "memory");
                                    sk_w_1[0 + 0] = _uv4_112_0;
                                    sk_w_1[0 + 1] = _uv4_112_1;
                                    sk_w_1[0 + 2] = _uv4_112_2;
                                    sk_w_1[0 + 3] = _uv4_112_3;
                                }
                                #pragma unroll
                                for (int i_12 = 0; i_12 < 4; i_12++) {
                                    unsigned int sk_nz_1 = ((sk_w_1[i_12] & 2139062143) + 2139062143 | sk_w_1[i_12]) & 2155905152u;
                                    unsigned int sk_m4_1 = sk_nz_1 >> 7 & 1 | sk_nz_1 >> 14 & 2 | sk_nz_1 >> 21 & 4 | sk_nz_1 >> 28 & 8;
                                    sk_m_1 = sk_m_1 | sk_m4_1 << (unsigned int)(4 * i_12);
                                }
                            }
                            int _max_114 = ((keep_n + 1 - sk_c0_1) > (0) ? (keep_n + 1 - sk_c0_1) : (0));
                            int _min_113 = ((_max_114) < (sk_nb_1) ? (_max_114) : (sk_nb_1));
                            unsigned int sk_lo_1 = (unsigned int)_min_113;
                            unsigned int sk_span_1 = (1 << (unsigned int)sk_nb_1) - (1 << sk_lo_1);
                            unsigned int sk_sel_1 = (sk_m_1 ^ 0) & sk_span_1;
                            int sk_cnt_1 = (int)(unsigned int)__popc(sk_sel_1);
                            int sk_incl_1 = sk_cnt_1;
                            {
                                uint32_t sk_incl_1_scan = (uint32_t)sk_incl_1;
                                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_1_scan, 1, 32); if (lane >= 1) { sk_incl_1_scan += peer; } }
                                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_1_scan, 2, 32); if (lane >= 2) { sk_incl_1_scan += peer; } }
                                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_1_scan, 4, 32); if (lane >= 4) { sk_incl_1_scan += peer; } }
                                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_1_scan, 8, 32); if (lane >= 8) { sk_incl_1_scan += peer; } }
                                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_1_scan, 16, 32); if (lane >= 16) { sk_incl_1_scan += peer; } }
                                sk_incl_1 = (int)sk_incl_1_scan;
                            }
                            int sk_t_2 = step - sk_base_1;
                            unsigned int sk_hit_2 = __ballot_sync(0xFFFFFFFF, sk_incl_1 >= sk_t_2);
                            int sk_k_2 = sk_t_2 - (sk_incl_1 - sk_cnt_1);
                            unsigned int sk_s_2 = sk_sel_1;
                            int sk_pos_2 = 0;
                            #pragma unroll
                            for (int sk_h_2 = 0; sk_h_2 < 4; sk_h_2++) {
                                unsigned int sk_half_2 = 8 >> (unsigned int)sk_h_2;
                                int sk_low_2 = (int)(unsigned int)__popc(sk_s_2 & (1 << sk_half_2) - 1);
                                if (sk_k_2 > sk_low_2) {
                                    sk_k_2 = sk_k_2 - sk_low_2;
                                    sk_pos_2 = sk_pos_2 + (int)sk_half_2;
                                    sk_s_2 = sk_s_2 >> sk_half_2;
                                }
                            }
                            int _find_first_set_115 = __ffs(sk_hit_2);
                            int sk_at_2 = __shfl_sync(0xFFFFFFFF, sk_c0_1 + sk_pos_2, _find_first_set_115 - 1);
                            if (tl_next[0] < 0) {
                                if (sk_hit_2 != 0) {
                                    tl_next[0] = sk_at_2;
                                }
                            }
                            if (tl_next[0] >= 0) {
                                break;
                            }
                            int _shfl_sync_116 = __shfl_sync(0xFFFFFFFF, sk_incl_1, 31);
                            sk_base_1 = sk_base_1 + _shfl_sync_116;
                        }
                    } else if (keep_n + step < T) {
                        tl_next[0] = keep_n + step;
                    }
                }
                if (lane == 0) {
                    nxt[0] = (unsigned int)tl_next[0];
                }
            }
            if (warp == 0) {
                int _stage_xbar = (int)((unsigned int)xk % 2);
                int _parity_xbar = (int)((unsigned int)xk / 2 % 2);
                asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(xbar_addr + (_stage_xbar) * 8), "r"(uint32_t(_parity_xbar)), "r"(uint32_t(10000000)) : "memory");
                asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
                int xt_from = xk % 2 * 80 + (lane & 3);
                #pragma unroll
                for (int q = 0; q < 3; q++) {
                    float xt_t = recv[xt_from + 4 * q];
                    #pragma unroll
                    for (int r = 1; r < 4; r++) {
                        xt_t = xt_t + recv[xt_from + r * 20 + 4 * q];
                    }
                    tot[q] = xt_t;
                }
                float xt_ts = recv[xt_from + 12];
                float xt_te = recv[xt_from + 16];
                #pragma unroll
                for (int r_1 = 1; r_1 < 4; r_1++) {
                    float ts_s_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_9) : "f"(xt_ts), "f"(recv[xt_from + r_1 * 20 + 12]));
                    float ts_bb_9;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_9) : "f"(ts_s_9), "f"(xt_ts));
                    float _sub_118;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_118) : "f"(ts_s_9), "f"(ts_bb_9));
                    float _sub_117;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_117) : "f"(xt_ts), "f"(_sub_118));
                    float _sub_119;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_119) : "f"(recv[xt_from + r_1 * 20 + 12]), "f"(ts_bb_9));
                    float ts_e_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9) : "f"(_sub_117), "f"(_sub_119));
                    xt_ts = ts_s_9;
                    float _add_121;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(xt_te), "f"(recv[xt_from + r_1 * 20 + 16]));
                    float _add_120;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_add_121), "f"(ts_e_9));
                    xt_te = _add_120;
                }
                float _add_122;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(xt_ts), "f"(xt_te));
                tot[3] = _add_122;
                float _fdiv_124 = __fdiv_rn(tot[0], 5120.0f);
                float _sqrt_123;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_123) : "f"(_fdiv_124 + eps));
                float sg_rh = __frcp_rn(_sqrt_123);
                float _fdiv_126 = __fdiv_rn(tot[1], 5120.0f);
                float _sqrt_125;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_125) : "f"(_fdiv_126 + eps));
                float sg_rk = __frcp_rn(_sqrt_125);
                float sg_dot;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sg_dot) : "f"(mul_f32_rn_noftz(tot[3], mul_f32_rn_noftz(sg_rh, sg_rk))), "f"(0.013975425f));
                float _abs_128;
                asm("abs.f32 %0, %1;" : "=f"(_abs_128) : "f"(sg_dot));
                float _max_127;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_127) : "f"(_abs_128), "f"(clamp));
                float sg_mag;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_127));
                float sg_u = __uint_as_float(__as_u32(sg_mag) & 2147483647 | __as_u32(sg_dot) & 2147483648u);
                float _neg_130;
                asm("neg.f32 %0, %1;" : "=f"(_neg_130) : "f"(sg_u));
                float _exp_129 = __nv_expf(_neg_130);
                float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_129);
                float sd_mag;
                asm("abs.f32 %0, %1;" : "=f"(sd_mag) : "f"(sg_dot));
                float sd_du = 0.0f;
                if (sd_mag > clamp) {
                    float _sqrt_132;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_132) : "f"(sd_mag));
                    float _fdiv_131 = __fdiv_rn(0.5f, _sqrt_132);
                    sd_du = _fdiv_131;
                }
                float sr_du = sd_du;
                float sr_c = tot[2] * sg_gate * (1.0f - sg_gate) * sr_du * sg_rh * sg_rk * 0.013975425f;
                float sr_ah = __fdiv_rn(tot[3] * (sg_rh * sg_rh), 5120.0f);
                float sr_ak = __fdiv_rn(tot[3] * (sg_rk * sg_rk), 5120.0f);
                if (lane < 4) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fac_addr + (unsigned int)(16 * lane)), "f"(sg_gate), "f"(sr_c), "f"(sr_ah), "f"(sr_ak) : "memory");
                }
            }
            __syncthreads();
            xk += 1;
            float tg_dv[8];
            #pragma unroll
            for (int i_13 = 0; i_13 < 8; i_13++) {
                tg_dv[i_13] = 0.0f;
            }
            #pragma unroll
            for (int s_26 = 0; s_26 < 4; s_26++) {
                float tg_f[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tg_f[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg_f[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg_f[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg_f[(0) + 3]))
                    : "r"(fac_addr + (unsigned int)(16 * s_26)));
                float tg_sg = tg_f[0];
                float tg_sc = tg_f[1];
                float tg_sh = tg_f[2];
                float tg_sk = tg_f[3];
                float2 tg_g2 = make_float2(tg_sg, tg_sg);
                float2 tg_c2 = make_float2(tg_sc, tg_sc);
                float _neg_133;
                asm("neg.f32 %0, %1;" : "=f"(_neg_133) : "f"(tg_sh));
                float _neg_134;
                asm("neg.f32 %0, %1;" : "=f"(_neg_134) : "f"(tg_sh));
                float2 tg_h2 = make_float2(_neg_133, _neg_134);
                float _neg_135;
                asm("neg.f32 %0, %1;" : "=f"(_neg_135) : "f"(tg_sk));
                float _neg_136;
                asm("neg.f32 %0, %1;" : "=f"(_neg_136) : "f"(tg_sk));
                float2 tg_k2 = make_float2(_neg_135, _neg_136);
                #pragma unroll
                for (int half_6 = 0; half_6 < 2; half_6++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_26 + 4 * half_6])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_26 + 4 * half_6) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_26 + 4 * half_6) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_26 + 4 * half_6) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_26 + half_6) * 160 + tid))));
                }
                #pragma unroll
                for (int half_7 = 0; half_7 < 2; half_7++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dwa[8 * s_26 + 4 * half_7])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_26 + 4 * half_7) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_26 + 4 * half_7) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_26 + 4 * half_7) + 3]))
                        : "r"(dslots_addr + (unsigned int)(16 * ((2 * s_26 + half_7) * 160 + tid))));
                }
                float tg_dx[8];
                float tg_dk[8];
                #pragma unroll
                for (int m_6 = 0; m_6 < 4; m_6++) {
                    float2 tg_hv = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_26 + m_6]));
                    float2 tg_kv = cake_bf16x2_to_f32x2(__as_bf16x2(kw[4 * s_26 + m_6]));
                    float2 tg_gv = cake_bf16x2_to_f32x2(__as_bf16x2(gw[4 * s_26 + m_6]));
                    float2 tg_wv = make_float2(wf[8 * s_26 + 2 * m_6], wf[8 * s_26 + 2 * m_6 + 1]);
                    float2 tg_ox = fma_f32x2_rn_noftz(tg_c2, fma_f32x2_rn_noftz(tg_h2, tg_hv, mul_f32x2_noftz(tg_wv, tg_kv)), tg_gv);
                    float2 _f32x2_mul_b_137 = fma_f32x2_rn_noftz(tg_k2, tg_kv, mul_f32x2_noftz(tg_wv, tg_hv));
                    float2 tg_ok;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_ok) : "l"(*(const unsigned long long*)&tg_c2), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
                    float2 tg_ov = fma_f32x2_rn_noftz(tg_g2, tg_gv, make_float2(tg_dv[2 * m_6], tg_dv[2 * m_6 + 1]));
                    float2 tg_hk;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_hk) : "l"(*(const unsigned long long*)&tg_hv), "l"(*(const unsigned long long*)&tg_kv));
                    float2 tg_ow = fma_f32x2_rn_noftz(tg_c2, tg_hk, make_float2(dwa[8 * s_26 + 2 * m_6], dwa[8 * s_26 + 2 * m_6 + 1]));
                    tg_dx[2 * m_6] = tg_ox.x;
                    tg_dk[2 * m_6] = tg_ok.x;
                    tg_dv[2 * m_6] = tg_ov.x;
                    dwa[8 * s_26 + 2 * m_6] = tg_ow.x;
                    tg_dx[2 * m_6 + 1] = tg_ox.y;
                    tg_dk[2 * m_6 + 1] = tg_ok.y;
                    tg_dv[2 * m_6 + 1] = tg_ov.y;
                    dwa[8 * s_26 + 2 * m_6 + 1] = tg_ow.y;
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
                reinterpret_cast<int4*>(dx + (tg_x + (long long)(s_26 * 5120)))[0] = reinterpret_cast<int4*>(tg_px)[0];
                reinterpret_cast<int4*>(dkey + (tg_k + (long long)s_26 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(tg_pk)[0];
                #pragma unroll
                for (int half_8 = 0; half_8 < 2; half_8++) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dslots_addr + (unsigned int)(16 * ((2 * s_26 + half_8) * 160 + tid))), "f"(dwa[8 * s_26 + 4 * half_8]), "f"(dwa[8 * s_26 + 4 * half_8 + 1]), "f"(dwa[8 * s_26 + 4 * half_8 + 2]), "f"(dwa[8 * s_26 + 4 * half_8 + 3]) : "memory");
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
            #pragma unroll
            for (int s_27 = 0; s_27 < 4; s_27++) {
                reinterpret_cast<int4*>(dx + (tg_x + (long long)(s_27 * 5120)))[0] = reinterpret_cast<int4*>(gw + (4 * s_27))[0];
                reinterpret_cast<int4*>(dkey + (tg_k + (long long)s_27 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(zw)[0];
            }
            reinterpret_cast<int4*>(dvalue + tg_v)[0] = reinterpret_cast<int4*>(zw)[0];
            int pn_tok_1 = keep_n;
            if (pn_tok_1 >= 0) {
                long long st_at_2 = (long long)pn_tok_1 * 20480ll + (long long)col;
                #pragma unroll
                for (int s_28 = 0; s_28 < 4; s_28++) {
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(stage_addr + (unsigned int)(16 * (s_28 * 160 + tid))), "l"(dout + (st_at_2 + (long long)s_28 * 5120ll)));
                }
                long long st_at_2_1 = (long long)pn_tok_1 * 20480ll + (long long)col;
                #pragma unroll
                for (int s_29 = 0; s_29 < 4; s_29++) {
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(stage_addr + (unsigned int)(16 * ((4 + s_29) * 160 + tid))), "l"(x + (st_at_2_1 + (long long)s_29 * 5120ll)));
                }
                long long st_at_3 = (long long)pn_tok_1 * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_30 = 0; s_30 < 4; s_30++) {
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(stage_addr + (unsigned int)(16 * ((8 + s_30) * 160 + tid))), "l"(key + (st_at_3 + (long long)s_30 * key_stride_p1)));
                }
                asm volatile("cp.async.commit_group;");
                {
                    const uint4* _vptr_138 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_138 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_138[_blk] = _vptr_138[_blk];
                    }
                }
            }
        }
        asm volatile("cp.async.wait_group 0;");
        if (keep_n >= 0) {
            #pragma unroll
            for (int s_31 = 0; s_31 < 4; s_31++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gw[4 * s_31])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_31) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_31) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_31) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (s_31 * 160 + tid))));
            }
            #pragma unroll
            for (int s_32 = 0; s_32 < 4; s_32++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xw[4 * s_32])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4 * s_32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4 * s_32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4 * s_32) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * ((4 + s_32) * 160 + tid))));
            }
            #pragma unroll
            for (int s_33 = 0; s_33 < 4; s_33++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kw[4 * s_33])), "=r"(*reinterpret_cast<uint32_t*>(&kw[(4 * s_33) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kw[(4 * s_33) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kw[(4 * s_33) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * ((8 + s_33) * 160 + tid))));
            }
            #pragma unroll
            for (int i_14 = 0; i_14 < 4; i_14++) {
                vw[i_14] = vn[i_14];
            }
        }
        tok_c = keep_n;
        uint32_t pw_w_1;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(pw_w_1) : "r"(static_cast<uint32_t>(nxt_addr)));
        keep_n = (int)pw_w_1;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    int pm0 = (int)cluster_id;
    unsigned int pt_seen_0 = 0;
    if (lane == 0) {
        #pragma unroll
        for (unsigned int pt_i_1 = 0; pt_i_1 < 5; pt_i_1++) {
            uint32_t pt_w_1;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(pt_w_1) : "r"(static_cast<uint32_t>(nxt_addr + 8 + 4 * pt_i_1)));
            pt_seen_0 = pt_seen_0 + pt_w_1;
        }
    }
    unsigned int _shfl_sync_139 = __shfl_sync(0xFFFFFFFF, pt_seen_0, 0);
    int mz = (int)_shfl_sync_139;
    if (warp == 1) {
        int mb_t[3];
        mb_t[0] = -1;
        mb_t[1] = -1;
        mb_t[2] = -1;
        if (pm0 < mz) {
            int sk_head_2 = (((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) < (T) ? ((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) : (T));
            int sk_words_2 = (T - sk_head_2) / 16;
            int sk_tail0_2 = sk_head_2 + 16 * sk_words_2;
            unsigned int sk_w_2[4];
            mb_t[0] = -1;
            mb_t[1] = -1;
            mb_t[2] = -1;
            int sk_base_2 = 0;
            int sk_ch_2 = sk_words_2 + 1;
            if (0 < sk_tail0_2) {
                sk_ch_2 = 1 + (-sk_head_2) / 16;
            }
            if (0 < sk_head_2) {
                sk_ch_2 = 0;
            }
            #pragma unroll 1
            for (int sk_r_2 = sk_ch_2; sk_r_2 < sk_words_2 + 2; sk_r_2 += 32) {
                int sk_c_2 = sk_r_2 + lane;
                int sk_c0_2 = sk_head_2 + 16 * (sk_c_2 - 1);
                int sk_nb_2 = 16;
                if (sk_c_2 == 0) {
                    sk_c0_2 = 0;
                    sk_nb_2 = sk_head_2;
                }
                if (sk_c_2 >= sk_words_2 + 1) {
                    sk_c0_2 = sk_tail0_2;
                    sk_nb_2 = T - sk_tail0_2;
                }
                if (sk_c_2 > sk_words_2 + 1) {
                    sk_nb_2 = 0;
                }
                unsigned int sk_m_2 = 0;
                if (sk_nb_2 < 16) {
                    #pragma unroll
                    for (int b_2 = 0; b_2 < 15; b_2++) {
                        if (b_2 < sk_nb_2) {
                            if (mask[sk_c0_2 + b_2]) {
                                sk_m_2 = sk_m_2 | 1 << (unsigned int)b_2;
                            }
                        }
                    }
                } else {
                    {
                        uint32_t _uv4_140_0;
                        uint32_t _uv4_140_1;
                        uint32_t _uv4_140_2;
                        uint32_t _uv4_140_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_140_0), "=r"(_uv4_140_1), "=r"(_uv4_140_2), "=r"(_uv4_140_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(mask + sk_c0_2) + (0))) : "memory");
                        sk_w_2[0 + 0] = _uv4_140_0;
                        sk_w_2[0 + 1] = _uv4_140_1;
                        sk_w_2[0 + 2] = _uv4_140_2;
                        sk_w_2[0 + 3] = _uv4_140_3;
                    }
                    #pragma unroll
                    for (int i_15 = 0; i_15 < 4; i_15++) {
                        unsigned int sk_nz_2 = ((sk_w_2[i_15] & 2139062143) + 2139062143 | sk_w_2[i_15]) & 2155905152u;
                        unsigned int sk_m4_2 = sk_nz_2 >> 7 & 1 | sk_nz_2 >> 14 & 2 | sk_nz_2 >> 21 & 4 | sk_nz_2 >> 28 & 8;
                        sk_m_2 = sk_m_2 | sk_m4_2 << (unsigned int)(4 * i_15);
                    }
                }
                int _max_142 = ((-sk_c0_2) > (0) ? (-sk_c0_2) : (0));
                int _min_141 = ((_max_142) < (sk_nb_2) ? (_max_142) : (sk_nb_2));
                unsigned int sk_lo_2 = (unsigned int)_min_141;
                unsigned int sk_span_2 = (1 << (unsigned int)sk_nb_2) - (1 << sk_lo_2);
                unsigned int sk_sel_2 = (sk_m_2 ^ 65535) & sk_span_2;
                int sk_cnt_2 = (int)(unsigned int)__popc(sk_sel_2);
                int sk_incl_2 = sk_cnt_2;
                {
                    uint32_t sk_incl_2_scan = (uint32_t)sk_incl_2;
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_2_scan, 1, 32); if (lane >= 1) { sk_incl_2_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_2_scan, 2, 32); if (lane >= 2) { sk_incl_2_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_2_scan, 4, 32); if (lane >= 4) { sk_incl_2_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_2_scan, 8, 32); if (lane >= 8) { sk_incl_2_scan += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_2_scan, 16, 32); if (lane >= 16) { sk_incl_2_scan += peer; } }
                    sk_incl_2 = (int)sk_incl_2_scan;
                }
                int sk_t_3 = pm0 + 1 - sk_base_2;
                unsigned int sk_hit_3 = __ballot_sync(0xFFFFFFFF, sk_incl_2 >= sk_t_3);
                int sk_k_3 = sk_t_3 - (sk_incl_2 - sk_cnt_2);
                unsigned int sk_s_3 = sk_sel_2;
                int sk_pos_3 = 0;
                #pragma unroll
                for (int sk_h_3 = 0; sk_h_3 < 4; sk_h_3++) {
                    unsigned int sk_half_3 = 8 >> (unsigned int)sk_h_3;
                    int sk_low_3 = (int)(unsigned int)__popc(sk_s_3 & (1 << sk_half_3) - 1);
                    if (sk_k_3 > sk_low_3) {
                        sk_k_3 = sk_k_3 - sk_low_3;
                        sk_pos_3 = sk_pos_3 + (int)sk_half_3;
                        sk_s_3 = sk_s_3 >> sk_half_3;
                    }
                }
                int _find_first_set_143 = __ffs(sk_hit_3);
                int sk_at_3 = __shfl_sync(0xFFFFFFFF, sk_c0_2 + sk_pos_3, _find_first_set_143 - 1);
                if (mb_t[0] < 0) {
                    if (sk_hit_3 != 0) {
                        mb_t[0] = sk_at_3;
                    }
                }
                int sk_t_4 = pm0 + 1 + step - sk_base_2;
                unsigned int sk_hit_4 = __ballot_sync(0xFFFFFFFF, sk_incl_2 >= sk_t_4);
                int sk_k_4 = sk_t_4 - (sk_incl_2 - sk_cnt_2);
                unsigned int sk_s_4 = sk_sel_2;
                int sk_pos_4 = 0;
                #pragma unroll
                for (int sk_h_4 = 0; sk_h_4 < 4; sk_h_4++) {
                    unsigned int sk_half_4 = 8 >> (unsigned int)sk_h_4;
                    int sk_low_4 = (int)(unsigned int)__popc(sk_s_4 & (1 << sk_half_4) - 1);
                    if (sk_k_4 > sk_low_4) {
                        sk_k_4 = sk_k_4 - sk_low_4;
                        sk_pos_4 = sk_pos_4 + (int)sk_half_4;
                        sk_s_4 = sk_s_4 >> sk_half_4;
                    }
                }
                int _find_first_set_144 = __ffs(sk_hit_4);
                int sk_at_4 = __shfl_sync(0xFFFFFFFF, sk_c0_2 + sk_pos_4, _find_first_set_144 - 1);
                if (mb_t[1] < 0) {
                    if (sk_hit_4 != 0) {
                        mb_t[1] = sk_at_4;
                    }
                }
                int sk_t_5 = pm0 + 1 + 2 * step - sk_base_2;
                unsigned int sk_hit_5 = __ballot_sync(0xFFFFFFFF, sk_incl_2 >= sk_t_5);
                int sk_k_5 = sk_t_5 - (sk_incl_2 - sk_cnt_2);
                unsigned int sk_s_5 = sk_sel_2;
                int sk_pos_5 = 0;
                #pragma unroll
                for (int sk_h_5 = 0; sk_h_5 < 4; sk_h_5++) {
                    unsigned int sk_half_5 = 8 >> (unsigned int)sk_h_5;
                    int sk_low_5 = (int)(unsigned int)__popc(sk_s_5 & (1 << sk_half_5) - 1);
                    if (sk_k_5 > sk_low_5) {
                        sk_k_5 = sk_k_5 - sk_low_5;
                        sk_pos_5 = sk_pos_5 + (int)sk_half_5;
                        sk_s_5 = sk_s_5 >> sk_half_5;
                    }
                }
                int _find_first_set_145 = __ffs(sk_hit_5);
                int sk_at_5 = __shfl_sync(0xFFFFFFFF, sk_c0_2 + sk_pos_5, _find_first_set_145 - 1);
                if (mb_t[2] < 0) {
                    if (sk_hit_5 != 0) {
                        mb_t[2] = sk_at_5;
                    }
                }
                if (mb_t[2] >= 0) {
                    break;
                }
                int _shfl_sync_146 = __shfl_sync(0xFFFFFFFF, sk_incl_2, 31);
                sk_base_2 = sk_base_2 + _shfl_sync_146;
            }
        }
        if (lane == 0) {
            nxt[7] = (unsigned int)mb_t[0];
            nxt[8] = (unsigned int)mb_t[1];
            nxt[9] = (unsigned int)mb_t[2];
        }
    }
    __syncthreads();
    int mts[3];
    #pragma unroll 1
    for (int pm = pm0; pm < mz; pm += 3 * step) {
        uint32_t pw_w_2;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(pw_w_2) : "r"(static_cast<uint32_t>(nxt_addr + 28)));
        mts[0] = (int)pw_w_2;
        uint32_t pw_w_3;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(pw_w_3) : "r"(static_cast<uint32_t>(nxt_addr + 32)));
        mts[1] = (int)pw_w_3;
        uint32_t pw_w_4;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(pw_w_4) : "r"(static_cast<uint32_t>(nxt_addr + 36)));
        mts[2] = (int)pw_w_4;
        if (mts[0] >= 0) {
            long long st_at_4 = (long long)mts[0] * 20480ll + (long long)col;
            #pragma unroll
            for (int s_34 = 0; s_34 < 4; s_34++) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * (s_34 * 160 + tid))), "l"(dout + (st_at_4 + (long long)s_34 * 5120ll)));
            }
        }
        if (mts[1] >= 0) {
            long long st_at_5 = (long long)mts[1] * 20480ll + (long long)col;
            #pragma unroll
            for (int s_35 = 0; s_35 < 4; s_35++) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * ((4 + s_35) * 160 + tid))), "l"(dout + (st_at_5 + (long long)s_35 * 5120ll)));
            }
        }
        if (mts[2] >= 0) {
            long long st_at_6 = (long long)mts[2] * 20480ll + (long long)col;
            #pragma unroll
            for (int s_36 = 0; s_36 < 4; s_36++) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(stage_addr + (unsigned int)(16 * ((8 + s_36) * 160 + tid))), "l"(dout + (st_at_6 + (long long)s_36 * 5120ll)));
            }
        }
        asm volatile("cp.async.commit_group;");
        __syncthreads();
        if (warp == 1) {
            int mb_t_1[3];
            mb_t_1[0] = -1;
            mb_t_1[1] = -1;
            mb_t_1[2] = -1;
            if (pm + 3 * step < mz) {
                int sk_head_3 = (((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) < (T) ? ((int)(16ull - ((unsigned long long)mask & 15ull) & 15ull)) : (T));
                int sk_words_3 = (T - sk_head_3) / 16;
                int sk_tail0_3 = sk_head_3 + 16 * sk_words_3;
                unsigned int sk_w_3[4];
                mb_t_1[0] = -1;
                mb_t_1[1] = -1;
                mb_t_1[2] = -1;
                int sk_base_3 = 0;
                int sk_ch_3 = sk_words_3 + 1;
                if (mts[2] + 1 < sk_tail0_3) {
                    sk_ch_3 = 1 + (mts[2] + 1 - sk_head_3) / 16;
                }
                if (mts[2] + 1 < sk_head_3) {
                    sk_ch_3 = 0;
                }
                #pragma unroll 1
                for (int sk_r_3 = sk_ch_3; sk_r_3 < sk_words_3 + 2; sk_r_3 += 32) {
                    int sk_c_3 = sk_r_3 + lane;
                    int sk_c0_3 = sk_head_3 + 16 * (sk_c_3 - 1);
                    int sk_nb_3 = 16;
                    if (sk_c_3 == 0) {
                        sk_c0_3 = 0;
                        sk_nb_3 = sk_head_3;
                    }
                    if (sk_c_3 >= sk_words_3 + 1) {
                        sk_c0_3 = sk_tail0_3;
                        sk_nb_3 = T - sk_tail0_3;
                    }
                    if (sk_c_3 > sk_words_3 + 1) {
                        sk_nb_3 = 0;
                    }
                    unsigned int sk_m_3 = 0;
                    if (sk_nb_3 < 16) {
                        #pragma unroll
                        for (int b_3 = 0; b_3 < 15; b_3++) {
                            if (b_3 < sk_nb_3) {
                                if (mask[sk_c0_3 + b_3]) {
                                    sk_m_3 = sk_m_3 | 1 << (unsigned int)b_3;
                                }
                            }
                        }
                    } else {
                        {
                            uint32_t _uv4_147_0;
                            uint32_t _uv4_147_1;
                            uint32_t _uv4_147_2;
                            uint32_t _uv4_147_3;
                            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_uv4_147_0), "=r"(_uv4_147_1), "=r"(_uv4_147_2), "=r"(_uv4_147_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(mask + sk_c0_3) + (0))) : "memory");
                            sk_w_3[0 + 0] = _uv4_147_0;
                            sk_w_3[0 + 1] = _uv4_147_1;
                            sk_w_3[0 + 2] = _uv4_147_2;
                            sk_w_3[0 + 3] = _uv4_147_3;
                        }
                        #pragma unroll
                        for (int i_16 = 0; i_16 < 4; i_16++) {
                            unsigned int sk_nz_3 = ((sk_w_3[i_16] & 2139062143) + 2139062143 | sk_w_3[i_16]) & 2155905152u;
                            unsigned int sk_m4_3 = sk_nz_3 >> 7 & 1 | sk_nz_3 >> 14 & 2 | sk_nz_3 >> 21 & 4 | sk_nz_3 >> 28 & 8;
                            sk_m_3 = sk_m_3 | sk_m4_3 << (unsigned int)(4 * i_16);
                        }
                    }
                    int _max_149 = ((mts[2] + 1 - sk_c0_3) > (0) ? (mts[2] + 1 - sk_c0_3) : (0));
                    int _min_148 = ((_max_149) < (sk_nb_3) ? (_max_149) : (sk_nb_3));
                    unsigned int sk_lo_3 = (unsigned int)_min_148;
                    unsigned int sk_span_3 = (1 << (unsigned int)sk_nb_3) - (1 << sk_lo_3);
                    unsigned int sk_sel_3 = (sk_m_3 ^ 65535) & sk_span_3;
                    int sk_cnt_3 = (int)(unsigned int)__popc(sk_sel_3);
                    int sk_incl_3 = sk_cnt_3;
                    {
                        uint32_t sk_incl_3_scan = (uint32_t)sk_incl_3;
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_3_scan, 1, 32); if (lane >= 1) { sk_incl_3_scan += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_3_scan, 2, 32); if (lane >= 2) { sk_incl_3_scan += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_3_scan, 4, 32); if (lane >= 4) { sk_incl_3_scan += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_3_scan, 8, 32); if (lane >= 8) { sk_incl_3_scan += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, sk_incl_3_scan, 16, 32); if (lane >= 16) { sk_incl_3_scan += peer; } }
                        sk_incl_3 = (int)sk_incl_3_scan;
                    }
                    int sk_t_6 = step - sk_base_3;
                    unsigned int sk_hit_6 = __ballot_sync(0xFFFFFFFF, sk_incl_3 >= sk_t_6);
                    int sk_k_6 = sk_t_6 - (sk_incl_3 - sk_cnt_3);
                    unsigned int sk_s_6 = sk_sel_3;
                    int sk_pos_6 = 0;
                    #pragma unroll
                    for (int sk_h_6 = 0; sk_h_6 < 4; sk_h_6++) {
                        unsigned int sk_half_6 = 8 >> (unsigned int)sk_h_6;
                        int sk_low_6 = (int)(unsigned int)__popc(sk_s_6 & (1 << sk_half_6) - 1);
                        if (sk_k_6 > sk_low_6) {
                            sk_k_6 = sk_k_6 - sk_low_6;
                            sk_pos_6 = sk_pos_6 + (int)sk_half_6;
                            sk_s_6 = sk_s_6 >> sk_half_6;
                        }
                    }
                    int _find_first_set_150 = __ffs(sk_hit_6);
                    int sk_at_6 = __shfl_sync(0xFFFFFFFF, sk_c0_3 + sk_pos_6, _find_first_set_150 - 1);
                    if (mb_t_1[0] < 0) {
                        if (sk_hit_6 != 0) {
                            mb_t_1[0] = sk_at_6;
                        }
                    }
                    int sk_t_7 = step + step - sk_base_3;
                    unsigned int sk_hit_7 = __ballot_sync(0xFFFFFFFF, sk_incl_3 >= sk_t_7);
                    int sk_k_7 = sk_t_7 - (sk_incl_3 - sk_cnt_3);
                    unsigned int sk_s_7 = sk_sel_3;
                    int sk_pos_7 = 0;
                    #pragma unroll
                    for (int sk_h_7 = 0; sk_h_7 < 4; sk_h_7++) {
                        unsigned int sk_half_7 = 8 >> (unsigned int)sk_h_7;
                        int sk_low_7 = (int)(unsigned int)__popc(sk_s_7 & (1 << sk_half_7) - 1);
                        if (sk_k_7 > sk_low_7) {
                            sk_k_7 = sk_k_7 - sk_low_7;
                            sk_pos_7 = sk_pos_7 + (int)sk_half_7;
                            sk_s_7 = sk_s_7 >> sk_half_7;
                        }
                    }
                    int _find_first_set_151 = __ffs(sk_hit_7);
                    int sk_at_7 = __shfl_sync(0xFFFFFFFF, sk_c0_3 + sk_pos_7, _find_first_set_151 - 1);
                    if (mb_t_1[1] < 0) {
                        if (sk_hit_7 != 0) {
                            mb_t_1[1] = sk_at_7;
                        }
                    }
                    int sk_t_8 = step + 2 * step - sk_base_3;
                    unsigned int sk_hit_8 = __ballot_sync(0xFFFFFFFF, sk_incl_3 >= sk_t_8);
                    int sk_k_8 = sk_t_8 - (sk_incl_3 - sk_cnt_3);
                    unsigned int sk_s_8 = sk_sel_3;
                    int sk_pos_8 = 0;
                    #pragma unroll
                    for (int sk_h_8 = 0; sk_h_8 < 4; sk_h_8++) {
                        unsigned int sk_half_8 = 8 >> (unsigned int)sk_h_8;
                        int sk_low_8 = (int)(unsigned int)__popc(sk_s_8 & (1 << sk_half_8) - 1);
                        if (sk_k_8 > sk_low_8) {
                            sk_k_8 = sk_k_8 - sk_low_8;
                            sk_pos_8 = sk_pos_8 + (int)sk_half_8;
                            sk_s_8 = sk_s_8 >> sk_half_8;
                        }
                    }
                    int _find_first_set_152 = __ffs(sk_hit_8);
                    int sk_at_8 = __shfl_sync(0xFFFFFFFF, sk_c0_3 + sk_pos_8, _find_first_set_152 - 1);
                    if (mb_t_1[2] < 0) {
                        if (sk_hit_8 != 0) {
                            mb_t_1[2] = sk_at_8;
                        }
                    }
                    if (mb_t_1[2] >= 0) {
                        break;
                    }
                    int _shfl_sync_153 = __shfl_sync(0xFFFFFFFF, sk_incl_3, 31);
                    sk_base_3 = sk_base_3 + _shfl_sync_153;
                }
            }
            if (lane == 0) {
                nxt[7] = (unsigned int)mb_t_1[0];
                nxt[8] = (unsigned int)mb_t_1[1];
                nxt[9] = (unsigned int)mb_t_1[2];
            }
        }
        asm volatile("cp.async.wait_group 0;");
        if (mts[0] >= 0) {
            #pragma unroll
            for (int s_37 = 0; s_37 < 4; s_37++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gw[4 * s_37])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_37) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_37) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_37) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * (s_37 * 160 + tid))));
            }
            long long ms_x = (long long)mts[0] * 20480ll + (long long)col;
            long long ms_k = (long long)mts[0] * dkey_stride_p0 + (long long)col;
            long long ms_v = (long long)mts[0] * dvalue_stride_p0 + (long long)col;
            #pragma unroll
            for (int s_38 = 0; s_38 < 4; s_38++) {
                reinterpret_cast<int4*>(dx + (ms_x + (long long)(s_38 * 5120)))[0] = reinterpret_cast<int4*>(gw + (4 * s_38))[0];
                reinterpret_cast<int4*>(dkey + (ms_k + (long long)s_38 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(zw)[0];
            }
            reinterpret_cast<int4*>(dvalue + ms_v)[0] = reinterpret_cast<int4*>(zw)[0];
        }
        if (mts[1] >= 0) {
            #pragma unroll
            for (int s_39 = 0; s_39 < 4; s_39++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gw[4 * s_39])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_39) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_39) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_39) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * ((4 + s_39) * 160 + tid))));
            }
            long long ms_x_1 = (long long)mts[1] * 20480ll + (long long)col;
            long long ms_k_1 = (long long)mts[1] * dkey_stride_p0 + (long long)col;
            long long ms_v_1 = (long long)mts[1] * dvalue_stride_p0 + (long long)col;
            #pragma unroll
            for (int s_40 = 0; s_40 < 4; s_40++) {
                reinterpret_cast<int4*>(dx + (ms_x_1 + (long long)(s_40 * 5120)))[0] = reinterpret_cast<int4*>(gw + (4 * s_40))[0];
                reinterpret_cast<int4*>(dkey + (ms_k_1 + (long long)s_40 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(zw)[0];
            }
            reinterpret_cast<int4*>(dvalue + ms_v_1)[0] = reinterpret_cast<int4*>(zw)[0];
        }
        if (mts[2] >= 0) {
            #pragma unroll
            for (int s_41 = 0; s_41 < 4; s_41++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gw[4 * s_41])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_41) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_41) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4 * s_41) + 3]))
                    : "r"(stage_addr + (unsigned int)(16 * ((8 + s_41) * 160 + tid))));
            }
            long long ms_x_2 = (long long)mts[2] * 20480ll + (long long)col;
            long long ms_k_2 = (long long)mts[2] * dkey_stride_p0 + (long long)col;
            long long ms_v_2 = (long long)mts[2] * dvalue_stride_p0 + (long long)col;
            #pragma unroll
            for (int s_42 = 0; s_42 < 4; s_42++) {
                reinterpret_cast<int4*>(dx + (ms_x_2 + (long long)(s_42 * 5120)))[0] = reinterpret_cast<int4*>(gw + (4 * s_42))[0];
                reinterpret_cast<int4*>(dkey + (ms_k_2 + (long long)s_42 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(zw)[0];
            }
            reinterpret_cast<int4*>(dvalue + ms_v_2)[0] = reinterpret_cast<int4*>(zw)[0];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int s_43 = 0; s_43 < 4; s_43++) {
        #pragma unroll
        for (int half_9 = 0; half_9 < 2; half_9++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dwa[8 * s_43 + 4 * half_9])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_43 + 4 * half_9) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_43 + 4 * half_9) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_43 + 4 * half_9) + 3]))
                : "r"(dslots_addr + (unsigned int)(16 * ((2 * s_43 + half_9) * 160 + tid))));
        }
    }
    unsigned int fs_row[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&fs_row[0])) : "r"(fs_addr + (unsigned int)(4 * tid)));
    long long row_first = (long long)fs_row[0];
    #pragma unroll
    for (int s_44 = 0; s_44 < 4; s_44++) {
        #pragma unroll
        for (int half_10 = 0; half_10 < 2; half_10++) {
            {
                float4 _v4 = make_float4(dwa[8 * s_44 + 4 * half_10 + 0], dwa[8 * s_44 + 4 * half_10 + 1], dwa[8 * s_44 + 4 * half_10 + 2], dwa[8 * s_44 + 4 * half_10 + 3]);
                *reinterpret_cast<float4*>((dw_part + (row_first * 20480ll + (long long)(s_44 * 5120 + col + 4 * half_10))) + 0) = _v4;
            }
        }
    }
}

} // extern "C"
