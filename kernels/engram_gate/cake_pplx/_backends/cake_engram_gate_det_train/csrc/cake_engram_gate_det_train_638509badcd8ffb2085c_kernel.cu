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
kernel_cake_engram_gate_det_train_638509badcd8ffb2085c(__nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, int* __restrict__ q_weight, int* __restrict__ k_weight, bool* __restrict__ mask, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dkey, __nv_bfloat16* __restrict__ dvalue, float* __restrict__ dw_part, int T, float eps, float clamp, int wmode, long long q_weight_size_p0, long long q_weight_size_p1, long long k_weight_size_p0, long long k_weight_size_p1, long long key_stride_p0, long long key_stride_p1, long long value_stride_p0, long long dkey_stride_p0, long long dkey_stride_p1, long long dvalue_stride_p0)
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
#if __CUDA_ARCH__ == 1030
    if (wmode == 0) {
        unsigned int lp_q[32];
        unsigned int lp_k[32];
        {
#else
    unsigned int lp_q[32];
    unsigned int lp_k[32];
    if ((wmode & 1) != 0) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            {
                uint32_t _uv4_0_0;
                uint32_t _uv4_0_1;
                uint32_t _uv4_0_2;
                uint32_t _uv4_0_3;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s * 2560 + (col >> 1))) + (0))) : "memory");
                lp_q[4 * s + 0] = _uv4_0_0;
                lp_q[4 * s + 1] = _uv4_0_1;
                lp_q[4 * s + 2] = _uv4_0_2;
                lp_q[4 * s + 3] = _uv4_0_3;
            }
        }
    } else {
        #pragma unroll
        for (int s_1 = 0; s_1 < 4; s_1++) {
#endif
            #pragma unroll
#if __CUDA_ARCH__ == 1030
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
#else
            for (int half = 0; half < 2; half++) {
                {
                    uint32_t _uv4_1_0;
                    uint32_t _uv4_1_1;
                    uint32_t _uv4_1_2;
                    uint32_t _uv4_1_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s_1 * 5120 + col + 4 * half)) + (0))) : "memory");
                    lp_q[8 * s_1 + 4 * half + 0] = _uv4_1_0;
                    lp_q[8 * s_1 + 4 * half + 1] = _uv4_1_1;
                    lp_q[8 * s_1 + 4 * half + 2] = _uv4_1_2;
                    lp_q[8 * s_1 + 4 * half + 3] = _uv4_1_3;
#endif
                }
            }
        }
#if __CUDA_ARCH__ == 1030
        {
#else
    }
    if ((wmode & 2) != 0) {
        #pragma unroll
        for (int s_2 = 0; s_2 < 4; s_2++) {
            {
                uint32_t _uv4_2_0;
                uint32_t _uv4_2_1;
                uint32_t _uv4_2_2;
                uint32_t _uv4_2_3;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_2 * 2560 + (col >> 1))) + (0))) : "memory");
                lp_k[4 * s_2 + 0] = _uv4_2_0;
                lp_k[4 * s_2 + 1] = _uv4_2_1;
                lp_k[4 * s_2 + 2] = _uv4_2_2;
                lp_k[4 * s_2 + 3] = _uv4_2_3;
            }
        }
    } else {
        #pragma unroll
        for (int s_3 = 0; s_3 < 4; s_3++) {
#endif
            #pragma unroll
#if __CUDA_ARCH__ == 1030
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
#else
            for (int half_1 = 0; half_1 < 2; half_1++) {
                {
                    uint32_t _uv4_3_0;
                    uint32_t _uv4_3_1;
                    uint32_t _uv4_3_2;
                    uint32_t _uv4_3_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_3_0), "=r"(_uv4_3_1), "=r"(_uv4_3_2), "=r"(_uv4_3_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_3 * 5120 + col + 4 * half_1)) + (0))) : "memory");
                    lp_k[8 * s_3 + 4 * half_1 + 0] = _uv4_3_0;
                    lp_k[8 * s_3 + 4 * half_1 + 1] = _uv4_3_1;
                    lp_k[8 * s_3 + 4 * half_1 + 2] = _uv4_3_2;
                    lp_k[8 * s_3 + 4 * half_1 + 3] = _uv4_3_3;
#endif
                }
            }
        }
#if __CUDA_ARCH__ == 1030
        float lp_qf[32];
        {
#else
    }
    float lp_qf[32];
    if ((wmode & 1) != 0) {
        #pragma unroll
        for (int s_4 = 0; s_4 < 4; s_4++) {
#endif
            #pragma unroll
#if __CUDA_ARCH__ == 1030
            for (int i = 0; i < 32; i++) {
                lp_qf[i] = __uint_as_float(lp_q[i]);
#else
            for (int m = 0; m < 4; m++) {
                float2 ww_v = cake_bf16x2_to_f32x2(__as_bf16x2(lp_q[4 * s_4 + m]));
                lp_qf[8 * s_4 + 2 * m] = ww_v.x;
                lp_qf[8 * s_4 + 2 * m + 1] = ww_v.y;
#endif
            }
        }
#if __CUDA_ARCH__ == 1030
        {
#else
    } else {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            lp_qf[i] = __uint_as_float(lp_q[i]);
        }
    }
    if ((wmode & 2) != 0) {
        #pragma unroll
        for (int s_5 = 0; s_5 < 4; s_5++) {
#endif
            #pragma unroll
#if __CUDA_ARCH__ == 1030
            for (int i_1 = 0; i_1 < 32; i_1++) {
                wf[i_1] = __uint_as_float(lp_k[i_1]);
#else
            for (int m_1 = 0; m_1 < 4; m_1++) {
                float2 ww_v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(lp_k[4 * s_5 + m_1]));
                wf[8 * s_5 + 2 * m_1] = ww_v_1.x;
                wf[8 * s_5 + 2 * m_1 + 1] = ww_v_1.y;
#endif
            }
        }
#if __CUDA_ARCH__ == 1030
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
#else
#endif
    } else {
#if __CUDA_ARCH__ == 1030
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
#else
        #pragma unroll
        for (int i_1 = 0; i_1 < 32; i_1++) {
            wf[i_1] = __uint_as_float(lp_k[i_1]);
#endif
        }
    }
    #pragma unroll
#if __CUDA_ARCH__ == 1030
    for (int s_12 = 0; s_12 < 4; s_12++) {
#else
    for (int i_2 = 0; i_2 < 32; i_2++) {
        wf[i_2] = mul_f32_rn_noftz(lp_qf[i_2], wf[i_2]);
    }
    #pragma unroll
    for (int s_6 = 0; s_6 < 4; s_6++) {
#endif
        #pragma unroll
#if __CUDA_ARCH__ == 1030
        for (int half_4 = 0; half_4 < 2; half_4++) {
            unsigned int bw_at = (unsigned int)(16 * ((2 * s_12 + half_4) * 160 + tid));
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(wslots_addr + bw_at), "f"(wf[8 * s_12 + 4 * half_4]), "f"(wf[8 * s_12 + 4 * half_4 + 1]), "f"(wf[8 * s_12 + 4 * half_4 + 2]), "f"(wf[8 * s_12 + 4 * half_4 + 3]) : "memory");
#else
        for (int half_2 = 0; half_2 < 2; half_2++) {
            unsigned int bw_at = (unsigned int)(16 * ((2 * s_6 + half_2) * 160 + tid));
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(wslots_addr + bw_at), "f"(wf[8 * s_6 + 4 * half_2]), "f"(wf[8 * s_6 + 4 * half_2 + 1]), "f"(wf[8 * s_6 + 4 * half_2 + 2]), "f"(wf[8 * s_6 + 4 * half_2 + 3]) : "memory");
#endif
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dslots_addr + bw_at), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f) : "memory");
        }
    }
    float plain[16];
    #pragma unroll
#if __CUDA_ARCH__ == 1030
    for (int i_8 = 12; i_8 < 16; i_8++) {
        plain[i_8] = 0.0f;
#else
    for (int i_3 = 12; i_3 < 16; i_3++) {
        plain[i_3] = 0.0f;
#endif
    }
    float ps[4];
    float pe[4];
    float tot[4];
    unsigned int zw[4];
    #pragma unroll
#if __CUDA_ARCH__ == 1030
    for (int i_9 = 0; i_9 < 4; i_9++) {
        zw[i_9] = 0;
#else
    for (int i_4 = 0; i_4 < 4; i_4++) {
        zw[i_4] = 0;
#endif
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
#if __CUDA_ARCH__ == 1030
    for (int s_13 = 0; s_13 < 4; s_13++) {
#else
    for (int s_7 = 0; s_7 < 4; s_7++) {
#endif
        {
#if __CUDA_ARCH__ == 1030
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(dout + (ls_at + (long long)(s_13 * 5120)) + 0);
            uint4* _vdst_8 = reinterpret_cast<uint4*>(&gw[4 * s_13]);
#else
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(dout + (ls_at + (long long)(s_7 * 5120)) + 0);
            uint4* _vdst_4 = reinterpret_cast<uint4*>(&gw[4 * s_7]);
#endif
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                _vdst_8[_blk] = _vptr_8[_blk];
#else
                _vdst_4[_blk] = _vptr_4[_blk];
#endif
            }
        }
    }
    long long ls_at_0 = (long long)ftok * 20480ll + (long long)col;
    #pragma unroll
#if __CUDA_ARCH__ == 1030
    for (int s_14 = 0; s_14 < 4; s_14++) {
#else
    for (int s_8 = 0; s_8 < 4; s_8++) {
#endif
        {
#if __CUDA_ARCH__ == 1030
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + (ls_at_0 + (long long)(s_14 * 5120)) + 0);
            uint4* _vdst_9 = reinterpret_cast<uint4*>(&xw[4 * s_14]);
#else
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + (ls_at_0 + (long long)(s_8 * 5120)) + 0);
            uint4* _vdst_5 = reinterpret_cast<uint4*>(&xw[4 * s_8]);
#endif
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                _vdst_9[_blk] = _vptr_9[_blk];
#else
                _vdst_5[_blk] = _vptr_5[_blk];
#endif
            }
        }
    }
    long long lk_at = (long long)ftok * key_stride_p0 + (long long)col;
    #pragma unroll
#if __CUDA_ARCH__ == 1030
    for (int s_15 = 0; s_15 < 4; s_15++) {
#else
    for (int s_9 = 0; s_9 < 4; s_9++) {
#endif
        {
#if __CUDA_ARCH__ == 1030
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(key + (lk_at + (long long)s_15 * key_stride_p1) + 0);
            uint4* _vdst_10 = reinterpret_cast<uint4*>(&kw[4 * s_15]);
#else
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(key + (lk_at + (long long)s_9 * key_stride_p1) + 0);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&kw[4 * s_9]);
#endif
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                _vdst_10[_blk] = _vptr_10[_blk];
#else
                _vdst_6[_blk] = _vptr_6[_blk];
#endif
            }
        }
    }
    {
#if __CUDA_ARCH__ == 1030
        const uint4* _vptr_11 = reinterpret_cast<const uint4*>(value + ((long long)ftok * value_stride_p0 + (long long)col) + 0);
        uint4* _vdst_11 = reinterpret_cast<uint4*>(&vw[0]);
#else
        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(value + ((long long)ftok * value_stride_p0 + (long long)col) + 0);
        uint4* _vdst_7 = reinterpret_cast<uint4*>(&vw[0]);
#endif
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
            _vdst_11[_blk] = _vptr_11[_blk];
#else
            _vdst_7[_blk] = _vptr_7[_blk];
#endif
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
#if __CUDA_ARCH__ == 1030
            for (int s_16 = 0; s_16 < 4; s_16++) {
#else
            for (int s_10 = 0; s_10 < 4; s_10++) {
#endif
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int half_5 = 0; half_5 < 2; half_5++) {
#else
                for (int half_3 = 0; half_3 < 2; half_3++) {
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if __CUDA_ARCH__ == 1030
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_16 + 4 * half_5])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_16 + 4 * half_5) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_16 + 4 * half_5) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_16 + 4 * half_5) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_16 + half_5) * 160 + tid))));
#else
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_10 + 4 * half_3])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_10 + 4 * half_3) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_10 + 4 * half_3) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_10 + 4 * half_3) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_10 + half_3) * 160 + tid))));
#endif
                }
            }
            #pragma unroll
#if __CUDA_ARCH__ == 1030
            for (int s_17 = 0; s_17 < 4; s_17++) {
#else
            for (int s_11 = 0; s_11 < 4; s_11++) {
#endif
                float ss_h = 0.0f;
                float ss_k = 0.0f;
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int m_4 = 0; m_4 < 4; m_4++) {
                    float _dot_12;
#else
                for (int m_2 = 0; m_2 < 4; m_2++) {
                    float _dot_8;
#endif
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
#if __CUDA_ARCH__ == 1030
                        : "=f"(_dot_12) : "r"(xw[4 * s_17 + m_4]), "r"(xw[4 * s_17 + m_4]), "f"(ss_h));
                    ss_h = _dot_12;
                    float _dot_13;
#else
                        : "=f"(_dot_8) : "r"(xw[4 * s_11 + m_2]), "r"(xw[4 * s_11 + m_2]), "f"(ss_h));
                    ss_h = _dot_8;
                    float _dot_9;
#endif
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
#if __CUDA_ARCH__ == 1030
                        : "=f"(_dot_13) : "r"(kw[4 * s_17 + m_4]), "r"(kw[4 * s_17 + m_4]), "f"(ss_k));
                    ss_k = _dot_13;
#else
                        : "=f"(_dot_9) : "r"(kw[4 * s_11 + m_2]), "r"(kw[4 * s_11 + m_2]), "f"(ss_k));
                    ss_k = _dot_9;
#endif
                }
#if __CUDA_ARCH__ == 1030
                plain[s_17] = ss_h;
                plain[4 + s_17] = ss_k;
#else
                plain[s_11] = ss_h;
                plain[4 + s_11] = ss_k;
#endif
            }
            #pragma unroll
#if __CUDA_ARCH__ == 1030
            for (int s_18 = 0; s_18 < 4; s_18++) {
#else
            for (int s_12 = 0; s_12 < 4; s_12++) {
#endif
                float sp_s = 0.0f;
                float sp_e = 0.0f;
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int m_5 = 0; m_5 < 4; m_5++) {
#else
                for (int m_3 = 0; m_3 < 4; m_3++) {
#endif
                    float sp_hk[2] = {0.0f, 0.0f};
#if __CUDA_ARCH__ == 1030
                    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk[0]) : "h"(static_cast<uint16_t>((xw[4 * s_18 + m_5]))), "h"(static_cast<uint16_t>((kw[4 * s_18 + m_5]))));
                    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk[1]) : "h"(static_cast<uint16_t>((xw[4 * s_18 + m_5]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_18 + m_5]) >> 16)));
                    float sp_w_0 = wf[8 * s_18 + 2 * m_5];
#else
                    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk[0]) : "h"(static_cast<uint16_t>((xw[4 * s_12 + m_3]))), "h"(static_cast<uint16_t>((kw[4 * s_12 + m_3]))));
                    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk[1]) : "h"(static_cast<uint16_t>((xw[4 * s_12 + m_3]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_12 + m_3]) >> 16)));
                    float sp_w_0 = wf[8 * s_12 + 2 * m_3];
#endif
                    float sp_p_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p_0) : "f"(sp_w_0), "f"(sp_hk[0]));
#if __CUDA_ARCH__ == 1030
                    float _neg_14;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(sp_p_0));
#else
                    float _neg_10;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_10) : "f"(sp_p_0));
#endif
                    float sp_q_0;
#if __CUDA_ARCH__ == 1030
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q_0) : "f"(sp_w_0), "f"(sp_hk[0]), "f"(_neg_14));
#else
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q_0) : "f"(sp_w_0), "f"(sp_hk[0]), "f"(_neg_10));
#endif
                    float ts_s;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(sp_s), "f"(sp_p_0));
                    float ts_bb;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                    float _sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(ts_s), "f"(ts_bb));
                    float _sub_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(sp_s), "f"(_sub_16));
                    float _sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(sp_p_0), "f"(ts_bb));
#else
                    float _sub_12;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(ts_s), "f"(ts_bb));
                    float _sub_11;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(sp_s), "f"(_sub_12));
                    float _sub_13;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(sp_p_0), "f"(ts_bb));
#endif
                    float ts_e;
#if __CUDA_ARCH__ == 1030
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_15), "f"(_sub_17));
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_11), "f"(_sub_13));
#endif
                    sp_s = ts_s;
#if __CUDA_ARCH__ == 1030
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(sp_q_0), "f"(ts_e));
                    float _add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(sp_e), "f"(_add_19));
                    sp_e = _add_18;
                    float sp_w_1 = wf[8 * s_18 + 2 * m_5 + 1];
#else
                    float _add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(sp_q_0), "f"(ts_e));
                    float _add_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(sp_e), "f"(_add_15));
                    sp_e = _add_14;
                    float sp_w_1 = wf[8 * s_12 + 2 * m_3 + 1];
#endif
                    float sp_p_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p_1) : "f"(sp_w_1), "f"(sp_hk[1]));
#if __CUDA_ARCH__ == 1030
                    float _neg_20;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(sp_p_1));
#else
                    float _neg_16;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(sp_p_1));
#endif
                    float sp_q_1;
#if __CUDA_ARCH__ == 1030
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q_1) : "f"(sp_w_1), "f"(sp_hk[1]), "f"(_neg_20));
#else
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q_1) : "f"(sp_w_1), "f"(sp_hk[1]), "f"(_neg_16));
#endif
                    float ts_s_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(sp_s), "f"(sp_p_1));
                    float ts_bb_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_0), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                    float _sub_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(ts_s_0), "f"(ts_bb_0));
                    float _sub_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(sp_s), "f"(_sub_22));
                    float _sub_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(sp_p_1), "f"(ts_bb_0));
#else
                    float _sub_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(ts_s_0), "f"(ts_bb_0));
                    float _sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(sp_s), "f"(_sub_18));
                    float _sub_19;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(sp_p_1), "f"(ts_bb_0));
#endif
                    float ts_e_0;
#if __CUDA_ARCH__ == 1030
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_21), "f"(_sub_23));
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_17), "f"(_sub_19));
#endif
                    sp_s = ts_s_0;
#if __CUDA_ARCH__ == 1030
                    float _add_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(sp_q_1), "f"(ts_e_0));
                    float _add_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(sp_e), "f"(_add_25));
                    sp_e = _add_24;
#else
                    float _add_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(sp_q_1), "f"(ts_e_0));
                    float _add_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(sp_e), "f"(_add_21));
                    sp_e = _add_20;
#endif
                }
#if __CUDA_ARCH__ == 1030
                ps[s_18] = sp_s;
                pe[s_18] = sp_e;
#else
                ps[s_12] = sp_s;
                pe[s_12] = sp_e;
#endif
            }
            #pragma unroll
#if __CUDA_ARCH__ == 1030
            for (int s_19 = 0; s_19 < 4; s_19++) {
#else
            for (int s_13 = 0; s_13 < 4; s_13++) {
#endif
                float tl_q = 0.0f;
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int m_6 = 0; m_6 < 4; m_6++) {
                    float _dot_26;
#else
                for (int m_4 = 0; m_4 < 4; m_4++) {
                    float _dot_22;
#endif
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
#if __CUDA_ARCH__ == 1030
                        : "=f"(_dot_26) : "r"(gw[4 * s_19 + m_6]), "r"(vw[m_6]), "f"(tl_q));
                    tl_q = _dot_26;
#else
                        : "=f"(_dot_22) : "r"(gw[4 * s_13 + m_4]), "r"(vw[m_4]), "f"(tl_q));
                    tl_q = _dot_22;
#endif
                }
#if __CUDA_ARCH__ == 1030
                plain[8 + s_19] = tl_q;
#else
                plain[8 + s_13] = tl_q;
#endif
            }
            int pn_tok = t + step;
            if (pn_tok < T) {
                long long ls_at_1 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int s_20 = 0; s_20 < 4; s_20++) {
#else
                for (int s_14 = 0; s_14 < 4; s_14++) {
#endif
                    {
#if __CUDA_ARCH__ == 1030
                        const uint4* _vptr_27 = reinterpret_cast<const uint4*>(dout + (ls_at_1 + (long long)(s_20 * 5120)) + 0);
                        uint4* _vdst_27 = reinterpret_cast<uint4*>(&gn[4 * s_20]);
#else
                        const uint4* _vptr_23 = reinterpret_cast<const uint4*>(dout + (ls_at_1 + (long long)(s_14 * 5120)) + 0);
                        uint4* _vdst_23 = reinterpret_cast<uint4*>(&gn[4 * s_14]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_27[_blk] = _vptr_27[_blk];
#else
                            _vdst_23[_blk] = _vptr_23[_blk];
#endif
                        }
                    }
                }
                long long ls_at_2 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int s_21 = 0; s_21 < 4; s_21++) {
#else
                for (int s_15 = 0; s_15 < 4; s_15++) {
#endif
                    {
#if __CUDA_ARCH__ == 1030
                        const uint4* _vptr_28 = reinterpret_cast<const uint4*>(x + (ls_at_2 + (long long)(s_21 * 5120)) + 0);
                        uint4* _vdst_28 = reinterpret_cast<uint4*>(&xn[4 * s_21]);
#else
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(x + (ls_at_2 + (long long)(s_15 * 5120)) + 0);
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&xn[4 * s_15]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_28[_blk] = _vptr_28[_blk];
#else
                            _vdst_24[_blk] = _vptr_24[_blk];
#endif
                        }
                    }
                }
                long long lk_at_0 = (long long)pn_tok * key_stride_p0 + (long long)col;
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int s_22 = 0; s_22 < 4; s_22++) {
#else
                for (int s_16 = 0; s_16 < 4; s_16++) {
#endif
                    {
#if __CUDA_ARCH__ == 1030
                        const uint4* _vptr_29 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_22 * key_stride_p1) + 0);
                        uint4* _vdst_29 = reinterpret_cast<uint4*>(&kn[4 * s_22]);
#else
                        const uint4* _vptr_25 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_16 * key_stride_p1) + 0);
                        uint4* _vdst_25 = reinterpret_cast<uint4*>(&kn[4 * s_16]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_29[_blk] = _vptr_29[_blk];
#else
                            _vdst_25[_blk] = _vptr_25[_blk];
#endif
                        }
                    }
                }
                {
#if __CUDA_ARCH__ == 1030
                    const uint4* _vptr_30 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_30 = reinterpret_cast<uint4*>(&vn[0]);
#else
                    const uint4* _vptr_26 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_26 = reinterpret_cast<uint4*>(&vn[0]);
#endif
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                        _vdst_30[_blk] = _vptr_30[_blk];
#else
                        _vdst_26[_blk] = _vptr_26[_blk];
#endif
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
            int xp_slot = xk % 2;
            int xp_row = xp_slot * 100 + warp * 20;
            #pragma unroll
            for (int i_0 = 0; i_0 < 8; i_0++) {
                int ws_upper_0 = lane & 16;
                float ws_keep_0 = ((ws_upper_0 != 0) ? plain[8 + i_0] : plain[i_0]);
                float ws_send_0 = ((ws_upper_0 != 0) ? plain[i_0] : plain[8 + i_0]);
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_31;
#else
                float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_27;
#endif
            }
            #pragma unroll
            for (int i_1_1 = 0; i_1_1 < 4; i_1_1++) {
                int ws_upper_1 = lane & 8;
                float ws_keep_1 = ((ws_upper_1 != 0) ? plain[4 + i_1_1] : plain[i_1_1]);
                float ws_send_1 = ((ws_upper_1 != 0) ? plain[i_1_1] : plain[4 + i_1_1]);
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1_1] = ws_keep_1 + _shfl_xor_sync_32;
#else
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1_1] = ws_keep_1 + _shfl_xor_sync_28;
#endif
            }
            #pragma unroll
            for (int i_2_1 = 0; i_2_1 < 2; i_2_1++) {
                int ws_upper_2 = lane & 4;
                float ws_keep_2 = ((ws_upper_2 != 0) ? plain[2 + i_2_1] : plain[i_2_1]);
                float ws_send_2 = ((ws_upper_2 != 0) ? plain[i_2_1] : plain[2 + i_2_1]);
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2_1] = ws_keep_2 + _shfl_xor_sync_33;
#else
                float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2_1] = ws_keep_2 + _shfl_xor_sync_29;
#endif
            }
            #pragma unroll
            for (int i_3_1 = 0; i_3_1 < 1; i_3_1++) {
                int ws_upper_3 = lane & 2;
                float ws_keep_3 = ((ws_upper_3 != 0) ? plain[1 + i_3_1] : plain[i_3_1]);
                float ws_send_3 = ((ws_upper_3 != 0) ? plain[i_3_1] : plain[1 + i_3_1]);
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
                plain[i_3_1] = ws_keep_3 + _shfl_xor_sync_34;
#else
                float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
                plain[i_3_1] = ws_keep_3 + _shfl_xor_sync_30;
#endif
            }
            float ws_own = plain[0];
#if __CUDA_ARCH__ == 1030
            float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_35;
#else
            float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_31;
#endif
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
                float ts_s_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(wp_ks_0), "f"(wp_rs_0));
                float ts_bb_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_1), "f"(wp_ks_0));
#if __CUDA_ARCH__ == 1030
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(ts_s_1), "f"(ts_bb_1));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(wp_ks_0), "f"(_sub_37));
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(wp_rs_0), "f"(ts_bb_1));
#else
                float _sub_33;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(ts_s_1), "f"(ts_bb_1));
                float _sub_32;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(wp_ks_0), "f"(_sub_33));
                float _sub_34;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(wp_rs_0), "f"(ts_bb_1));
#endif
                float ts_e_1;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_36), "f"(_sub_38));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_32), "f"(_sub_34));
#endif
                wp_ks_0 = ts_s_1;
#if __CUDA_ARCH__ == 1030
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(wp_ke_0), "f"(wp_re_0));
                float _add_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(_add_40), "f"(ts_e_1));
                wp_ke_0 = _add_39;
#else
                float _add_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(wp_ke_0), "f"(wp_re_0));
                float _add_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(_add_36), "f"(ts_e_1));
                wp_ke_0 = _add_35;
#endif
                ps[i_4_1] = wp_ks_0;
                pe[i_4_1] = wp_ke_0;
            }
            #pragma unroll
#if __CUDA_ARCH__ == 1030
            for (int i_5_1 = 0; i_5_1 < 1; i_5_1++) {
#else
            for (int i_5 = 0; i_5 < 1; i_5++) {
#endif
                int wp_upper_1 = lane & 8;
#if __CUDA_ARCH__ == 1030
                float wp_ks_1 = ((wp_upper_1 != 0) ? ps[1 + i_5_1] : ps[i_5_1]);
                float wp_ke_1 = ((wp_upper_1 != 0) ? pe[1 + i_5_1] : pe[i_5_1]);
                float wp_rs_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? ps[i_5_1] : ps[1 + i_5_1]), 8);
                float wp_re_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? pe[i_5_1] : pe[1 + i_5_1]), 8);
#else
                float wp_ks_1 = ((wp_upper_1 != 0) ? ps[1 + i_5] : ps[i_5]);
                float wp_ke_1 = ((wp_upper_1 != 0) ? pe[1 + i_5] : pe[i_5]);
                float wp_rs_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? ps[i_5] : ps[1 + i_5]), 8);
                float wp_re_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? pe[i_5] : pe[1 + i_5]), 8);
#endif
                float ts_s_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(wp_ks_1), "f"(wp_rs_1));
                float ts_bb_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_2), "f"(wp_ks_1));
#if __CUDA_ARCH__ == 1030
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(ts_s_2), "f"(ts_bb_2));
                float _sub_41;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(wp_ks_1), "f"(_sub_42));
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(wp_rs_1), "f"(ts_bb_2));
#else
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(ts_s_2), "f"(ts_bb_2));
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(wp_ks_1), "f"(_sub_38));
                float _sub_39;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(wp_rs_1), "f"(ts_bb_2));
#endif
                float ts_e_2;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_41), "f"(_sub_43));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_37), "f"(_sub_39));
#endif
                wp_ks_1 = ts_s_2;
#if __CUDA_ARCH__ == 1030
                float _add_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(wp_ke_1), "f"(wp_re_1));
                float _add_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_add_45), "f"(ts_e_2));
                wp_ke_1 = _add_44;
                ps[i_5_1] = wp_ks_1;
                pe[i_5_1] = wp_ke_1;
#else
                float _add_41;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(wp_ke_1), "f"(wp_re_1));
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_add_41), "f"(ts_e_2));
                wp_ke_1 = _add_40;
                ps[i_5] = wp_ks_1;
                pe[i_5] = wp_ke_1;
#endif
            }
            float wp_s = ps[0];
            float wp_e = pe[0];
            float wp_bs_0 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 4);
            float wp_be_0 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 4);
            float ts_s_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3) : "f"(wp_s), "f"(wp_bs_0));
            float ts_bb_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3) : "f"(ts_s_3), "f"(wp_s));
#if __CUDA_ARCH__ == 1030
            float _sub_47;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(ts_s_3), "f"(ts_bb_3));
            float _sub_46;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(wp_s), "f"(_sub_47));
            float _sub_48;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(wp_bs_0), "f"(ts_bb_3));
#else
            float _sub_43;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(ts_s_3), "f"(ts_bb_3));
            float _sub_42;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(wp_s), "f"(_sub_43));
            float _sub_44;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(wp_bs_0), "f"(ts_bb_3));
#endif
            float ts_e_3;
#if __CUDA_ARCH__ == 1030
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_46), "f"(_sub_48));
#else
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_42), "f"(_sub_44));
#endif
            wp_s = ts_s_3;
#if __CUDA_ARCH__ == 1030
            float _add_50;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(wp_e), "f"(wp_be_0));
            float _add_49;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_add_50), "f"(ts_e_3));
            wp_e = _add_49;
#else
            float _add_46;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(wp_e), "f"(wp_be_0));
            float _add_45;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_add_46), "f"(ts_e_3));
            wp_e = _add_45;
#endif
            float wp_bs_1 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 2);
            float wp_be_1 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 2);
            float ts_s_1_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1_1) : "f"(wp_s), "f"(wp_bs_1));
            float ts_bb_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1_1) : "f"(ts_s_1_1), "f"(wp_s));
#if __CUDA_ARCH__ == 1030
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(ts_s_1_1), "f"(ts_bb_1_1));
            float _sub_51;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(wp_s), "f"(_sub_52));
            float _sub_53;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(wp_bs_1), "f"(ts_bb_1_1));
#else
            float _sub_48;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(ts_s_1_1), "f"(ts_bb_1_1));
            float _sub_47;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(wp_s), "f"(_sub_48));
            float _sub_49;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(wp_bs_1), "f"(ts_bb_1_1));
#endif
            float ts_e_1_1;
#if __CUDA_ARCH__ == 1030
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1_1) : "f"(_sub_51), "f"(_sub_53));
#else
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1_1) : "f"(_sub_47), "f"(_sub_49));
#endif
            wp_s = ts_s_1_1;
#if __CUDA_ARCH__ == 1030
            float _add_55;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(wp_e), "f"(wp_be_1));
            float _add_54;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(_add_55), "f"(ts_e_1_1));
            wp_e = _add_54;
#else
            float _add_51;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(wp_e), "f"(wp_be_1));
            float _add_50;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(_add_51), "f"(ts_e_1_1));
            wp_e = _add_50;
#endif
            float wp_bs_2 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 1);
            float wp_be_2 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 1);
            float ts_s_2_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2_1) : "f"(wp_s), "f"(wp_bs_2));
            float ts_bb_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2_1) : "f"(ts_s_2_1), "f"(wp_s));
#if __CUDA_ARCH__ == 1030
            float _sub_57;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(ts_s_2_1), "f"(ts_bb_2_1));
            float _sub_56;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(wp_s), "f"(_sub_57));
            float _sub_58;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(wp_bs_2), "f"(ts_bb_2_1));
#else
            float _sub_53;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(ts_s_2_1), "f"(ts_bb_2_1));
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(wp_s), "f"(_sub_53));
            float _sub_54;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(wp_bs_2), "f"(ts_bb_2_1));
#endif
            float ts_e_2_1;
#if __CUDA_ARCH__ == 1030
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2_1) : "f"(_sub_56), "f"(_sub_58));
#else
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2_1) : "f"(_sub_52), "f"(_sub_54));
#endif
            wp_s = ts_s_2_1;
#if __CUDA_ARCH__ == 1030
            float _add_60;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(wp_e), "f"(wp_be_2));
            float _add_59;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(_add_60), "f"(ts_e_2_1));
            wp_e = _add_59;
#else
            float _add_56;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(wp_e), "f"(wp_be_2));
            float _add_55;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_add_56), "f"(ts_e_2_1));
            wp_e = _add_55;
#endif
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
                                float ts_s_3_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3_1) : "f"(xp_fs), "f"(wtab[xp_base + wi_1 * 20 + c_2]));
                                float ts_bb_3_1;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3_1) : "f"(ts_s_3_1), "f"(xp_fs));
#if __CUDA_ARCH__ == 1030
                                float _sub_62;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(ts_s_3_1), "f"(ts_bb_3_1));
                                float _sub_61;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(xp_fs), "f"(_sub_62));
                                float _sub_63;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(wtab[xp_base + wi_1 * 20 + c_2]), "f"(ts_bb_3_1));
#else
                                float _sub_58;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(ts_s_3_1), "f"(ts_bb_3_1));
                                float _sub_57;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(xp_fs), "f"(_sub_58));
                                float _sub_59;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(wtab[xp_base + wi_1 * 20 + c_2]), "f"(ts_bb_3_1));
#endif
                                float ts_e_3_1;
#if __CUDA_ARCH__ == 1030
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3_1) : "f"(_sub_61), "f"(_sub_63));
#else
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3_1) : "f"(_sub_57), "f"(_sub_59));
#endif
                                xp_fs = ts_s_3_1;
#if __CUDA_ARCH__ == 1030
                                float _add_65;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(xp_fe), "f"(wtab[xp_base + wi_1 * 20 + 4 + c_2]));
                                float _add_64;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_add_65), "f"(ts_e_3_1));
                                xp_fe = _add_64;
#else
                                float _add_61;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(xp_fe), "f"(wtab[xp_base + wi_1 * 20 + 4 + c_2]));
                                float _add_60;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_add_61), "f"(ts_e_3_1));
                                xp_fe = _add_60;
#endif
                            }
                            xp_s[c_2] = xp_fs;
                            xp_e[c_2] = xp_fe;
                        }
#if __CUDA_ARCH__ == 1030
                        uint32_t _cluster_mapa_66;
#else
                        uint32_t _cluster_mapa_62;
#endif
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                            : "=r"(_cluster_mapa_66) : "r"(xp_to + (unsigned int)(16 * (xp_job + 1))), "r"(xp_dst));
                        uint32_t _mbar_peer_67;
#else
                            : "=r"(_cluster_mapa_62) : "r"(xp_to + (unsigned int)(16 * (xp_job + 1))), "r"(xp_dst));
                        uint32_t _mbar_peer_63;
#endif
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                            : "=r"(_mbar_peer_67) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
#else
                            : "=r"(_mbar_peer_63) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
#endif
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
#if __CUDA_ARCH__ == 1030
                            :: "r"(_cluster_mapa_66), "r"(__float_as_uint(xp_e[0])), "r"(__float_as_uint(xp_e[1])), "r"(__float_as_uint(xp_e[2])), "r"(__float_as_uint(xp_e[3])), "r"(_mbar_peer_67) : "memory");
#else
                            :: "r"(_cluster_mapa_62), "r"(__float_as_uint(xp_e[0])), "r"(__float_as_uint(xp_e[1])), "r"(__float_as_uint(xp_e[2])), "r"(__float_as_uint(xp_e[3])), "r"(_mbar_peer_63) : "memory");
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    uint32_t _cluster_mapa_68;
#else
                    uint32_t _cluster_mapa_64;
#endif
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                        : "=r"(_cluster_mapa_68) : "r"(xp_to + (unsigned int)(16 * xp_job)), "r"(xp_dst));
                    uint32_t _mbar_peer_69;
#else
                        : "=r"(_cluster_mapa_64) : "r"(xp_to + (unsigned int)(16 * xp_job)), "r"(xp_dst));
                    uint32_t _mbar_peer_65;
#endif
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                        : "=r"(_mbar_peer_69) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
#else
                        : "=r"(_mbar_peer_65) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
#endif
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
#if __CUDA_ARCH__ == 1030
                        :: "r"(_cluster_mapa_68), "r"(__float_as_uint(xp_s[0])), "r"(__float_as_uint(xp_s[1])), "r"(__float_as_uint(xp_s[2])), "r"(__float_as_uint(xp_s[3])), "r"(_mbar_peer_69) : "memory");
#else
                        :: "r"(_cluster_mapa_64), "r"(__float_as_uint(xp_s[0])), "r"(__float_as_uint(xp_s[1])), "r"(__float_as_uint(xp_s[2])), "r"(__float_as_uint(xp_s[3])), "r"(_mbar_peer_65) : "memory");
#endif
                }
            }
            if (tid == 0) {
                mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 320);
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
                    float ts_s_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_4) : "f"(xt_ts), "f"(recv[xt_from + r_1 * 20 + 12]));
                    float ts_bb_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_4) : "f"(ts_s_4), "f"(xt_ts));
#if __CUDA_ARCH__ == 1030
                    float _sub_71;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(ts_s_4), "f"(ts_bb_4));
                    float _sub_70;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(xt_ts), "f"(_sub_71));
                    float _sub_72;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(recv[xt_from + r_1 * 20 + 12]), "f"(ts_bb_4));
#else
                    float _sub_67;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(ts_s_4), "f"(ts_bb_4));
                    float _sub_66;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(xt_ts), "f"(_sub_67));
                    float _sub_68;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(recv[xt_from + r_1 * 20 + 12]), "f"(ts_bb_4));
#endif
                    float ts_e_4;
#if __CUDA_ARCH__ == 1030
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_70), "f"(_sub_72));
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_66), "f"(_sub_68));
#endif
                    xt_ts = ts_s_4;
#if __CUDA_ARCH__ == 1030
                    float _add_74;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(xt_te), "f"(recv[xt_from + r_1 * 20 + 16]));
                    float _add_73;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_74), "f"(ts_e_4));
                    xt_te = _add_73;
#else
                    float _add_70;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(xt_te), "f"(recv[xt_from + r_1 * 20 + 16]));
                    float _add_69;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_add_70), "f"(ts_e_4));
                    xt_te = _add_69;
#endif
                }
#if __CUDA_ARCH__ == 1030
                float _add_75;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(xt_ts), "f"(xt_te));
                tot[3] = _add_75;
                float _fdiv_77 = __fdiv_rn(tot[0], 5120.0f);
                float _sqrt_76;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_76) : "f"(_fdiv_77 + eps));
                float sg_rh = __frcp_rn(_sqrt_76);
                float _fdiv_79 = __fdiv_rn(tot[1], 5120.0f);
                float _sqrt_78;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_78) : "f"(_fdiv_79 + eps));
                float sg_rk = __frcp_rn(_sqrt_78);
#else
                float _add_71;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(xt_ts), "f"(xt_te));
                tot[3] = _add_71;
                float _fdiv_73 = __fdiv_rn(tot[0], 5120.0f);
                float _sqrt_72;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_72) : "f"(_fdiv_73 + eps));
                float sg_rh = __frcp_rn(_sqrt_72);
                float _fdiv_75 = __fdiv_rn(tot[1], 5120.0f);
                float _sqrt_74;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_74) : "f"(_fdiv_75 + eps));
                float sg_rk = __frcp_rn(_sqrt_74);
#endif
                float sg_dot;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sg_dot) : "f"(mul_f32_rn_noftz(tot[3], mul_f32_rn_noftz(sg_rh, sg_rk))), "f"(0.013975425f));
#if __CUDA_ARCH__ == 1030
                float _abs_81;
                asm("abs.f32 %0, %1;" : "=f"(_abs_81) : "f"(sg_dot));
                float _max_80;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_80) : "f"(_abs_81), "f"(clamp));
#else
                float _abs_77;
                asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(sg_dot));
                float _max_76;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_76) : "f"(_abs_77), "f"(clamp));
#endif
                float sg_mag;
#if __CUDA_ARCH__ == 1030
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_80));
#else
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_76));
#endif
                float sg_u = __uint_as_float(__as_u32(sg_mag) & 2147483647 | __as_u32(sg_dot) & 2147483648u);
#if __CUDA_ARCH__ == 1030
                float _neg_83;
                asm("neg.f32 %0, %1;" : "=f"(_neg_83) : "f"(sg_u));
                float _exp_82 = __nv_expf(_neg_83);
                float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_82);
#else
                float _neg_79;
                asm("neg.f32 %0, %1;" : "=f"(_neg_79) : "f"(sg_u));
                float _exp_78 = __nv_expf(_neg_79);
                float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_78);
#endif
                float sd_mag;
                asm("abs.f32 %0, %1;" : "=f"(sd_mag) : "f"(sg_dot));
                float sd_du = 0.0f;
                if (sd_mag > clamp) {
#if __CUDA_ARCH__ == 1030
                    float _sqrt_85;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_85) : "f"(sd_mag));
                    float _fdiv_84 = __fdiv_rn(0.5f, _sqrt_85);
                    sd_du = _fdiv_84;
#else
                    float _sqrt_81;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_81) : "f"(sd_mag));
                    float _fdiv_80 = __fdiv_rn(0.5f, _sqrt_81);
                    sd_du = _fdiv_80;
#endif
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
#if __CUDA_ARCH__ == 1030
            for (int i_10 = 0; i_10 < 8; i_10++) {
                tg_dv[i_10] = 0.0f;
#else
            for (int i_6 = 0; i_6 < 8; i_6++) {
                tg_dv[i_6] = 0.0f;
#endif
            }
            #pragma unroll
#if __CUDA_ARCH__ == 1030
            for (int s_23 = 0; s_23 < 4; s_23++) {
#else
            for (int s_17 = 0; s_17 < 4; s_17++) {
#endif
                float tg_f[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tg_f[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg_f[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg_f[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg_f[(0) + 3]))
#if __CUDA_ARCH__ == 1030
                    : "r"(fac_addr + (unsigned int)(16 * s_23)));
#else
                    : "r"(fac_addr + (unsigned int)(16 * s_17)));
#endif
                float tg_sg = tg_f[0];
                float tg_sc = tg_f[1];
                float tg_sh = tg_f[2];
                float tg_sk = tg_f[3];
                float2 tg_g2 = make_float2(tg_sg, tg_sg);
                float2 tg_c2 = make_float2(tg_sc, tg_sc);
#if __CUDA_ARCH__ == 1030
                float _neg_86;
                asm("neg.f32 %0, %1;" : "=f"(_neg_86) : "f"(tg_sh));
                float _neg_87;
                asm("neg.f32 %0, %1;" : "=f"(_neg_87) : "f"(tg_sh));
                float2 tg_h2 = make_float2(_neg_86, _neg_87);
                float _neg_88;
                asm("neg.f32 %0, %1;" : "=f"(_neg_88) : "f"(tg_sk));
                float _neg_89;
                asm("neg.f32 %0, %1;" : "=f"(_neg_89) : "f"(tg_sk));
                float2 tg_k2 = make_float2(_neg_88, _neg_89);
#else
                float _neg_82;
                asm("neg.f32 %0, %1;" : "=f"(_neg_82) : "f"(tg_sh));
                float _neg_83;
                asm("neg.f32 %0, %1;" : "=f"(_neg_83) : "f"(tg_sh));
                float2 tg_h2 = make_float2(_neg_82, _neg_83);
                float _neg_84;
                asm("neg.f32 %0, %1;" : "=f"(_neg_84) : "f"(tg_sk));
                float _neg_85;
                asm("neg.f32 %0, %1;" : "=f"(_neg_85) : "f"(tg_sk));
                float2 tg_k2 = make_float2(_neg_84, _neg_85);
#endif
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int half_6 = 0; half_6 < 2; half_6++) {
#else
                for (int half_4 = 0; half_4 < 2; half_4++) {
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if __CUDA_ARCH__ == 1030
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_23 + 4 * half_6])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_23 + 4 * half_6) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_23 + 4 * half_6) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_23 + 4 * half_6) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_23 + half_6) * 160 + tid))));
#else
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_17 + 4 * half_4])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_17 + 4 * half_4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_17 + 4 * half_4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_17 + 4 * half_4) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_17 + half_4) * 160 + tid))));
#endif
                }
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int half_7 = 0; half_7 < 2; half_7++) {
#else
                for (int half_5 = 0; half_5 < 2; half_5++) {
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if __CUDA_ARCH__ == 1030
                        : "=r"(*reinterpret_cast<uint32_t*>(&dwa[8 * s_23 + 4 * half_7])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_23 + 4 * half_7) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_23 + 4 * half_7) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_23 + 4 * half_7) + 3]))
                        : "r"(dslots_addr + (unsigned int)(16 * ((2 * s_23 + half_7) * 160 + tid))));
#else
                        : "=r"(*reinterpret_cast<uint32_t*>(&dwa[8 * s_17 + 4 * half_5])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_17 + 4 * half_5) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_17 + 4 * half_5) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_17 + 4 * half_5) + 3]))
                        : "r"(dslots_addr + (unsigned int)(16 * ((2 * s_17 + half_5) * 160 + tid))));
#endif
                }
                float tg_dx[8];
                float tg_dk[8];
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int m_7 = 0; m_7 < 4; m_7++) {
                    float2 tg_hv = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_23 + m_7]));
                    float2 tg_kv = cake_bf16x2_to_f32x2(__as_bf16x2(kw[4 * s_23 + m_7]));
                    float2 tg_gv = cake_bf16x2_to_f32x2(__as_bf16x2(gw[4 * s_23 + m_7]));
                    float2 tg_wv = make_float2(wf[8 * s_23 + 2 * m_7], wf[8 * s_23 + 2 * m_7 + 1]);
#else
                for (int m_5 = 0; m_5 < 4; m_5++) {
                    float2 tg_hv = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_17 + m_5]));
                    float2 tg_kv = cake_bf16x2_to_f32x2(__as_bf16x2(kw[4 * s_17 + m_5]));
                    float2 tg_gv = cake_bf16x2_to_f32x2(__as_bf16x2(gw[4 * s_17 + m_5]));
                    float2 tg_wv = make_float2(wf[8 * s_17 + 2 * m_5], wf[8 * s_17 + 2 * m_5 + 1]);
#endif
                    float2 tg_ox = fma_f32x2_rn_noftz(tg_c2, fma_f32x2_rn_noftz(tg_h2, tg_hv, mul_f32x2_noftz(tg_wv, tg_kv)), tg_gv);
#if __CUDA_ARCH__ == 1030
                    float2 _f32x2_mul_b_90 = fma_f32x2_rn_noftz(tg_k2, tg_kv, mul_f32x2_noftz(tg_wv, tg_hv));
#else
                    float2 _f32x2_mul_b_86 = fma_f32x2_rn_noftz(tg_k2, tg_kv, mul_f32x2_noftz(tg_wv, tg_hv));
#endif
                    float2 tg_ok;
#if __CUDA_ARCH__ == 1030
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_ok) : "l"(*(const unsigned long long*)&tg_c2), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
                    float2 tg_ov = fma_f32x2_rn_noftz(tg_g2, tg_gv, make_float2(tg_dv[2 * m_7], tg_dv[2 * m_7 + 1]));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_ok) : "l"(*(const unsigned long long*)&tg_c2), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                    float2 tg_ov = fma_f32x2_rn_noftz(tg_g2, tg_gv, make_float2(tg_dv[2 * m_5], tg_dv[2 * m_5 + 1]));
#endif
                    float2 tg_hk;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_hk) : "l"(*(const unsigned long long*)&tg_hv), "l"(*(const unsigned long long*)&tg_kv));
#if __CUDA_ARCH__ == 1030
                    float2 tg_ow = fma_f32x2_rn_noftz(tg_c2, tg_hk, make_float2(dwa[8 * s_23 + 2 * m_7], dwa[8 * s_23 + 2 * m_7 + 1]));
                    tg_dx[2 * m_7] = tg_ox.x;
                    tg_dk[2 * m_7] = tg_ok.x;
                    tg_dv[2 * m_7] = tg_ov.x;
                    dwa[8 * s_23 + 2 * m_7] = tg_ow.x;
                    tg_dx[2 * m_7 + 1] = tg_ox.y;
                    tg_dk[2 * m_7 + 1] = tg_ok.y;
                    tg_dv[2 * m_7 + 1] = tg_ov.y;
                    dwa[8 * s_23 + 2 * m_7 + 1] = tg_ow.y;
#else
                    float2 tg_ow = fma_f32x2_rn_noftz(tg_c2, tg_hk, make_float2(dwa[8 * s_17 + 2 * m_5], dwa[8 * s_17 + 2 * m_5 + 1]));
                    tg_dx[2 * m_5] = tg_ox.x;
                    tg_dk[2 * m_5] = tg_ok.x;
                    tg_dv[2 * m_5] = tg_ov.x;
                    dwa[8 * s_17 + 2 * m_5] = tg_ow.x;
                    tg_dx[2 * m_5 + 1] = tg_ox.y;
                    tg_dk[2 * m_5 + 1] = tg_ok.y;
                    tg_dv[2 * m_5 + 1] = tg_ov.y;
                    dwa[8 * s_17 + 2 * m_5 + 1] = tg_ow.y;
#endif
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
#if __CUDA_ARCH__ == 1030
                reinterpret_cast<int4*>(dx + (tg_x + (long long)(s_23 * 5120)))[0] = reinterpret_cast<int4*>(tg_px)[0];
                reinterpret_cast<int4*>(dkey + (tg_k + (long long)s_23 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(tg_pk)[0];
#else
                reinterpret_cast<int4*>(dx + (tg_x + (long long)(s_17 * 5120)))[0] = reinterpret_cast<int4*>(tg_px)[0];
                reinterpret_cast<int4*>(dkey + (tg_k + (long long)s_17 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(tg_pk)[0];
#endif
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int half_8 = 0; half_8 < 2; half_8++) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dslots_addr + (unsigned int)(16 * ((2 * s_23 + half_8) * 160 + tid))), "f"(dwa[8 * s_23 + 4 * half_8]), "f"(dwa[8 * s_23 + 4 * half_8 + 1]), "f"(dwa[8 * s_23 + 4 * half_8 + 2]), "f"(dwa[8 * s_23 + 4 * half_8 + 3]) : "memory");
#else
                for (int half_6 = 0; half_6 < 2; half_6++) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dslots_addr + (unsigned int)(16 * ((2 * s_17 + half_6) * 160 + tid))), "f"(dwa[8 * s_17 + 4 * half_6]), "f"(dwa[8 * s_17 + 4 * half_6 + 1]), "f"(dwa[8 * s_17 + 4 * half_6 + 2]), "f"(dwa[8 * s_17 + 4 * half_6 + 3]) : "memory");
#endif
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
#if __CUDA_ARCH__ == 1030
                for (int s_24 = 0; s_24 < 4; s_24++) {
#else
                for (int s_18 = 0; s_18 < 4; s_18++) {
#endif
                    {
#if __CUDA_ARCH__ == 1030
                        const uint4* _vptr_91 = reinterpret_cast<const uint4*>(dout + (ls_at_3 + (long long)(s_24 * 5120)) + 0);
                        uint4* _vdst_91 = reinterpret_cast<uint4*>(&gn[4 * s_24]);
#else
                        const uint4* _vptr_87 = reinterpret_cast<const uint4*>(dout + (ls_at_3 + (long long)(s_18 * 5120)) + 0);
                        uint4* _vdst_87 = reinterpret_cast<uint4*>(&gn[4 * s_18]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_91[_blk] = _vptr_91[_blk];
#else
                            _vdst_87[_blk] = _vptr_87[_blk];
#endif
                        }
                    }
                }
                long long ls_at_4 = (long long)pn_tok_1 * 20480ll + (long long)col;
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int s_25 = 0; s_25 < 4; s_25++) {
#else
                for (int s_19 = 0; s_19 < 4; s_19++) {
#endif
                    {
#if __CUDA_ARCH__ == 1030
                        const uint4* _vptr_92 = reinterpret_cast<const uint4*>(x + (ls_at_4 + (long long)(s_25 * 5120)) + 0);
                        uint4* _vdst_92 = reinterpret_cast<uint4*>(&xn[4 * s_25]);
#else
                        const uint4* _vptr_88 = reinterpret_cast<const uint4*>(x + (ls_at_4 + (long long)(s_19 * 5120)) + 0);
                        uint4* _vdst_88 = reinterpret_cast<uint4*>(&xn[4 * s_19]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_92[_blk] = _vptr_92[_blk];
#else
                            _vdst_88[_blk] = _vptr_88[_blk];
#endif
                        }
                    }
                }
                long long lk_at_1 = (long long)pn_tok_1 * key_stride_p0 + (long long)col;
                #pragma unroll
#if __CUDA_ARCH__ == 1030
                for (int s_26 = 0; s_26 < 4; s_26++) {
#else
                for (int s_20 = 0; s_20 < 4; s_20++) {
#endif
                    {
#if __CUDA_ARCH__ == 1030
                        const uint4* _vptr_93 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_26 * key_stride_p1) + 0);
                        uint4* _vdst_93 = reinterpret_cast<uint4*>(&kn[4 * s_26]);
#else
                        const uint4* _vptr_89 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_20 * key_stride_p1) + 0);
                        uint4* _vdst_89 = reinterpret_cast<uint4*>(&kn[4 * s_20]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_93[_blk] = _vptr_93[_blk];
#else
                            _vdst_89[_blk] = _vptr_89[_blk];
#endif
                        }
                    }
                }
                {
#if __CUDA_ARCH__ == 1030
                    const uint4* _vptr_94 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_94 = reinterpret_cast<uint4*>(&vn[0]);
#else
                    const uint4* _vptr_90 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_90 = reinterpret_cast<uint4*>(&vn[0]);
#endif
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                        _vdst_94[_blk] = _vptr_94[_blk];
#else
                        _vdst_90[_blk] = _vptr_90[_blk];
#endif
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
#if __CUDA_ARCH__ == 1030
            for (int s_27 = 0; s_27 < 4; s_27++) {
                reinterpret_cast<int4*>(dx + (tg_x + (long long)(s_27 * 5120)))[0] = reinterpret_cast<int4*>(gw + (4 * s_27))[0];
                reinterpret_cast<int4*>(dkey + (tg_k + (long long)s_27 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(zw)[0];
#else
            for (int s_21 = 0; s_21 < 4; s_21++) {
                reinterpret_cast<int4*>(dx + (tg_x + (long long)(s_21 * 5120)))[0] = reinterpret_cast<int4*>(gw + (4 * s_21))[0];
                reinterpret_cast<int4*>(dkey + (tg_k + (long long)s_21 * dkey_stride_p1))[0] = reinterpret_cast<int4*>(zw)[0];
#endif
            }
            reinterpret_cast<int4*>(dvalue + tg_v)[0] = reinterpret_cast<int4*>(zw)[0];
        }
        #pragma unroll
#if __CUDA_ARCH__ == 1030
        for (int i_11 = 0; i_11 < 16; i_11++) {
            gw[i_11] = gn[i_11];
            xw[i_11] = xn[i_11];
            kw[i_11] = kn[i_11];
#else
        for (int i_7 = 0; i_7 < 16; i_7++) {
            gw[i_7] = gn[i_7];
            xw[i_7] = xn[i_7];
            kw[i_7] = kn[i_7];
#endif
        }
        #pragma unroll
#if __CUDA_ARCH__ == 1030
        for (int i_12 = 0; i_12 < 4; i_12++) {
            vw[i_12] = vn[i_12];
#else
        for (int i_8 = 0; i_8 < 4; i_8++) {
            vw[i_8] = vn[i_8];
#endif
        }
        keep = keep_n;
    }
    #pragma unroll
#if __CUDA_ARCH__ == 1030
    for (int s_28 = 0; s_28 < 4; s_28++) {
#else
    for (int s_22 = 0; s_22 < 4; s_22++) {
#endif
        #pragma unroll
#if __CUDA_ARCH__ == 1030
        for (int half_9 = 0; half_9 < 2; half_9++) {
#else
        for (int half_7 = 0; half_7 < 2; half_7++) {
#endif
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if __CUDA_ARCH__ == 1030
                : "=r"(*reinterpret_cast<uint32_t*>(&dwa[8 * s_28 + 4 * half_9])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_28 + 4 * half_9) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_28 + 4 * half_9) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_28 + 4 * half_9) + 3]))
                : "r"(dslots_addr + (unsigned int)(16 * ((2 * s_28 + half_9) * 160 + tid))));
#else
                : "=r"(*reinterpret_cast<uint32_t*>(&dwa[8 * s_22 + 4 * half_7])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_22 + 4 * half_7) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_22 + 4 * half_7) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa[(8 * s_22 + 4 * half_7) + 3]))
                : "r"(dslots_addr + (unsigned int)(16 * ((2 * s_22 + half_7) * 160 + tid))));
#endif
        }
    }
    #pragma unroll
#if __CUDA_ARCH__ == 1030
    for (int s_29 = 0; s_29 < 4; s_29++) {
#else
    for (int s_23 = 0; s_23 < 4; s_23++) {
#endif
        #pragma unroll
#if __CUDA_ARCH__ == 1030
        for (int half_10 = 0; half_10 < 2; half_10++) {
#else
        for (int half_8 = 0; half_8 < 2; half_8++) {
#endif
            {
#if __CUDA_ARCH__ == 1030
                float4 _v4 = make_float4(dwa[8 * s_29 + 4 * half_10 + 0], dwa[8 * s_29 + 4 * half_10 + 1], dwa[8 * s_29 + 4 * half_10 + 2], dwa[8 * s_29 + 4 * half_10 + 3]);
                *reinterpret_cast<float4*>((dw_part + ((long long)first * 20480ll + (long long)(s_29 * 5120 + col + 4 * half_10))) + 0) = _v4;
#else
                float4 _v4 = make_float4(dwa[8 * s_23 + 4 * half_8 + 0], dwa[8 * s_23 + 4 * half_8 + 1], dwa[8 * s_23 + 4 * half_8 + 2], dwa[8 * s_23 + 4 * half_8 + 3]);
                *reinterpret_cast<float4*>((dw_part + ((long long)first * 20480ll + (long long)(s_23 * 5120 + col + 4 * half_8))) + 0) = _v4;
#endif
            }
        }
    }
}

} // extern "C"
