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
kernel_cake_engram_gate_det_train_d0c6b3417c9cd0b85e9f(__nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, int* __restrict__ q_weight, int* __restrict__ k_weight, bool* __restrict__ mask, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dkey, __nv_bfloat16* __restrict__ dvalue, float* __restrict__ dw_part, int T, float eps, float clamp, int wmode, long long q_weight_size_p0, long long q_weight_size_p1, long long k_weight_size_p0, long long k_weight_size_p1, long long key_stride_p0, long long key_stride_p1, long long value_stride_p0, long long dkey_stride_p0, long long dkey_stride_p1, long long dvalue_stride_p0)
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
                float sp_hk1_0[2] = {0.0f, 0.0f};
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_0[0]) : "h"(static_cast<uint16_t>((xw[4 * s_18]))), "h"(static_cast<uint16_t>((kw[4 * s_18]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_0[1]) : "h"(static_cast<uint16_t>((xw[4 * s_18]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_18]) >> 16)));
                float sp_w1_0 = wf[8 * s_18];
#else
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_0[0]) : "h"(static_cast<uint16_t>((xw[4 * s_12]))), "h"(static_cast<uint16_t>((kw[4 * s_12]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_0[1]) : "h"(static_cast<uint16_t>((xw[4 * s_12]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_12]) >> 16)));
                float sp_w1_0 = wf[8 * s_12];
#endif
                float sp_p1_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_0) : "f"(sp_w1_0), "f"(sp_hk1_0[0]));
#if __CUDA_ARCH__ == 1030
                float _neg_14;
                asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(sp_p1_0));
#else
                float _neg_10;
                asm("neg.f32 %0, %1;" : "=f"(_neg_10) : "f"(sp_p1_0));
#endif
                float sp_q1_0;
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_0) : "f"(sp_w1_0), "f"(sp_hk1_0[0]), "f"(_neg_14));
                float _add_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(0.0f), "f"(sp_p1_0));
                sp_s = _add_15;
                float _add_16;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(0.0f), "f"(sp_q1_0));
                sp_e = _add_16;
                float sp_w1_1 = wf[8 * s_18 + 1];
#else
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_0) : "f"(sp_w1_0), "f"(sp_hk1_0[0]), "f"(_neg_10));
                float _add_11;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(0.0f), "f"(sp_p1_0));
                sp_s = _add_11;
                float _add_12;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(0.0f), "f"(sp_q1_0));
                sp_e = _add_12;
                float sp_w1_1 = wf[8 * s_12 + 1];
#endif
                float sp_p1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_1) : "f"(sp_w1_1), "f"(sp_hk1_0[1]));
#if __CUDA_ARCH__ == 1030
                float _neg_17;
                asm("neg.f32 %0, %1;" : "=f"(_neg_17) : "f"(sp_p1_1));
#else
                float _neg_13;
                asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(sp_p1_1));
#endif
                float sp_q1_1;
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_1) : "f"(sp_w1_1), "f"(sp_hk1_0[1]), "f"(_neg_17));
#else
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_1) : "f"(sp_w1_1), "f"(sp_hk1_0[1]), "f"(_neg_13));
#endif
                float ts_s;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(sp_s), "f"(sp_p1_1));
                float ts_bb;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                float _sub_19;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(ts_s), "f"(ts_bb));
                float _sub_18;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(sp_s), "f"(_sub_19));
                float _sub_20;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(sp_p1_1), "f"(ts_bb));
#else
                float _sub_15;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(ts_s), "f"(ts_bb));
                float _sub_14;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(sp_s), "f"(_sub_15));
                float _sub_16;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(sp_p1_1), "f"(ts_bb));
#endif
                float ts_e;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_18), "f"(_sub_20));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_14), "f"(_sub_16));
#endif
                sp_s = ts_s;
#if __CUDA_ARCH__ == 1030
                float _add_22;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(sp_q1_1), "f"(ts_e));
                float _add_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(sp_e), "f"(_add_22));
                sp_e = _add_21;
#else
                float _add_18;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(sp_q1_1), "f"(ts_e));
                float _add_17;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(sp_e), "f"(_add_18));
                sp_e = _add_17;
#endif
                float sp_hk1_1[2] = {0.0f, 0.0f};
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_1[0]) : "h"(static_cast<uint16_t>((xw[4 * s_18 + 1]))), "h"(static_cast<uint16_t>((kw[4 * s_18 + 1]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_1[1]) : "h"(static_cast<uint16_t>((xw[4 * s_18 + 1]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_18 + 1]) >> 16)));
                float sp_w1_2 = wf[8 * s_18 + 2];
#else
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_1[0]) : "h"(static_cast<uint16_t>((xw[4 * s_12 + 1]))), "h"(static_cast<uint16_t>((kw[4 * s_12 + 1]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_1[1]) : "h"(static_cast<uint16_t>((xw[4 * s_12 + 1]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_12 + 1]) >> 16)));
                float sp_w1_2 = wf[8 * s_12 + 2];
#endif
                float sp_p1_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_2) : "f"(sp_w1_2), "f"(sp_hk1_1[0]));
#if __CUDA_ARCH__ == 1030
                float _neg_23;
                asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(sp_p1_2));
#else
                float _neg_19;
                asm("neg.f32 %0, %1;" : "=f"(_neg_19) : "f"(sp_p1_2));
#endif
                float sp_q1_2;
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_2) : "f"(sp_w1_2), "f"(sp_hk1_1[0]), "f"(_neg_23));
#else
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_2) : "f"(sp_w1_2), "f"(sp_hk1_1[0]), "f"(_neg_19));
#endif
                float ts_s_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(sp_s), "f"(sp_p1_2));
                float ts_bb_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_0), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                float _sub_25;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(ts_s_0), "f"(ts_bb_0));
                float _sub_24;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(sp_s), "f"(_sub_25));
                float _sub_26;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(sp_p1_2), "f"(ts_bb_0));
#else
                float _sub_21;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(ts_s_0), "f"(ts_bb_0));
                float _sub_20;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(sp_s), "f"(_sub_21));
                float _sub_22;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(sp_p1_2), "f"(ts_bb_0));
#endif
                float ts_e_0;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_24), "f"(_sub_26));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_20), "f"(_sub_22));
#endif
                sp_s = ts_s_0;
#if __CUDA_ARCH__ == 1030
                float _add_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(sp_q1_2), "f"(ts_e_0));
                float _add_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(sp_e), "f"(_add_28));
                sp_e = _add_27;
                float sp_w1_3 = wf[8 * s_18 + 2 + 1];
#else
                float _add_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(sp_q1_2), "f"(ts_e_0));
                float _add_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(sp_e), "f"(_add_24));
                sp_e = _add_23;
                float sp_w1_3 = wf[8 * s_12 + 2 + 1];
#endif
                float sp_p1_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_3) : "f"(sp_w1_3), "f"(sp_hk1_1[1]));
#if __CUDA_ARCH__ == 1030
                float _neg_29;
                asm("neg.f32 %0, %1;" : "=f"(_neg_29) : "f"(sp_p1_3));
#else
                float _neg_25;
                asm("neg.f32 %0, %1;" : "=f"(_neg_25) : "f"(sp_p1_3));
#endif
                float sp_q1_3;
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_3) : "f"(sp_w1_3), "f"(sp_hk1_1[1]), "f"(_neg_29));
#else
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_3) : "f"(sp_w1_3), "f"(sp_hk1_1[1]), "f"(_neg_25));
#endif
                float ts_s_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(sp_s), "f"(sp_p1_3));
                float ts_bb_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_1), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                float _sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(ts_s_1), "f"(ts_bb_1));
                float _sub_30;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(sp_s), "f"(_sub_31));
                float _sub_32;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(sp_p1_3), "f"(ts_bb_1));
#else
                float _sub_27;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(ts_s_1), "f"(ts_bb_1));
                float _sub_26;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(sp_s), "f"(_sub_27));
                float _sub_28;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(sp_p1_3), "f"(ts_bb_1));
#endif
                float ts_e_1;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_30), "f"(_sub_32));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_26), "f"(_sub_28));
#endif
                sp_s = ts_s_1;
#if __CUDA_ARCH__ == 1030
                float _add_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(sp_q1_3), "f"(ts_e_1));
                float _add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(sp_e), "f"(_add_34));
                sp_e = _add_33;
#else
                float _add_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(sp_q1_3), "f"(ts_e_1));
                float _add_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(sp_e), "f"(_add_30));
                sp_e = _add_29;
#endif
                float sp_hk1_2[2] = {0.0f, 0.0f};
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_2[0]) : "h"(static_cast<uint16_t>((xw[4 * s_18 + 2]))), "h"(static_cast<uint16_t>((kw[4 * s_18 + 2]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_2[1]) : "h"(static_cast<uint16_t>((xw[4 * s_18 + 2]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_18 + 2]) >> 16)));
                float sp_w1_4 = wf[8 * s_18 + 4];
#else
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_2[0]) : "h"(static_cast<uint16_t>((xw[4 * s_12 + 2]))), "h"(static_cast<uint16_t>((kw[4 * s_12 + 2]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_2[1]) : "h"(static_cast<uint16_t>((xw[4 * s_12 + 2]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_12 + 2]) >> 16)));
                float sp_w1_4 = wf[8 * s_12 + 4];
#endif
                float sp_p1_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_4) : "f"(sp_w1_4), "f"(sp_hk1_2[0]));
#if __CUDA_ARCH__ == 1030
                float _neg_35;
                asm("neg.f32 %0, %1;" : "=f"(_neg_35) : "f"(sp_p1_4));
#else
                float _neg_31;
                asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(sp_p1_4));
#endif
                float sp_q1_4;
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_4) : "f"(sp_w1_4), "f"(sp_hk1_2[0]), "f"(_neg_35));
#else
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_4) : "f"(sp_w1_4), "f"(sp_hk1_2[0]), "f"(_neg_31));
#endif
                float ts_s_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(sp_s), "f"(sp_p1_4));
                float ts_bb_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_2), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(ts_s_2), "f"(ts_bb_2));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(sp_s), "f"(_sub_37));
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(sp_p1_4), "f"(ts_bb_2));
#else
                float _sub_33;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(ts_s_2), "f"(ts_bb_2));
                float _sub_32;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(sp_s), "f"(_sub_33));
                float _sub_34;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(sp_p1_4), "f"(ts_bb_2));
#endif
                float ts_e_2;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_36), "f"(_sub_38));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_32), "f"(_sub_34));
#endif
                sp_s = ts_s_2;
#if __CUDA_ARCH__ == 1030
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(sp_q1_4), "f"(ts_e_2));
                float _add_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(sp_e), "f"(_add_40));
                sp_e = _add_39;
                float sp_w1_5 = wf[8 * s_18 + 4 + 1];
#else
                float _add_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(sp_q1_4), "f"(ts_e_2));
                float _add_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(sp_e), "f"(_add_36));
                sp_e = _add_35;
                float sp_w1_5 = wf[8 * s_12 + 4 + 1];
#endif
                float sp_p1_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_5) : "f"(sp_w1_5), "f"(sp_hk1_2[1]));
#if __CUDA_ARCH__ == 1030
                float _neg_41;
                asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(sp_p1_5));
#else
                float _neg_37;
                asm("neg.f32 %0, %1;" : "=f"(_neg_37) : "f"(sp_p1_5));
#endif
                float sp_q1_5;
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_5) : "f"(sp_w1_5), "f"(sp_hk1_2[1]), "f"(_neg_41));
#else
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_5) : "f"(sp_w1_5), "f"(sp_hk1_2[1]), "f"(_neg_37));
#endif
                float ts_s_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3) : "f"(sp_s), "f"(sp_p1_5));
                float ts_bb_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3) : "f"(ts_s_3), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(ts_s_3), "f"(ts_bb_3));
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(sp_s), "f"(_sub_43));
                float _sub_44;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(sp_p1_5), "f"(ts_bb_3));
#else
                float _sub_39;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(ts_s_3), "f"(ts_bb_3));
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(sp_s), "f"(_sub_39));
                float _sub_40;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(sp_p1_5), "f"(ts_bb_3));
#endif
                float ts_e_3;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_42), "f"(_sub_44));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_38), "f"(_sub_40));
#endif
                sp_s = ts_s_3;
#if __CUDA_ARCH__ == 1030
                float _add_46;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(sp_q1_5), "f"(ts_e_3));
                float _add_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(sp_e), "f"(_add_46));
                sp_e = _add_45;
#else
                float _add_42;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(sp_q1_5), "f"(ts_e_3));
                float _add_41;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(sp_e), "f"(_add_42));
                sp_e = _add_41;
#endif
                float sp_hk1_3[2] = {0.0f, 0.0f};
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_3[0]) : "h"(static_cast<uint16_t>((xw[4 * s_18 + 3]))), "h"(static_cast<uint16_t>((kw[4 * s_18 + 3]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_3[1]) : "h"(static_cast<uint16_t>((xw[4 * s_18 + 3]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_18 + 3]) >> 16)));
                float sp_w1_6 = wf[8 * s_18 + 6];
#else
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_3[0]) : "h"(static_cast<uint16_t>((xw[4 * s_12 + 3]))), "h"(static_cast<uint16_t>((kw[4 * s_12 + 3]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sp_hk1_3[1]) : "h"(static_cast<uint16_t>((xw[4 * s_12 + 3]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_12 + 3]) >> 16)));
                float sp_w1_6 = wf[8 * s_12 + 6];
#endif
                float sp_p1_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_6) : "f"(sp_w1_6), "f"(sp_hk1_3[0]));
#if __CUDA_ARCH__ == 1030
                float _neg_47;
                asm("neg.f32 %0, %1;" : "=f"(_neg_47) : "f"(sp_p1_6));
#else
                float _neg_43;
                asm("neg.f32 %0, %1;" : "=f"(_neg_43) : "f"(sp_p1_6));
#endif
                float sp_q1_6;
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_6) : "f"(sp_w1_6), "f"(sp_hk1_3[0]), "f"(_neg_47));
#else
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_6) : "f"(sp_w1_6), "f"(sp_hk1_3[0]), "f"(_neg_43));
#endif
                float ts_s_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_4) : "f"(sp_s), "f"(sp_p1_6));
                float ts_bb_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_4) : "f"(ts_s_4), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                float _sub_49;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(ts_s_4), "f"(ts_bb_4));
                float _sub_48;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(sp_s), "f"(_sub_49));
                float _sub_50;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(sp_p1_6), "f"(ts_bb_4));
#else
                float _sub_45;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(ts_s_4), "f"(ts_bb_4));
                float _sub_44;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(sp_s), "f"(_sub_45));
                float _sub_46;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(sp_p1_6), "f"(ts_bb_4));
#endif
                float ts_e_4;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_48), "f"(_sub_50));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_44), "f"(_sub_46));
#endif
                sp_s = ts_s_4;
#if __CUDA_ARCH__ == 1030
                float _add_52;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(sp_q1_6), "f"(ts_e_4));
                float _add_51;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(sp_e), "f"(_add_52));
                sp_e = _add_51;
                float sp_w1_7 = wf[8 * s_18 + 6 + 1];
#else
                float _add_48;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(sp_q1_6), "f"(ts_e_4));
                float _add_47;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(sp_e), "f"(_add_48));
                sp_e = _add_47;
                float sp_w1_7 = wf[8 * s_12 + 6 + 1];
#endif
                float sp_p1_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sp_p1_7) : "f"(sp_w1_7), "f"(sp_hk1_3[1]));
#if __CUDA_ARCH__ == 1030
                float _neg_53;
                asm("neg.f32 %0, %1;" : "=f"(_neg_53) : "f"(sp_p1_7));
#else
                float _neg_49;
                asm("neg.f32 %0, %1;" : "=f"(_neg_49) : "f"(sp_p1_7));
#endif
                float sp_q1_7;
#if __CUDA_ARCH__ == 1030
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_7) : "f"(sp_w1_7), "f"(sp_hk1_3[1]), "f"(_neg_53));
#else
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sp_q1_7) : "f"(sp_w1_7), "f"(sp_hk1_3[1]), "f"(_neg_49));
#endif
                float ts_s_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_5) : "f"(sp_s), "f"(sp_p1_7));
                float ts_bb_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_5) : "f"(ts_s_5), "f"(sp_s));
#if __CUDA_ARCH__ == 1030
                float _sub_55;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(ts_s_5), "f"(ts_bb_5));
                float _sub_54;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(sp_s), "f"(_sub_55));
                float _sub_56;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(sp_p1_7), "f"(ts_bb_5));
#else
                float _sub_51;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(ts_s_5), "f"(ts_bb_5));
                float _sub_50;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(sp_s), "f"(_sub_51));
                float _sub_52;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(sp_p1_7), "f"(ts_bb_5));
#endif
                float ts_e_5;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_5) : "f"(_sub_54), "f"(_sub_56));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_5) : "f"(_sub_50), "f"(_sub_52));
#endif
                sp_s = ts_s_5;
#if __CUDA_ARCH__ == 1030
                float _add_58;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(sp_q1_7), "f"(ts_e_5));
                float _add_57;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(sp_e), "f"(_add_58));
                sp_e = _add_57;
                ps[s_18] = sp_s;
                pe[s_18] = sp_e;
#else
                float _add_54;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(sp_q1_7), "f"(ts_e_5));
                float _add_53;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(sp_e), "f"(_add_54));
                sp_e = _add_53;
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
                for (int m_5 = 0; m_5 < 4; m_5++) {
                    float _dot_59;
#else
                for (int m_3 = 0; m_3 < 4; m_3++) {
                    float _dot_55;
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
                        : "=f"(_dot_59) : "r"(gw[4 * s_19 + m_5]), "r"(vw[m_5]), "f"(tl_q));
                    tl_q = _dot_59;
#else
                        : "=f"(_dot_55) : "r"(gw[4 * s_13 + m_3]), "r"(vw[m_3]), "f"(tl_q));
                    tl_q = _dot_55;
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
                        const uint4* _vptr_60 = reinterpret_cast<const uint4*>(dout + (ls_at_1 + (long long)(s_20 * 5120)) + 0);
                        uint4* _vdst_60 = reinterpret_cast<uint4*>(&gn[4 * s_20]);
#else
                        const uint4* _vptr_56 = reinterpret_cast<const uint4*>(dout + (ls_at_1 + (long long)(s_14 * 5120)) + 0);
                        uint4* _vdst_56 = reinterpret_cast<uint4*>(&gn[4 * s_14]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_60[_blk] = _vptr_60[_blk];
#else
                            _vdst_56[_blk] = _vptr_56[_blk];
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
                        const uint4* _vptr_61 = reinterpret_cast<const uint4*>(x + (ls_at_2 + (long long)(s_21 * 5120)) + 0);
                        uint4* _vdst_61 = reinterpret_cast<uint4*>(&xn[4 * s_21]);
#else
                        const uint4* _vptr_57 = reinterpret_cast<const uint4*>(x + (ls_at_2 + (long long)(s_15 * 5120)) + 0);
                        uint4* _vdst_57 = reinterpret_cast<uint4*>(&xn[4 * s_15]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_61[_blk] = _vptr_61[_blk];
#else
                            _vdst_57[_blk] = _vptr_57[_blk];
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
                        const uint4* _vptr_62 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_22 * key_stride_p1) + 0);
                        uint4* _vdst_62 = reinterpret_cast<uint4*>(&kn[4 * s_22]);
#else
                        const uint4* _vptr_58 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_16 * key_stride_p1) + 0);
                        uint4* _vdst_58 = reinterpret_cast<uint4*>(&kn[4 * s_16]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_62[_blk] = _vptr_62[_blk];
#else
                            _vdst_58[_blk] = _vptr_58[_blk];
#endif
                        }
                    }
                }
                {
#if __CUDA_ARCH__ == 1030
                    const uint4* _vptr_63 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_63 = reinterpret_cast<uint4*>(&vn[0]);
#else
                    const uint4* _vptr_59 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_59 = reinterpret_cast<uint4*>(&vn[0]);
#endif
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                        _vdst_63[_blk] = _vptr_63[_blk];
#else
                        _vdst_59[_blk] = _vptr_59[_blk];
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
                float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_64;
#else
                float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_60;
#endif
            }
            #pragma unroll
            for (int i_1_1 = 0; i_1_1 < 4; i_1_1++) {
                int ws_upper_1 = lane & 8;
                float ws_keep_1 = ((ws_upper_1 != 0) ? plain[4 + i_1_1] : plain[i_1_1]);
                float ws_send_1 = ((ws_upper_1 != 0) ? plain[i_1_1] : plain[4 + i_1_1]);
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1_1] = ws_keep_1 + _shfl_xor_sync_65;
#else
                float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1_1] = ws_keep_1 + _shfl_xor_sync_61;
#endif
            }
            #pragma unroll
            for (int i_2_1 = 0; i_2_1 < 2; i_2_1++) {
                int ws_upper_2 = lane & 4;
                float ws_keep_2 = ((ws_upper_2 != 0) ? plain[2 + i_2_1] : plain[i_2_1]);
                float ws_send_2 = ((ws_upper_2 != 0) ? plain[i_2_1] : plain[2 + i_2_1]);
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2_1] = ws_keep_2 + _shfl_xor_sync_66;
#else
                float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2_1] = ws_keep_2 + _shfl_xor_sync_62;
#endif
            }
            #pragma unroll
            for (int i_3_1 = 0; i_3_1 < 1; i_3_1++) {
                int ws_upper_3 = lane & 2;
                float ws_keep_3 = ((ws_upper_3 != 0) ? plain[1 + i_3_1] : plain[i_3_1]);
                float ws_send_3 = ((ws_upper_3 != 0) ? plain[i_3_1] : plain[1 + i_3_1]);
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
                plain[i_3_1] = ws_keep_3 + _shfl_xor_sync_67;
#else
                float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
                plain[i_3_1] = ws_keep_3 + _shfl_xor_sync_63;
#endif
            }
            float ws_own = plain[0];
#if __CUDA_ARCH__ == 1030
            float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_68;
#else
            float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_64;
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
                float ts_s_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_6) : "f"(wp_ks_0), "f"(wp_rs_0));
                float ts_bb_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6) : "f"(ts_s_6), "f"(wp_ks_0));
#if __CUDA_ARCH__ == 1030
                float _sub_70;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(ts_s_6), "f"(ts_bb_6));
                float _sub_69;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(wp_ks_0), "f"(_sub_70));
                float _sub_71;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(wp_rs_0), "f"(ts_bb_6));
#else
                float _sub_66;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(ts_s_6), "f"(ts_bb_6));
                float _sub_65;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(wp_ks_0), "f"(_sub_66));
                float _sub_67;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(wp_rs_0), "f"(ts_bb_6));
#endif
                float ts_e_6;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6) : "f"(_sub_69), "f"(_sub_71));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6) : "f"(_sub_65), "f"(_sub_67));
#endif
                wp_ks_0 = ts_s_6;
#if __CUDA_ARCH__ == 1030
                float _add_73;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(wp_ke_0), "f"(wp_re_0));
                float _add_72;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_add_73), "f"(ts_e_6));
                wp_ke_0 = _add_72;
#else
                float _add_69;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(wp_ke_0), "f"(wp_re_0));
                float _add_68;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(ts_e_6));
                wp_ke_0 = _add_68;
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
                float ts_s_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7) : "f"(wp_ks_1), "f"(wp_rs_1));
                float ts_bb_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7) : "f"(ts_s_7), "f"(wp_ks_1));
#if __CUDA_ARCH__ == 1030
                float _sub_75;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(ts_s_7), "f"(ts_bb_7));
                float _sub_74;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(wp_ks_1), "f"(_sub_75));
                float _sub_76;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(wp_rs_1), "f"(ts_bb_7));
#else
                float _sub_71;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(ts_s_7), "f"(ts_bb_7));
                float _sub_70;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(wp_ks_1), "f"(_sub_71));
                float _sub_72;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(wp_rs_1), "f"(ts_bb_7));
#endif
                float ts_e_7;
#if __CUDA_ARCH__ == 1030
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7) : "f"(_sub_74), "f"(_sub_76));
#else
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7) : "f"(_sub_70), "f"(_sub_72));
#endif
                wp_ks_1 = ts_s_7;
#if __CUDA_ARCH__ == 1030
                float _add_78;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(wp_ke_1), "f"(wp_re_1));
                float _add_77;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_add_78), "f"(ts_e_7));
                wp_ke_1 = _add_77;
                ps[i_5_1] = wp_ks_1;
                pe[i_5_1] = wp_ke_1;
#else
                float _add_74;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(wp_ke_1), "f"(wp_re_1));
                float _add_73;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_74), "f"(ts_e_7));
                wp_ke_1 = _add_73;
                ps[i_5] = wp_ks_1;
                pe[i_5] = wp_ke_1;
#endif
            }
            float wp_s = ps[0];
            float wp_e = pe[0];
            float wp_bs_0 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 4);
            float wp_be_0 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 4);
            float ts_s_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8) : "f"(wp_s), "f"(wp_bs_0));
            float ts_bb_8;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8) : "f"(ts_s_8), "f"(wp_s));
#if __CUDA_ARCH__ == 1030
            float _sub_80;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(ts_s_8), "f"(ts_bb_8));
            float _sub_79;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(wp_s), "f"(_sub_80));
            float _sub_81;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(wp_bs_0), "f"(ts_bb_8));
#else
            float _sub_76;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(ts_s_8), "f"(ts_bb_8));
            float _sub_75;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(wp_s), "f"(_sub_76));
            float _sub_77;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(wp_bs_0), "f"(ts_bb_8));
#endif
            float ts_e_8;
#if __CUDA_ARCH__ == 1030
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8) : "f"(_sub_79), "f"(_sub_81));
#else
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8) : "f"(_sub_75), "f"(_sub_77));
#endif
            wp_s = ts_s_8;
#if __CUDA_ARCH__ == 1030
            float _add_83;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(wp_e), "f"(wp_be_0));
            float _add_82;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_add_83), "f"(ts_e_8));
            wp_e = _add_82;
#else
            float _add_79;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(wp_e), "f"(wp_be_0));
            float _add_78;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_add_79), "f"(ts_e_8));
            wp_e = _add_78;
#endif
            float wp_bs_1 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 2);
            float wp_be_1 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 2);
            float ts_s_6_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_6_1) : "f"(wp_s), "f"(wp_bs_1));
            float ts_bb_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6_1) : "f"(ts_s_6_1), "f"(wp_s));
#if __CUDA_ARCH__ == 1030
            float _sub_85;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(ts_s_6_1), "f"(ts_bb_6_1));
            float _sub_84;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(wp_s), "f"(_sub_85));
            float _sub_86;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(wp_bs_1), "f"(ts_bb_6_1));
#else
            float _sub_81;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(ts_s_6_1), "f"(ts_bb_6_1));
            float _sub_80;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(wp_s), "f"(_sub_81));
            float _sub_82;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(wp_bs_1), "f"(ts_bb_6_1));
#endif
            float ts_e_6_1;
#if __CUDA_ARCH__ == 1030
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6_1) : "f"(_sub_84), "f"(_sub_86));
#else
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6_1) : "f"(_sub_80), "f"(_sub_82));
#endif
            wp_s = ts_s_6_1;
#if __CUDA_ARCH__ == 1030
            float _add_88;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(wp_e), "f"(wp_be_1));
            float _add_87;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_add_88), "f"(ts_e_6_1));
            wp_e = _add_87;
#else
            float _add_84;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(wp_e), "f"(wp_be_1));
            float _add_83;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_add_84), "f"(ts_e_6_1));
            wp_e = _add_83;
#endif
            float wp_bs_2 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 1);
            float wp_be_2 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 1);
            float ts_s_7_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7_1) : "f"(wp_s), "f"(wp_bs_2));
            float ts_bb_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7_1) : "f"(ts_s_7_1), "f"(wp_s));
#if __CUDA_ARCH__ == 1030
            float _sub_90;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(ts_s_7_1), "f"(ts_bb_7_1));
            float _sub_89;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_89) : "f"(wp_s), "f"(_sub_90));
            float _sub_91;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91) : "f"(wp_bs_2), "f"(ts_bb_7_1));
#else
            float _sub_86;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(ts_s_7_1), "f"(ts_bb_7_1));
            float _sub_85;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(wp_s), "f"(_sub_86));
            float _sub_87;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(wp_bs_2), "f"(ts_bb_7_1));
#endif
            float ts_e_7_1;
#if __CUDA_ARCH__ == 1030
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7_1) : "f"(_sub_89), "f"(_sub_91));
#else
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7_1) : "f"(_sub_85), "f"(_sub_87));
#endif
            wp_s = ts_s_7_1;
#if __CUDA_ARCH__ == 1030
            float _add_93;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(wp_e), "f"(wp_be_2));
            float _add_92;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_add_93), "f"(ts_e_7_1));
            wp_e = _add_92;
#else
            float _add_89;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(wp_e), "f"(wp_be_2));
            float _add_88;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_89), "f"(ts_e_7_1));
            wp_e = _add_88;
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
                                float ts_s_8_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8_1) : "f"(xp_fs), "f"(wtab[xp_base + wi_1 * 20 + c_2]));
                                float ts_bb_8_1;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8_1) : "f"(ts_s_8_1), "f"(xp_fs));
#if __CUDA_ARCH__ == 1030
                                float _sub_95;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(ts_s_8_1), "f"(ts_bb_8_1));
                                float _sub_94;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(xp_fs), "f"(_sub_95));
                                float _sub_96;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(wtab[xp_base + wi_1 * 20 + c_2]), "f"(ts_bb_8_1));
#else
                                float _sub_91;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91) : "f"(ts_s_8_1), "f"(ts_bb_8_1));
                                float _sub_90;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(xp_fs), "f"(_sub_91));
                                float _sub_92;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_92) : "f"(wtab[xp_base + wi_1 * 20 + c_2]), "f"(ts_bb_8_1));
#endif
                                float ts_e_8_1;
#if __CUDA_ARCH__ == 1030
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8_1) : "f"(_sub_94), "f"(_sub_96));
#else
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8_1) : "f"(_sub_90), "f"(_sub_92));
#endif
                                xp_fs = ts_s_8_1;
#if __CUDA_ARCH__ == 1030
                                float _add_98;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(xp_fe), "f"(wtab[xp_base + wi_1 * 20 + 4 + c_2]));
                                float _add_97;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_add_98), "f"(ts_e_8_1));
                                xp_fe = _add_97;
#else
                                float _add_94;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(xp_fe), "f"(wtab[xp_base + wi_1 * 20 + 4 + c_2]));
                                float _add_93;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_add_94), "f"(ts_e_8_1));
                                xp_fe = _add_93;
#endif
                            }
                            xp_s[c_2] = xp_fs;
                            xp_e[c_2] = xp_fe;
                        }
#if __CUDA_ARCH__ == 1030
                        uint32_t _cluster_mapa_99;
#else
                        uint32_t _cluster_mapa_95;
#endif
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                            : "=r"(_cluster_mapa_99) : "r"(xp_to + (unsigned int)(16 * (xp_job + 1))), "r"(xp_dst));
                        uint32_t _mbar_peer_100;
#else
                            : "=r"(_cluster_mapa_95) : "r"(xp_to + (unsigned int)(16 * (xp_job + 1))), "r"(xp_dst));
                        uint32_t _mbar_peer_96;
#endif
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                            : "=r"(_mbar_peer_100) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
#else
                            : "=r"(_mbar_peer_96) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
#endif
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
#if __CUDA_ARCH__ == 1030
                            :: "r"(_cluster_mapa_99), "r"(__float_as_uint(xp_e[0])), "r"(__float_as_uint(xp_e[1])), "r"(__float_as_uint(xp_e[2])), "r"(__float_as_uint(xp_e[3])), "r"(_mbar_peer_100) : "memory");
#else
                            :: "r"(_cluster_mapa_95), "r"(__float_as_uint(xp_e[0])), "r"(__float_as_uint(xp_e[1])), "r"(__float_as_uint(xp_e[2])), "r"(__float_as_uint(xp_e[3])), "r"(_mbar_peer_96) : "memory");
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    uint32_t _cluster_mapa_101;
#else
                    uint32_t _cluster_mapa_97;
#endif
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                        : "=r"(_cluster_mapa_101) : "r"(xp_to + (unsigned int)(16 * xp_job)), "r"(xp_dst));
                    uint32_t _mbar_peer_102;
#else
                        : "=r"(_cluster_mapa_97) : "r"(xp_to + (unsigned int)(16 * xp_job)), "r"(xp_dst));
                    uint32_t _mbar_peer_98;
#endif
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
#if __CUDA_ARCH__ == 1030
                        : "=r"(_mbar_peer_102) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
#else
                        : "=r"(_mbar_peer_98) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xp_dst));
#endif
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
#if __CUDA_ARCH__ == 1030
                        :: "r"(_cluster_mapa_101), "r"(__float_as_uint(xp_s[0])), "r"(__float_as_uint(xp_s[1])), "r"(__float_as_uint(xp_s[2])), "r"(__float_as_uint(xp_s[3])), "r"(_mbar_peer_102) : "memory");
#else
                        :: "r"(_cluster_mapa_97), "r"(__float_as_uint(xp_s[0])), "r"(__float_as_uint(xp_s[1])), "r"(__float_as_uint(xp_s[2])), "r"(__float_as_uint(xp_s[3])), "r"(_mbar_peer_98) : "memory");
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
                    float ts_s_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_9) : "f"(xt_ts), "f"(recv[xt_from + r_1 * 20 + 12]));
                    float ts_bb_9;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_9) : "f"(ts_s_9), "f"(xt_ts));
#if __CUDA_ARCH__ == 1030
                    float _sub_104;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(ts_s_9), "f"(ts_bb_9));
                    float _sub_103;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_103) : "f"(xt_ts), "f"(_sub_104));
                    float _sub_105;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(recv[xt_from + r_1 * 20 + 12]), "f"(ts_bb_9));
#else
                    float _sub_100;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(ts_s_9), "f"(ts_bb_9));
                    float _sub_99;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_99) : "f"(xt_ts), "f"(_sub_100));
                    float _sub_101;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_101) : "f"(recv[xt_from + r_1 * 20 + 12]), "f"(ts_bb_9));
#endif
                    float ts_e_9;
#if __CUDA_ARCH__ == 1030
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9) : "f"(_sub_103), "f"(_sub_105));
#else
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9) : "f"(_sub_99), "f"(_sub_101));
#endif
                    xt_ts = ts_s_9;
#if __CUDA_ARCH__ == 1030
                    float _add_107;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(xt_te), "f"(recv[xt_from + r_1 * 20 + 16]));
                    float _add_106;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(_add_107), "f"(ts_e_9));
                    xt_te = _add_106;
#else
                    float _add_103;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(xt_te), "f"(recv[xt_from + r_1 * 20 + 16]));
                    float _add_102;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(_add_103), "f"(ts_e_9));
                    xt_te = _add_102;
#endif
                }
#if __CUDA_ARCH__ == 1030
                float _add_108;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(xt_ts), "f"(xt_te));
                tot[3] = _add_108;
                float _fdiv_110 = __fdiv_rn(tot[0], 5120.0f);
                float _sqrt_109;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_109) : "f"(_fdiv_110 + eps));
                float sg_rh = __frcp_rn(_sqrt_109);
                float _fdiv_112 = __fdiv_rn(tot[1], 5120.0f);
                float _sqrt_111;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_111) : "f"(_fdiv_112 + eps));
                float sg_rk = __frcp_rn(_sqrt_111);
#else
                float _add_104;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(xt_ts), "f"(xt_te));
                tot[3] = _add_104;
                float _fdiv_106 = __fdiv_rn(tot[0], 5120.0f);
                float _sqrt_105;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_105) : "f"(_fdiv_106 + eps));
                float sg_rh = __frcp_rn(_sqrt_105);
                float _fdiv_108 = __fdiv_rn(tot[1], 5120.0f);
                float _sqrt_107;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_107) : "f"(_fdiv_108 + eps));
                float sg_rk = __frcp_rn(_sqrt_107);
#endif
                float sg_dot;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sg_dot) : "f"(mul_f32_rn_noftz(tot[3], mul_f32_rn_noftz(sg_rh, sg_rk))), "f"(0.013975425f));
#if __CUDA_ARCH__ == 1030
                float _abs_114;
                asm("abs.f32 %0, %1;" : "=f"(_abs_114) : "f"(sg_dot));
                float _max_113;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_113) : "f"(_abs_114), "f"(clamp));
#else
                float _abs_110;
                asm("abs.f32 %0, %1;" : "=f"(_abs_110) : "f"(sg_dot));
                float _max_109;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_109) : "f"(_abs_110), "f"(clamp));
#endif
                float sg_mag;
#if __CUDA_ARCH__ == 1030
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_113));
#else
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_109));
#endif
                float sg_u = __uint_as_float(__as_u32(sg_mag) & 2147483647 | __as_u32(sg_dot) & 2147483648u);
#if __CUDA_ARCH__ == 1030
                float _neg_116;
                asm("neg.f32 %0, %1;" : "=f"(_neg_116) : "f"(sg_u));
                float _exp_115 = __nv_expf(_neg_116);
                float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_115);
#else
                float _neg_112;
                asm("neg.f32 %0, %1;" : "=f"(_neg_112) : "f"(sg_u));
                float _exp_111 = __nv_expf(_neg_112);
                float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_111);
#endif
                float sd_mag;
                asm("abs.f32 %0, %1;" : "=f"(sd_mag) : "f"(sg_dot));
                float sd_du = 0.0f;
                if (sd_mag > clamp) {
#if __CUDA_ARCH__ == 1030
                    float _sqrt_118;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_118) : "f"(sd_mag));
                    float _fdiv_117 = __fdiv_rn(0.5f, _sqrt_118);
                    sd_du = _fdiv_117;
#else
                    float _sqrt_114;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_114) : "f"(sd_mag));
                    float _fdiv_113 = __fdiv_rn(0.5f, _sqrt_114);
                    sd_du = _fdiv_113;
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
                float _neg_119;
                asm("neg.f32 %0, %1;" : "=f"(_neg_119) : "f"(tg_sh));
                float _neg_120;
                asm("neg.f32 %0, %1;" : "=f"(_neg_120) : "f"(tg_sh));
                float2 tg_h2 = make_float2(_neg_119, _neg_120);
                float _neg_121;
                asm("neg.f32 %0, %1;" : "=f"(_neg_121) : "f"(tg_sk));
                float _neg_122;
                asm("neg.f32 %0, %1;" : "=f"(_neg_122) : "f"(tg_sk));
                float2 tg_k2 = make_float2(_neg_121, _neg_122);
#else
                float _neg_115;
                asm("neg.f32 %0, %1;" : "=f"(_neg_115) : "f"(tg_sh));
                float _neg_116;
                asm("neg.f32 %0, %1;" : "=f"(_neg_116) : "f"(tg_sh));
                float2 tg_h2 = make_float2(_neg_115, _neg_116);
                float _neg_117;
                asm("neg.f32 %0, %1;" : "=f"(_neg_117) : "f"(tg_sk));
                float _neg_118;
                asm("neg.f32 %0, %1;" : "=f"(_neg_118) : "f"(tg_sk));
                float2 tg_k2 = make_float2(_neg_117, _neg_118);
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
                for (int m_6 = 0; m_6 < 4; m_6++) {
                    float2 tg_hv = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_23 + m_6]));
                    float2 tg_kv = cake_bf16x2_to_f32x2(__as_bf16x2(kw[4 * s_23 + m_6]));
                    float2 tg_gv = cake_bf16x2_to_f32x2(__as_bf16x2(gw[4 * s_23 + m_6]));
                    float2 tg_wv = make_float2(wf[8 * s_23 + 2 * m_6], wf[8 * s_23 + 2 * m_6 + 1]);
#else
                for (int m_4 = 0; m_4 < 4; m_4++) {
                    float2 tg_hv = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_17 + m_4]));
                    float2 tg_kv = cake_bf16x2_to_f32x2(__as_bf16x2(kw[4 * s_17 + m_4]));
                    float2 tg_gv = cake_bf16x2_to_f32x2(__as_bf16x2(gw[4 * s_17 + m_4]));
                    float2 tg_wv = make_float2(wf[8 * s_17 + 2 * m_4], wf[8 * s_17 + 2 * m_4 + 1]);
#endif
                    float2 tg_ox = fma_f32x2_rn_noftz(tg_c2, fma_f32x2_rn_noftz(tg_h2, tg_hv, mul_f32x2_noftz(tg_wv, tg_kv)), tg_gv);
#if __CUDA_ARCH__ == 1030
                    float2 _f32x2_mul_b_123 = fma_f32x2_rn_noftz(tg_k2, tg_kv, mul_f32x2_noftz(tg_wv, tg_hv));
#else
                    float2 _f32x2_mul_b_119 = fma_f32x2_rn_noftz(tg_k2, tg_kv, mul_f32x2_noftz(tg_wv, tg_hv));
#endif
                    float2 tg_ok;
#if __CUDA_ARCH__ == 1030
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_ok) : "l"(*(const unsigned long long*)&tg_c2), "l"(*(const unsigned long long*)&_f32x2_mul_b_123));
                    float2 tg_ov = fma_f32x2_rn_noftz(tg_g2, tg_gv, make_float2(tg_dv[2 * m_6], tg_dv[2 * m_6 + 1]));
#else
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_ok) : "l"(*(const unsigned long long*)&tg_c2), "l"(*(const unsigned long long*)&_f32x2_mul_b_119));
                    float2 tg_ov = fma_f32x2_rn_noftz(tg_g2, tg_gv, make_float2(tg_dv[2 * m_4], tg_dv[2 * m_4 + 1]));
#endif
                    float2 tg_hk;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tg_hk) : "l"(*(const unsigned long long*)&tg_hv), "l"(*(const unsigned long long*)&tg_kv));
#if __CUDA_ARCH__ == 1030
                    float2 tg_ow = fma_f32x2_rn_noftz(tg_c2, tg_hk, make_float2(dwa[8 * s_23 + 2 * m_6], dwa[8 * s_23 + 2 * m_6 + 1]));
                    tg_dx[2 * m_6] = tg_ox.x;
                    tg_dk[2 * m_6] = tg_ok.x;
                    tg_dv[2 * m_6] = tg_ov.x;
                    dwa[8 * s_23 + 2 * m_6] = tg_ow.x;
                    tg_dx[2 * m_6 + 1] = tg_ox.y;
                    tg_dk[2 * m_6 + 1] = tg_ok.y;
                    tg_dv[2 * m_6 + 1] = tg_ov.y;
                    dwa[8 * s_23 + 2 * m_6 + 1] = tg_ow.y;
#else
                    float2 tg_ow = fma_f32x2_rn_noftz(tg_c2, tg_hk, make_float2(dwa[8 * s_17 + 2 * m_4], dwa[8 * s_17 + 2 * m_4 + 1]));
                    tg_dx[2 * m_4] = tg_ox.x;
                    tg_dk[2 * m_4] = tg_ok.x;
                    tg_dv[2 * m_4] = tg_ov.x;
                    dwa[8 * s_17 + 2 * m_4] = tg_ow.x;
                    tg_dx[2 * m_4 + 1] = tg_ox.y;
                    tg_dk[2 * m_4 + 1] = tg_ok.y;
                    tg_dv[2 * m_4 + 1] = tg_ov.y;
                    dwa[8 * s_17 + 2 * m_4 + 1] = tg_ow.y;
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
                        const uint4* _vptr_124 = reinterpret_cast<const uint4*>(dout + (ls_at_3 + (long long)(s_24 * 5120)) + 0);
                        uint4* _vdst_124 = reinterpret_cast<uint4*>(&gn[4 * s_24]);
#else
                        const uint4* _vptr_120 = reinterpret_cast<const uint4*>(dout + (ls_at_3 + (long long)(s_18 * 5120)) + 0);
                        uint4* _vdst_120 = reinterpret_cast<uint4*>(&gn[4 * s_18]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_124[_blk] = _vptr_124[_blk];
#else
                            _vdst_120[_blk] = _vptr_120[_blk];
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
                        const uint4* _vptr_125 = reinterpret_cast<const uint4*>(x + (ls_at_4 + (long long)(s_25 * 5120)) + 0);
                        uint4* _vdst_125 = reinterpret_cast<uint4*>(&xn[4 * s_25]);
#else
                        const uint4* _vptr_121 = reinterpret_cast<const uint4*>(x + (ls_at_4 + (long long)(s_19 * 5120)) + 0);
                        uint4* _vdst_121 = reinterpret_cast<uint4*>(&xn[4 * s_19]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_125[_blk] = _vptr_125[_blk];
#else
                            _vdst_121[_blk] = _vptr_121[_blk];
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
                        const uint4* _vptr_126 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_26 * key_stride_p1) + 0);
                        uint4* _vdst_126 = reinterpret_cast<uint4*>(&kn[4 * s_26]);
#else
                        const uint4* _vptr_122 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_20 * key_stride_p1) + 0);
                        uint4* _vdst_122 = reinterpret_cast<uint4*>(&kn[4 * s_20]);
#endif
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                            _vdst_126[_blk] = _vptr_126[_blk];
#else
                            _vdst_122[_blk] = _vptr_122[_blk];
#endif
                        }
                    }
                }
                {
#if __CUDA_ARCH__ == 1030
                    const uint4* _vptr_127 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_127 = reinterpret_cast<uint4*>(&vn[0]);
#else
                    const uint4* _vptr_123 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_123 = reinterpret_cast<uint4*>(&vn[0]);
#endif
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
#if __CUDA_ARCH__ == 1030
                        _vdst_127[_blk] = _vptr_127[_blk];
#else
                        _vdst_123[_blk] = _vptr_123[_blk];
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
