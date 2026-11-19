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
kernel_cake_engram_gate_train_1857a748ab0245abafbe(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, int* __restrict__ q_weight, int* __restrict__ k_weight, bool* __restrict__ mask, __nv_bfloat16* __restrict__ out, int T, float eps, float clamp, int wmode, long long q_weight_size_p0, long long q_weight_size_p1, long long k_weight_size_p0, long long k_weight_size_p1, long long key_stride_p0, long long key_stride_p1, long long value_stride_p0)
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
    unsigned int raw_q[32];
    unsigned int raw_k[32];
    if (wmode == 0) {
        {
            #pragma unroll
            for (int s_2 = 0; s_2 < 4; s_2++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    {
                        uint32_t _uv4_3_0;
                        uint32_t _uv4_3_1;
                        uint32_t _uv4_3_2;
                        uint32_t _uv4_3_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_3_0), "=r"(_uv4_3_1), "=r"(_uv4_3_2), "=r"(_uv4_3_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s_2 * 5120 + col + 4 * half)) + (0))) : "memory");
                        raw_q[8 * s_2 + 4 * half + 0] = _uv4_3_0;
                        raw_q[8 * s_2 + 4 * half + 1] = _uv4_3_1;
                        raw_q[8 * s_2 + 4 * half + 2] = _uv4_3_2;
                        raw_q[8 * s_2 + 4 * half + 3] = _uv4_3_3;
                    }
                }
            }
        }
        {
            #pragma unroll
            for (int s_3 = 0; s_3 < 4; s_3++) {
                #pragma unroll
                for (int half_1 = 0; half_1 < 2; half_1++) {
                    {
                        uint32_t _uv4_4_0;
                        uint32_t _uv4_4_1;
                        uint32_t _uv4_4_2;
                        uint32_t _uv4_4_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_4_0), "=r"(_uv4_4_1), "=r"(_uv4_4_2), "=r"(_uv4_4_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_3 * 5120 + col + 4 * half_1)) + (0))) : "memory");
                        raw_k[8 * s_3 + 4 * half_1 + 0] = _uv4_4_0;
                        raw_k[8 * s_3 + 4 * half_1 + 1] = _uv4_4_1;
                        raw_k[8 * s_3 + 4 * half_1 + 2] = _uv4_4_2;
                        raw_k[8 * s_3 + 4 * half_1 + 3] = _uv4_4_3;
                    }
                }
            }
        }
    } else if (wmode == 1) {
        {
            #pragma unroll
            for (int s_4 = 0; s_4 < 4; s_4++) {
                {
                    uint32_t _uv4_5_0;
                    uint32_t _uv4_5_1;
                    uint32_t _uv4_5_2;
                    uint32_t _uv4_5_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_5_0), "=r"(_uv4_5_1), "=r"(_uv4_5_2), "=r"(_uv4_5_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s_4 * 2560 + (col >> 1))) + (0))) : "memory");
                    raw_q[4 * s_4 + 0] = _uv4_5_0;
                    raw_q[4 * s_4 + 1] = _uv4_5_1;
                    raw_q[4 * s_4 + 2] = _uv4_5_2;
                    raw_q[4 * s_4 + 3] = _uv4_5_3;
                }
            }
        }
        {
            #pragma unroll
            for (int s_5 = 0; s_5 < 4; s_5++) {
                #pragma unroll
                for (int half_2 = 0; half_2 < 2; half_2++) {
                    {
                        uint32_t _uv4_6_0;
                        uint32_t _uv4_6_1;
                        uint32_t _uv4_6_2;
                        uint32_t _uv4_6_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_6_0), "=r"(_uv4_6_1), "=r"(_uv4_6_2), "=r"(_uv4_6_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_5 * 5120 + col + 4 * half_2)) + (0))) : "memory");
                        raw_k[8 * s_5 + 4 * half_2 + 0] = _uv4_6_0;
                        raw_k[8 * s_5 + 4 * half_2 + 1] = _uv4_6_1;
                        raw_k[8 * s_5 + 4 * half_2 + 2] = _uv4_6_2;
                        raw_k[8 * s_5 + 4 * half_2 + 3] = _uv4_6_3;
                    }
                }
            }
        }
    } else {
        if (wmode == 2) {
            {
                #pragma unroll
                for (int s_6 = 0; s_6 < 4; s_6++) {
                    #pragma unroll
                    for (int half_3 = 0; half_3 < 2; half_3++) {
                        {
                            uint32_t _uv4_7_0;
                            uint32_t _uv4_7_1;
                            uint32_t _uv4_7_2;
                            uint32_t _uv4_7_3;
                            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_uv4_7_0), "=r"(_uv4_7_1), "=r"(_uv4_7_2), "=r"(_uv4_7_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s_6 * 5120 + col + 4 * half_3)) + (0))) : "memory");
                            raw_q[8 * s_6 + 4 * half_3 + 0] = _uv4_7_0;
                            raw_q[8 * s_6 + 4 * half_3 + 1] = _uv4_7_1;
                            raw_q[8 * s_6 + 4 * half_3 + 2] = _uv4_7_2;
                            raw_q[8 * s_6 + 4 * half_3 + 3] = _uv4_7_3;
                        }
                    }
                }
            }
            {
                #pragma unroll
                for (int s_7 = 0; s_7 < 4; s_7++) {
                    {
                        uint32_t _uv4_8_0;
                        uint32_t _uv4_8_1;
                        uint32_t _uv4_8_2;
                        uint32_t _uv4_8_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_8_0), "=r"(_uv4_8_1), "=r"(_uv4_8_2), "=r"(_uv4_8_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_7 * 2560 + (col >> 1))) + (0))) : "memory");
                        raw_k[4 * s_7 + 0] = _uv4_8_0;
                        raw_k[4 * s_7 + 1] = _uv4_8_1;
                        raw_k[4 * s_7 + 2] = _uv4_8_2;
                        raw_k[4 * s_7 + 3] = _uv4_8_3;
                    }
                }
            }
        } else {
            {
                #pragma unroll
                for (int s_8 = 0; s_8 < 4; s_8++) {
                    {
                        uint32_t _uv4_9_0;
                        uint32_t _uv4_9_1;
                        uint32_t _uv4_9_2;
                        uint32_t _uv4_9_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_9_0), "=r"(_uv4_9_1), "=r"(_uv4_9_2), "=r"(_uv4_9_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(q_weight + (s_8 * 2560 + (col >> 1))) + (0))) : "memory");
                        raw_q[4 * s_8 + 0] = _uv4_9_0;
                        raw_q[4 * s_8 + 1] = _uv4_9_1;
                        raw_q[4 * s_8 + 2] = _uv4_9_2;
                        raw_q[4 * s_8 + 3] = _uv4_9_3;
                    }
                }
            }
            {
                #pragma unroll
                for (int s_9 = 0; s_9 < 4; s_9++) {
                    {
                        uint32_t _uv4_10_0;
                        uint32_t _uv4_10_1;
                        uint32_t _uv4_10_2;
                        uint32_t _uv4_10_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_10_0), "=r"(_uv4_10_1), "=r"(_uv4_10_2), "=r"(_uv4_10_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(k_weight + (s_9 * 2560 + (col >> 1))) + (0))) : "memory");
                        raw_k[4 * s_9 + 0] = _uv4_10_0;
                        raw_k[4 * s_9 + 1] = _uv4_10_1;
                        raw_k[4 * s_9 + 2] = _uv4_10_2;
                        raw_k[4 * s_9 + 3] = _uv4_10_3;
                    }
                }
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
    if (wmode == 0) {
        float wp_qf[32];
        {
            #pragma unroll
            for (int i_1 = 0; i_1 < 32; i_1++) {
                wp_qf[i_1] = __uint_as_float(raw_q[i_1]);
            }
        }
        {
            #pragma unroll
            for (int i_2 = 0; i_2 < 32; i_2++) {
                wf[i_2] = __uint_as_float(raw_k[i_2]);
            }
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 32; i_3++) {
            wf[i_3] = mul_f32_rn_noftz(wp_qf[i_3], wf[i_3]);
        }
    } else if (wmode == 1) {
        float wp_qf_1[32];
        {
            #pragma unroll
            for (int s_10 = 0; s_10 < 4; s_10++) {
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    float2 ww_v = cake_bf16x2_to_f32x2(__as_bf16x2(raw_q[4 * s_10 + m]));
                    wp_qf_1[8 * s_10 + 2 * m] = ww_v.x;
                    wp_qf_1[8 * s_10 + 2 * m + 1] = ww_v.y;
                }
            }
        }
        {
            #pragma unroll
            for (int i_4 = 0; i_4 < 32; i_4++) {
                wf[i_4] = __uint_as_float(raw_k[i_4]);
            }
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 32; i_5++) {
            wf[i_5] = mul_f32_rn_noftz(wp_qf_1[i_5], wf[i_5]);
        }
    } else {
        if (wmode == 2) {
            float wp_qf_2[32];
            {
                #pragma unroll
                for (int i_6 = 0; i_6 < 32; i_6++) {
                    wp_qf_2[i_6] = __uint_as_float(raw_q[i_6]);
                }
            }
            {
                #pragma unroll
                for (int s_11 = 0; s_11 < 4; s_11++) {
                    #pragma unroll
                    for (int m_1 = 0; m_1 < 4; m_1++) {
                        float2 ww_v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(raw_k[4 * s_11 + m_1]));
                        wf[8 * s_11 + 2 * m_1] = ww_v_1.x;
                        wf[8 * s_11 + 2 * m_1 + 1] = ww_v_1.y;
                    }
                }
            }
            #pragma unroll
            for (int i_7 = 0; i_7 < 32; i_7++) {
                wf[i_7] = mul_f32_rn_noftz(wp_qf_2[i_7], wf[i_7]);
            }
        } else {
            float wp_qf_3[32];
            {
                #pragma unroll
                for (int s_12 = 0; s_12 < 4; s_12++) {
                    #pragma unroll
                    for (int m_2 = 0; m_2 < 4; m_2++) {
                        float2 ww_v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(raw_q[4 * s_12 + m_2]));
                        wp_qf_3[8 * s_12 + 2 * m_2] = ww_v_2.x;
                        wp_qf_3[8 * s_12 + 2 * m_2 + 1] = ww_v_2.y;
                    }
                }
            }
            {
                #pragma unroll
                for (int s_13 = 0; s_13 < 4; s_13++) {
                    #pragma unroll
                    for (int m_3 = 0; m_3 < 4; m_3++) {
                        float2 ww_v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(raw_k[4 * s_13 + m_3]));
                        wf[8 * s_13 + 2 * m_3] = ww_v_3.x;
                        wf[8 * s_13 + 2 * m_3 + 1] = ww_v_3.y;
                    }
                }
            }
            #pragma unroll
            for (int i_8 = 0; i_8 < 32; i_8++) {
                wf[i_8] = mul_f32_rn_noftz(wp_qf_3[i_8], wf[i_8]);
            }
        }
    }
    #pragma unroll
    for (int s_14 = 0; s_14 < 4; s_14++) {
        #pragma unroll
        for (int half_4 = 0; half_4 < 2; half_4++) {
            unsigned int fw_at = (unsigned int)(16 * ((2 * s_14 + half_4) * 160 + tid));
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(wslots_addr + fw_at), "f"(wf[8 * s_14 + 4 * half_4]), "f"(wf[8 * s_14 + 4 * half_4 + 1]), "f"(wf[8 * s_14 + 4 * half_4 + 2]), "f"(wf[8 * s_14 + 4 * half_4 + 3]) : "memory");
        }
    }
    #pragma unroll 1
    for (int t = first; t < T; t += step) {
        long long to_base = (long long)t * 20480ll + (long long)col;
        if (keep != 0) {
            #pragma unroll
            for (int s_15 = 0; s_15 < 4; s_15++) {
                #pragma unroll
                for (int half_5 = 0; half_5 < 2; half_5++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wf[8 * s_15 + 4 * half_5])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_15 + 4 * half_5) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_15 + 4 * half_5) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(8 * s_15 + 4 * half_5) + 3]))
                        : "r"(wslots_addr + (unsigned int)(16 * ((2 * s_15 + half_5) * 160 + tid))));
                }
                float ts_h = 0.0f;
                float ts_k = 0.0f;
                #pragma unroll
                for (int m_4 = 0; m_4 < 4; m_4++) {
                    float _dot_11;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_11) : "r"(xw[4 * s_15 + m_4]), "r"(xw[4 * s_15 + m_4]), "f"(ts_h));
                    ts_h = _dot_11;
                    float _dot_12;
                    asm(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_12) : "r"(kw[4 * s_15 + m_4]), "r"(kw[4 * s_15 + m_4]), "f"(ts_k));
                    ts_k = _dot_12;
                }
                plain[s_15] = ts_h;
                plain[4 + s_15] = ts_k;
                float2 ts_f = make_float2(0.0f, 0.0f);
                #pragma unroll
                for (int m_5 = 0; m_5 < 4; m_5++) {
                    float2 _f32x2_mul_a_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_15 + m_5]));
                    float2 _f32x2_mul_b_14 = cake_bf16x2_to_f32x2(__as_bf16x2(kw[4 * s_15 + m_5]));
                    float2 ts_hk2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ts_hk2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
                    ts_f = fma_f32x2_rn_noftz(make_float2(wf[8 * s_15 + 2 * m_5], wf[8 * s_15 + 2 * m_5 + 1]), ts_hk2, ts_f);
                }
                float _add_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(ts_f.x), "f"(ts_f.y));
                plain[8 + s_15] = _add_15;
            }
            int pn_tok = t + step;
            if (pn_tok < T) {
                long long ls_at_0 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
                for (int s_16 = 0; s_16 < 4; s_16++) {
                    {
                        const uint4* _vptr_16 = reinterpret_cast<const uint4*>(x + (ls_at_0 + (long long)(s_16 * 5120)) + 0);
                        uint4* _vdst_16 = reinterpret_cast<uint4*>(&xn[4 * s_16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_16[_blk] = _vptr_16[_blk];
                        }
                    }
                }
                long long lk_at_0 = (long long)pn_tok * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_17 = 0; s_17 < 4; s_17++) {
                    {
                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_17 * key_stride_p1) + 0);
                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&kn[4 * s_17]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_17[_blk] = _vptr_17[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_18 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_18[_blk] = _vptr_18[_blk];
                    }
                }
#if __CUDA_ARCH__ == 1070
                int tw_keep = 1;
#else
                int tk_keep_0 = 1;
#endif
                if ((unsigned long long)mask != 0ull) {
#if __CUDA_ARCH__ == 1070
                    tw_keep = (int)*reinterpret_cast<uint8_t*>(mask + pn_tok);
#else
                    if (!mask[pn_tok]) {
                        tk_keep_0 = 0;
                    }
#endif
                }
#if __CUDA_ARCH__ == 1070
                keep_n = tw_keep;
#else
                keep_n = tk_keep_0;
#endif
            }
            int xl_slot = xk % 2;
            int xl_row = xl_slot * 100 + warp * 12;
            #pragma unroll
            for (int i_0 = 0; i_0 < 8; i_0++) {
                int ws_upper_0 = lane & 16;
                float ws_keep_0 = ((ws_upper_0 != 0) ? plain[8 + i_0] : plain[i_0]);
                float ws_send_0 = ((ws_upper_0 != 0) ? plain[i_0] : plain[8 + i_0]);
                float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_19;
            }
            #pragma unroll
            for (int i_1_1 = 0; i_1_1 < 4; i_1_1++) {
                int ws_upper_1 = lane & 8;
                float ws_keep_1 = ((ws_upper_1 != 0) ? plain[4 + i_1_1] : plain[i_1_1]);
                float ws_send_1 = ((ws_upper_1 != 0) ? plain[i_1_1] : plain[4 + i_1_1]);
                float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1_1] = ws_keep_1 + _shfl_xor_sync_20;
            }
            #pragma unroll
            for (int i_2_1 = 0; i_2_1 < 2; i_2_1++) {
                int ws_upper_2 = lane & 4;
                float ws_keep_2 = ((ws_upper_2 != 0) ? plain[2 + i_2_1] : plain[i_2_1]);
                float ws_send_2 = ((ws_upper_2 != 0) ? plain[i_2_1] : plain[2 + i_2_1]);
                float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2_1] = ws_keep_2 + _shfl_xor_sync_21;
            }
            #pragma unroll
            for (int i_3_1 = 0; i_3_1 < 1; i_3_1++) {
                int ws_upper_3 = lane & 2;
                float ws_keep_3 = ((ws_upper_3 != 0) ? plain[1 + i_3_1] : plain[i_3_1]);
                float ws_send_3 = ((ws_upper_3 != 0) ? plain[i_3_1] : plain[1 + i_3_1]);
                float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, ws_send_3, 2);
                plain[i_3_1] = ws_keep_3 + _shfl_xor_sync_22;
            }
            float ws_own = plain[0];
            float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_23;
            float xl_word = ws_own;
            if ((lane & 1) == 0) {
                if (lane >> 1 < 12) {
                    wtab[xl_row + (lane >> 1)] = xl_word;
                }
            }
            __syncthreads();
            if (warp == 0) {
                if (lane < 12) {
                    int xl_job = lane % 3;
                    int xl_dst = lane / 3;
                    int xl_base = xl_slot * 100 + 4 * xl_job;
                    unsigned int xl_to = recv_addr + (unsigned int)(4 * (xl_slot * 80 + rank * 12));
                    float xl_s[4];
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        xl_s[c] = wtab[xl_base + c];
                    }
                    if (xl_job < 3) {
                        #pragma unroll
                        for (int wi = 1; wi < 5; wi++) {
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 4; c_1++) {
                                xl_s[c_1] = xl_s[c_1] + wtab[xl_base + wi * 12 + c_1];
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
                                float ts_s;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(xl_fs), "f"(wtab[xl_base + wi_1 * 12 + c_2]));
                                float ts_bb;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(xl_fs));
                                float _sub_25;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(ts_s), "f"(ts_bb));
                                float _sub_24;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(xl_fs), "f"(_sub_25));
                                float _sub_26;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(wtab[xl_base + wi_1 * 12 + c_2]), "f"(ts_bb));
                                float ts_e;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_24), "f"(_sub_26));
                                xl_fs = ts_s;
                                float _add_28;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(xl_fe), "f"(wtab[xl_base + wi_1 * 12 + 4 + c_2]));
                                float _add_27;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_add_28), "f"(ts_e));
                                xl_fe = _add_27;
                            }
                            xl_s[c_2] = xl_fs;
                            xl_e[c_2] = xl_fe;
                        }
                        uint32_t _cluster_mapa_29;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_29) : "r"(xl_to + (unsigned int)(16 * (xl_job + 1))), "r"(xl_dst));
                        uint32_t _mbar_peer_30;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_30) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xl_dst));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_29), "r"(__float_as_uint(xl_e[0])), "r"(__float_as_uint(xl_e[1])), "r"(__float_as_uint(xl_e[2])), "r"(__float_as_uint(xl_e[3])), "r"(_mbar_peer_30) : "memory");
                    }
                    uint32_t _cluster_mapa_31;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_31) : "r"(xl_to + (unsigned int)(16 * xl_job)), "r"(xl_dst));
                    uint32_t _mbar_peer_32;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_32) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xl_dst));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_31), "r"(__float_as_uint(xl_s[0])), "r"(__float_as_uint(xl_s[1])), "r"(__float_as_uint(xl_s[2])), "r"(__float_as_uint(xl_s[3])), "r"(_mbar_peer_32) : "memory");
                }
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(xbar_addr + ((int)((unsigned int)xk % 2)) * 8, 192);
                }
                int _stage_xbar = (int)((unsigned int)xk % 2);
                int _parity_xbar = (int)((unsigned int)xk / 2 % 2);
                asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(xbar_addr + (_stage_xbar) * 8), "r"(uint32_t(_parity_xbar)), "r"(uint32_t(10000000)) : "memory");
                asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
                int xl_from = xl_slot * 80 + (lane & 3);
                float xl_tot[3];
                #pragma unroll
                for (int q = 0; q < 3; q++) {
                    float xl_t = recv[xl_from + 4 * q];
                    #pragma unroll
                    for (int r = 1; r < 4; r++) {
                        xl_t = xl_t + recv[xl_from + r * 12 + 4 * q];
                    }
                    xl_tot[q] = xl_t;
                }
                float _fdiv_34 = __fdiv_rn(xl_tot[0], 5120.0f);
                float _sqrt_33;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_33) : "f"(_fdiv_34 + eps));
                float sg_rh = __frcp_rn(_sqrt_33);
                float _fdiv_36 = __fdiv_rn(xl_tot[1], 5120.0f);
                float _sqrt_35;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_35) : "f"(_fdiv_36 + eps));
                float sg_rk = __frcp_rn(_sqrt_35);
                float sg_dot;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sg_dot) : "f"(mul_f32_rn_noftz(xl_tot[2], mul_f32_rn_noftz(sg_rh, sg_rk))), "f"(0.013975425f));
                float _abs_38;
                asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(sg_dot));
                float _max_37;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(_abs_38), "f"(clamp));
                float sg_mag;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_37));
                float sg_u = __uint_as_float(__as_u32(sg_mag) & 2147483647 | __as_u32(sg_dot) & 2147483648u);
                float _neg_40;
                asm("neg.f32 %0, %1;" : "=f"(_neg_40) : "f"(sg_u));
                float _exp_39 = __nv_expf(_neg_40);
                float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_39);
                if (lane < 4) {
                    gtab[lane] = sg_gate;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int s_18 = 0; s_18 < 4; s_18++) {
                gates[s_18] = gtab[s_18];
            }
            xk += 1;
            #pragma unroll
            for (int s_19 = 0; s_19 < 4; s_19++) {
                float2 to_g2 = make_float2(gates[s_19], gates[s_19]);
                float to_out[8];
                #pragma unroll
                for (int m_6 = 0; m_6 < 4; m_6++) {
                    float2 to_o2 = fma_f32x2_rn_noftz(to_g2, cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_6])), cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_19 + m_6])));
                    to_out[2 * m_6] = to_o2.x;
                    to_out[2 * m_6 + 1] = to_o2.y;
                }
                unsigned int to_packed[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(to_out[_lp*2 + 0], to_out[_lp*2+1 + 0]));
                    to_packed[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (to_base + (long long)(s_19 * 5120)))[0] = reinterpret_cast<int4*>(to_packed)[0];
            }
        } else {
            int pn_tok_1 = t + step;
            if (pn_tok_1 < T) {
                long long ls_at_1 = (long long)pn_tok_1 * 20480ll + (long long)col;
                #pragma unroll
                for (int s_20 = 0; s_20 < 4; s_20++) {
                    {
                        const uint4* _vptr_41 = reinterpret_cast<const uint4*>(x + (ls_at_1 + (long long)(s_20 * 5120)) + 0);
                        uint4* _vdst_41 = reinterpret_cast<uint4*>(&xn[4 * s_20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_41[_blk] = _vptr_41[_blk];
                        }
                    }
                }
                long long lk_at_1 = (long long)pn_tok_1 * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_21 = 0; s_21 < 4; s_21++) {
                    {
                        const uint4* _vptr_42 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_21 * key_stride_p1) + 0);
                        uint4* _vdst_42 = reinterpret_cast<uint4*>(&kn[4 * s_21]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_42[_blk] = _vptr_42[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_43 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_43 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_43[_blk] = _vptr_43[_blk];
                    }
                }
#if __CUDA_ARCH__ == 1070
                int tw_keep_1 = 1;
#else
                int tk_keep_1 = 1;
#endif
                if ((unsigned long long)mask != 0ull) {
#if __CUDA_ARCH__ == 1070
                    tw_keep_1 = (int)*reinterpret_cast<uint8_t*>(mask + pn_tok_1);
#else
                    if (!mask[pn_tok_1]) {
                        tk_keep_1 = 0;
                    }
#endif
                }
#if __CUDA_ARCH__ == 1070
                keep_n = tw_keep_1;
#else
                keep_n = tk_keep_1;
#endif
            }
            #pragma unroll
            for (int s_22 = 0; s_22 < 4; s_22++) {
                reinterpret_cast<int4*>(out + (to_base + (long long)(s_22 * 5120)))[0] = reinterpret_cast<int4*>(xw + (4 * s_22))[0];
            }
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 16; i_9++) {
            xw[i_9] = xn[i_9];
            kw[i_9] = kn[i_9];
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 4; i_10++) {
            vw[i_10] = vn[i_10];
        }
        keep = keep_n;
    }
}

} // extern "C"
