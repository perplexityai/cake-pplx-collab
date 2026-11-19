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
kernel_cake_engram_gate_det_train_d68e1766584b775b8f3f(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, int* __restrict__ q_weight, int* __restrict__ k_weight, bool* __restrict__ mask, __nv_bfloat16* __restrict__ out, int T, float eps, float clamp, int wmode, long long q_weight_size_p0, long long q_weight_size_p1, long long k_weight_size_p0, long long k_weight_size_p1, long long key_stride_p0, long long key_stride_p1, long long value_stride_p0)
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
            for (int i = 0; i < 32; i++) {
                wp_qf[i] = __uint_as_float(raw_q[i]);
            }
        }
        {
            #pragma unroll
            for (int i_1 = 0; i_1 < 32; i_1++) {
                wf[i_1] = __uint_as_float(raw_k[i_1]);
            }
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 32; i_2++) {
            wf[i_2] = mul_f32_rn_noftz(wp_qf[i_2], wf[i_2]);
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
            for (int i_3 = 0; i_3 < 32; i_3++) {
                wf[i_3] = __uint_as_float(raw_k[i_3]);
            }
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 32; i_4++) {
            wf[i_4] = mul_f32_rn_noftz(wp_qf_1[i_4], wf[i_4]);
        }
    } else {
        if (wmode == 2) {
            float wp_qf_2[32];
            {
                #pragma unroll
                for (int i_5 = 0; i_5 < 32; i_5++) {
                    wp_qf_2[i_5] = __uint_as_float(raw_q[i_5]);
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
            for (int i_6 = 0; i_6 < 32; i_6++) {
                wf[i_6] = mul_f32_rn_noftz(wp_qf_2[i_6], wf[i_6]);
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
            for (int i_7 = 0; i_7 < 32; i_7++) {
                wf[i_7] = mul_f32_rn_noftz(wp_qf_3[i_7], wf[i_7]);
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
                float ts_s = 0.0f;
                float ts_e = 0.0f;
                float ts_hk_0[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_0[0]) : "h"(static_cast<uint16_t>((xw[4 * s_15]))), "h"(static_cast<uint16_t>((kw[4 * s_15]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_0[1]) : "h"(static_cast<uint16_t>((xw[4 * s_15]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_15]) >> 16)));
                float ts_w_0 = wf[8 * s_15];
                float ts_p_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_0) : "f"(ts_w_0), "f"(ts_hk_0[0]));
                float _neg_13;
                asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(ts_p_0));
                float ts_q_0;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_0) : "f"(ts_w_0), "f"(ts_hk_0[0]), "f"(_neg_13));
                float _add_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(0.0f), "f"(ts_p_0));
                ts_s = _add_14;
                float _add_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(0.0f), "f"(ts_q_0));
                ts_e = _add_15;
                float ts_w_1 = wf[8 * s_15 + 1];
                float ts_p_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_1) : "f"(ts_w_1), "f"(ts_hk_0[1]));
                float _neg_16;
                asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(ts_p_1));
                float ts_q_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_1) : "f"(ts_w_1), "f"(ts_hk_0[1]), "f"(_neg_16));
                float ts_s_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(ts_s), "f"(ts_p_1));
                float ts_bb;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s_0), "f"(ts_s));
                float _sub_18;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(ts_s_0), "f"(ts_bb));
                float _sub_17;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(ts_s), "f"(_sub_18));
                float _sub_19;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(ts_p_1), "f"(ts_bb));
                float ts_e_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_17), "f"(_sub_19));
                ts_s = ts_s_0;
                float _add_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(ts_q_1), "f"(ts_e_0));
                float _add_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(ts_e), "f"(_add_21));
                ts_e = _add_20;
                float ts_hk_1[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_1[0]) : "h"(static_cast<uint16_t>((xw[4 * s_15 + 1]))), "h"(static_cast<uint16_t>((kw[4 * s_15 + 1]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_1[1]) : "h"(static_cast<uint16_t>((xw[4 * s_15 + 1]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_15 + 1]) >> 16)));
                float ts_w_2 = wf[8 * s_15 + 2];
                float ts_p_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_2) : "f"(ts_w_2), "f"(ts_hk_1[0]));
                float _neg_22;
                asm("neg.f32 %0, %1;" : "=f"(_neg_22) : "f"(ts_p_2));
                float ts_q_2;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_2) : "f"(ts_w_2), "f"(ts_hk_1[0]), "f"(_neg_22));
                float ts_s_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(ts_s), "f"(ts_p_2));
                float ts_bb_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_1), "f"(ts_s));
                float _sub_24;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(ts_s_1), "f"(ts_bb_0));
                float _sub_23;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(ts_s), "f"(_sub_24));
                float _sub_25;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(ts_p_2), "f"(ts_bb_0));
                float ts_e_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_23), "f"(_sub_25));
                ts_s = ts_s_1;
                float _add_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(ts_q_2), "f"(ts_e_1));
                float _add_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(ts_e), "f"(_add_27));
                ts_e = _add_26;
                float ts_w_3 = wf[8 * s_15 + 2 + 1];
                float ts_p_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_3) : "f"(ts_w_3), "f"(ts_hk_1[1]));
                float _neg_28;
                asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(ts_p_3));
                float ts_q_3;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_3) : "f"(ts_w_3), "f"(ts_hk_1[1]), "f"(_neg_28));
                float ts_s_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(ts_s), "f"(ts_p_3));
                float ts_bb_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_2), "f"(ts_s));
                float _sub_30;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(ts_s_2), "f"(ts_bb_1));
                float _sub_29;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(ts_s), "f"(_sub_30));
                float _sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(ts_p_3), "f"(ts_bb_1));
                float ts_e_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_29), "f"(_sub_31));
                ts_s = ts_s_2;
                float _add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(ts_q_3), "f"(ts_e_2));
                float _add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(ts_e), "f"(_add_33));
                ts_e = _add_32;
                float ts_hk_2[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_2[0]) : "h"(static_cast<uint16_t>((xw[4 * s_15 + 2]))), "h"(static_cast<uint16_t>((kw[4 * s_15 + 2]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_2[1]) : "h"(static_cast<uint16_t>((xw[4 * s_15 + 2]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_15 + 2]) >> 16)));
                float ts_w_4 = wf[8 * s_15 + 4];
                float ts_p_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_4) : "f"(ts_w_4), "f"(ts_hk_2[0]));
                float _neg_34;
                asm("neg.f32 %0, %1;" : "=f"(_neg_34) : "f"(ts_p_4));
                float ts_q_4;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_4) : "f"(ts_w_4), "f"(ts_hk_2[0]), "f"(_neg_34));
                float ts_s_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3) : "f"(ts_s), "f"(ts_p_4));
                float ts_bb_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_3), "f"(ts_s));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(ts_s_3), "f"(ts_bb_2));
                float _sub_35;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(ts_s), "f"(_sub_36));
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(ts_p_4), "f"(ts_bb_2));
                float ts_e_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_35), "f"(_sub_37));
                ts_s = ts_s_3;
                float _add_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(ts_q_4), "f"(ts_e_3));
                float _add_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(ts_e), "f"(_add_39));
                ts_e = _add_38;
                float ts_w_5 = wf[8 * s_15 + 4 + 1];
                float ts_p_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_5) : "f"(ts_w_5), "f"(ts_hk_2[1]));
                float _neg_40;
                asm("neg.f32 %0, %1;" : "=f"(_neg_40) : "f"(ts_p_5));
                float ts_q_5;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_5) : "f"(ts_w_5), "f"(ts_hk_2[1]), "f"(_neg_40));
                float ts_s_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_4) : "f"(ts_s), "f"(ts_p_5));
                float ts_bb_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3) : "f"(ts_s_4), "f"(ts_s));
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(ts_s_4), "f"(ts_bb_3));
                float _sub_41;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(ts_s), "f"(_sub_42));
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(ts_p_5), "f"(ts_bb_3));
                float ts_e_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_41), "f"(_sub_43));
                ts_s = ts_s_4;
                float _add_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(ts_q_5), "f"(ts_e_4));
                float _add_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(ts_e), "f"(_add_45));
                ts_e = _add_44;
                float ts_hk_3[2] = {0.0f, 0.0f};
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_3[0]) : "h"(static_cast<uint16_t>((xw[4 * s_15 + 3]))), "h"(static_cast<uint16_t>((kw[4 * s_15 + 3]))));
                asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(ts_hk_3[1]) : "h"(static_cast<uint16_t>((xw[4 * s_15 + 3]) >> 16)), "h"(static_cast<uint16_t>((kw[4 * s_15 + 3]) >> 16)));
                float ts_w_6 = wf[8 * s_15 + 6];
                float ts_p_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_6) : "f"(ts_w_6), "f"(ts_hk_3[0]));
                float _neg_46;
                asm("neg.f32 %0, %1;" : "=f"(_neg_46) : "f"(ts_p_6));
                float ts_q_6;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_6) : "f"(ts_w_6), "f"(ts_hk_3[0]), "f"(_neg_46));
                float ts_s_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_5) : "f"(ts_s), "f"(ts_p_6));
                float ts_bb_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_4) : "f"(ts_s_5), "f"(ts_s));
                float _sub_48;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(ts_s_5), "f"(ts_bb_4));
                float _sub_47;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(ts_s), "f"(_sub_48));
                float _sub_49;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(ts_p_6), "f"(ts_bb_4));
                float ts_e_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_5) : "f"(_sub_47), "f"(_sub_49));
                ts_s = ts_s_5;
                float _add_51;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(ts_q_6), "f"(ts_e_5));
                float _add_50;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(ts_e), "f"(_add_51));
                ts_e = _add_50;
                float ts_w_7 = wf[8 * s_15 + 6 + 1];
                float ts_p_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ts_p_7) : "f"(ts_w_7), "f"(ts_hk_3[1]));
                float _neg_52;
                asm("neg.f32 %0, %1;" : "=f"(_neg_52) : "f"(ts_p_7));
                float ts_q_7;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ts_q_7) : "f"(ts_w_7), "f"(ts_hk_3[1]), "f"(_neg_52));
                float ts_s_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_6) : "f"(ts_s), "f"(ts_p_7));
                float ts_bb_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_5) : "f"(ts_s_6), "f"(ts_s));
                float _sub_54;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(ts_s_6), "f"(ts_bb_5));
                float _sub_53;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(ts_s), "f"(_sub_54));
                float _sub_55;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(ts_p_7), "f"(ts_bb_5));
                float ts_e_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6) : "f"(_sub_53), "f"(_sub_55));
                ts_s = ts_s_6;
                float _add_57;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(ts_q_7), "f"(ts_e_6));
                float _add_56;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(ts_e), "f"(_add_57));
                ts_e = _add_56;
                ps[s_15] = ts_s;
                pe[s_15] = ts_e;
            }
            int pn_tok = t + step;
            if (pn_tok < T) {
                long long ls_at_0 = (long long)pn_tok * 20480ll + (long long)col;
                #pragma unroll
                for (int s_16 = 0; s_16 < 4; s_16++) {
                    {
                        const uint4* _vptr_58 = reinterpret_cast<const uint4*>(x + (ls_at_0 + (long long)(s_16 * 5120)) + 0);
                        uint4* _vdst_58 = reinterpret_cast<uint4*>(&xn[4 * s_16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_58[_blk] = _vptr_58[_blk];
                        }
                    }
                }
                long long lk_at_0 = (long long)pn_tok * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_17 = 0; s_17 < 4; s_17++) {
                    {
                        const uint4* _vptr_59 = reinterpret_cast<const uint4*>(key + (lk_at_0 + (long long)s_17 * key_stride_p1) + 0);
                        uint4* _vdst_59 = reinterpret_cast<uint4*>(&kn[4 * s_17]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_59[_blk] = _vptr_59[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_60 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_60 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_60[_blk] = _vptr_60[_blk];
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
                float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, ws_send_0, 16);
                plain[i_0] = ws_keep_0 + _shfl_xor_sync_61;
            }
            #pragma unroll
            for (int i_1_1 = 0; i_1_1 < 2; i_1_1++) {
                int ws_upper_1 = lane & 8;
                float ws_keep_1 = ((ws_upper_1 != 0) ? plain[2 + i_1_1] : plain[i_1_1]);
                float ws_send_1 = ((ws_upper_1 != 0) ? plain[i_1_1] : plain[2 + i_1_1]);
                float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, ws_send_1, 8);
                plain[i_1_1] = ws_keep_1 + _shfl_xor_sync_62;
            }
            #pragma unroll
            for (int i_2_1 = 0; i_2_1 < 1; i_2_1++) {
                int ws_upper_2 = lane & 4;
                float ws_keep_2 = ((ws_upper_2 != 0) ? plain[1 + i_2_1] : plain[i_2_1]);
                float ws_send_2 = ((ws_upper_2 != 0) ? plain[i_2_1] : plain[1 + i_2_1]);
                float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, ws_send_2, 4);
                plain[i_2_1] = ws_keep_2 + _shfl_xor_sync_63;
            }
            float ws_own = plain[0];
            float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 2);
            ws_own = ws_own + _shfl_xor_sync_64;
            float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, ws_own, 1);
            ws_own = ws_own + _shfl_xor_sync_65;
            float xl_word = ws_own;
            if ((lane & 3) == 0) {
                if (lane >> 2 < 8) {
                    wtab[xl_row + (lane >> 2)] = xl_word;
                }
            }
            #pragma unroll
            for (int i_3_1 = 0; i_3_1 < 2; i_3_1++) {
                int wp_upper_0 = lane & 16;
                float wp_ks_0 = ((wp_upper_0 != 0) ? ps[2 + i_3_1] : ps[i_3_1]);
                float wp_ke_0 = ((wp_upper_0 != 0) ? pe[2 + i_3_1] : pe[i_3_1]);
                float wp_rs_0 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_0 != 0) ? ps[i_3_1] : ps[2 + i_3_1]), 16);
                float wp_re_0 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_0 != 0) ? pe[i_3_1] : pe[2 + i_3_1]), 16);
                float ts_s_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7) : "f"(wp_ks_0), "f"(wp_rs_0));
                float ts_bb_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6) : "f"(ts_s_7), "f"(wp_ks_0));
                float _sub_67;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(ts_s_7), "f"(ts_bb_6));
                float _sub_66;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(wp_ks_0), "f"(_sub_67));
                float _sub_68;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(wp_rs_0), "f"(ts_bb_6));
                float ts_e_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7) : "f"(_sub_66), "f"(_sub_68));
                wp_ks_0 = ts_s_7;
                float _add_70;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(wp_ke_0), "f"(wp_re_0));
                float _add_69;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_add_70), "f"(ts_e_7));
                wp_ke_0 = _add_69;
                ps[i_3_1] = wp_ks_0;
                pe[i_3_1] = wp_ke_0;
            }
            #pragma unroll
            for (int i_4_1 = 0; i_4_1 < 1; i_4_1++) {
                int wp_upper_1 = lane & 8;
                float wp_ks_1 = ((wp_upper_1 != 0) ? ps[1 + i_4_1] : ps[i_4_1]);
                float wp_ke_1 = ((wp_upper_1 != 0) ? pe[1 + i_4_1] : pe[i_4_1]);
                float wp_rs_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? ps[i_4_1] : ps[1 + i_4_1]), 8);
                float wp_re_1 = __shfl_xor_sync(0xFFFFFFFF, ((wp_upper_1 != 0) ? pe[i_4_1] : pe[1 + i_4_1]), 8);
                float ts_s_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8) : "f"(wp_ks_1), "f"(wp_rs_1));
                float ts_bb_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7) : "f"(ts_s_8), "f"(wp_ks_1));
                float _sub_72;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(ts_s_8), "f"(ts_bb_7));
                float _sub_71;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(wp_ks_1), "f"(_sub_72));
                float _sub_73;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(wp_rs_1), "f"(ts_bb_7));
                float ts_e_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8) : "f"(_sub_71), "f"(_sub_73));
                wp_ks_1 = ts_s_8;
                float _add_75;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(wp_ke_1), "f"(wp_re_1));
                float _add_74;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(_add_75), "f"(ts_e_8));
                wp_ke_1 = _add_74;
                ps[i_4_1] = wp_ks_1;
                pe[i_4_1] = wp_ke_1;
            }
            float wp_s = ps[0];
            float wp_e = pe[0];
            float wp_bs_0 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 4);
            float wp_be_0 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 4);
            float ts_s_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_9) : "f"(wp_s), "f"(wp_bs_0));
            float ts_bb_8;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8) : "f"(ts_s_9), "f"(wp_s));
            float _sub_77;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(ts_s_9), "f"(ts_bb_8));
            float _sub_76;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(wp_s), "f"(_sub_77));
            float _sub_78;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(wp_bs_0), "f"(ts_bb_8));
            float ts_e_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9) : "f"(_sub_76), "f"(_sub_78));
            wp_s = ts_s_9;
            float _add_80;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(wp_e), "f"(wp_be_0));
            float _add_79;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(_add_80), "f"(ts_e_9));
            wp_e = _add_79;
            float wp_bs_1 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 2);
            float wp_be_1 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 2);
            float ts_s_7_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7_1) : "f"(wp_s), "f"(wp_bs_1));
            float ts_bb_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6_1) : "f"(ts_s_7_1), "f"(wp_s));
            float _sub_82;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(ts_s_7_1), "f"(ts_bb_6_1));
            float _sub_81;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(wp_s), "f"(_sub_82));
            float _sub_83;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(wp_bs_1), "f"(ts_bb_6_1));
            float ts_e_7_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7_1) : "f"(_sub_81), "f"(_sub_83));
            wp_s = ts_s_7_1;
            float _add_85;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(wp_e), "f"(wp_be_1));
            float _add_84;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(_add_85), "f"(ts_e_7_1));
            wp_e = _add_84;
            float wp_bs_2 = __shfl_xor_sync(0xFFFFFFFF, wp_s, 1);
            float wp_be_2 = __shfl_xor_sync(0xFFFFFFFF, wp_e, 1);
            float ts_s_8_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8_1) : "f"(wp_s), "f"(wp_bs_2));
            float ts_bb_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7_1) : "f"(ts_s_8_1), "f"(wp_s));
            float _sub_87;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(ts_s_8_1), "f"(ts_bb_7_1));
            float _sub_86;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(wp_s), "f"(_sub_87));
            float _sub_88;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(wp_bs_2), "f"(ts_bb_7_1));
            float ts_e_8_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8_1) : "f"(_sub_86), "f"(_sub_88));
            wp_s = ts_s_8_1;
            float _add_90;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(wp_e), "f"(wp_be_2));
            float _add_89;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(_add_90), "f"(ts_e_8_1));
            wp_e = _add_89;
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
                                float _sub_92;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_92) : "f"(ts_s_9_1), "f"(ts_bb_8_1));
                                float _sub_91;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91) : "f"(xl_fs), "f"(_sub_92));
                                float _sub_93;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(wtab[xl_base + wi_1 * 16 + c_2]), "f"(ts_bb_8_1));
                                float ts_e_9_1;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9_1) : "f"(_sub_91), "f"(_sub_93));
                                xl_fs = ts_s_9_1;
                                float _add_95;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(xl_fe), "f"(wtab[xl_base + wi_1 * 16 + 4 + c_2]));
                                float _add_94;
                                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(_add_95), "f"(ts_e_9_1));
                                xl_fe = _add_94;
                            }
                            xl_s[c_2] = xl_fs;
                            xl_e[c_2] = xl_fe;
                        }
                        uint32_t _cluster_mapa_96;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_96) : "r"(xl_to + (unsigned int)(16 * (xl_job + 1))), "r"(xl_dst));
                        uint32_t _mbar_peer_97;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_97) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xl_dst));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_96), "r"(__float_as_uint(xl_e[0])), "r"(__float_as_uint(xl_e[1])), "r"(__float_as_uint(xl_e[2])), "r"(__float_as_uint(xl_e[3])), "r"(_mbar_peer_97) : "memory");
                    }
                    uint32_t _cluster_mapa_98;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_98) : "r"(xl_to + (unsigned int)(16 * xl_job)), "r"(xl_dst));
                    uint32_t _mbar_peer_99;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_99) : "r"(xbar_addr + ((int)((unsigned int)xk % 2)) * 8), "r"(xl_dst));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_98), "r"(__float_as_uint(xl_s[0])), "r"(__float_as_uint(xl_s[1])), "r"(__float_as_uint(xl_s[2])), "r"(__float_as_uint(xl_s[3])), "r"(_mbar_peer_99) : "memory");
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
                    float _sub_101;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_101) : "f"(ts_s_10), "f"(ts_bb_9));
                    float _sub_100;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(xl_ts), "f"(_sub_101));
                    float _sub_102;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(recv[xl_from + r_1 * 16 + 8]), "f"(ts_bb_9));
                    float ts_e_10;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_10) : "f"(_sub_100), "f"(_sub_102));
                    xl_ts = ts_s_10;
                    float _add_104;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(xl_te), "f"(recv[xl_from + r_1 * 16 + 12]));
                    float _add_103;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_add_104), "f"(ts_e_10));
                    xl_te = _add_103;
                }
                float _add_105;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(xl_ts), "f"(xl_te));
                xl_tot[2] = _add_105;
                float _fdiv_107 = __fdiv_rn(xl_tot[0], 5120.0f);
                float _sqrt_106;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_106) : "f"(_fdiv_107 + eps));
                float sg_rh = __frcp_rn(_sqrt_106);
                float _fdiv_109 = __fdiv_rn(xl_tot[1], 5120.0f);
                float _sqrt_108;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_108) : "f"(_fdiv_109 + eps));
                float sg_rk = __frcp_rn(_sqrt_108);
                float sg_dot;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(sg_dot) : "f"(mul_f32_rn_noftz(xl_tot[2], mul_f32_rn_noftz(sg_rh, sg_rk))), "f"(0.013975425f));
                float _abs_111;
                asm("abs.f32 %0, %1;" : "=f"(_abs_111) : "f"(sg_dot));
                float _max_110;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_110) : "f"(_abs_111), "f"(clamp));
                float sg_mag;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(sg_mag) : "f"(_max_110));
                float sg_u = __uint_as_float(__as_u32(sg_mag) & 2147483647 | __as_u32(sg_dot) & 2147483648u);
                float _neg_113;
                asm("neg.f32 %0, %1;" : "=f"(_neg_113) : "f"(sg_u));
                float _exp_112 = __nv_expf(_neg_113);
                float sg_gate = __fdiv_rn(1.0f, 1.0f + _exp_112);
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
                for (int m_5 = 0; m_5 < 4; m_5++) {
                    float2 to_o2 = fma_f32x2_rn_noftz(to_g2, cake_bf16x2_to_f32x2(__as_bf16x2(vw[m_5])), cake_bf16x2_to_f32x2(__as_bf16x2(xw[4 * s_19 + m_5])));
                    to_out[2 * m_5] = to_o2.x;
                    to_out[2 * m_5 + 1] = to_o2.y;
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
                        const uint4* _vptr_114 = reinterpret_cast<const uint4*>(x + (ls_at_1 + (long long)(s_20 * 5120)) + 0);
                        uint4* _vdst_114 = reinterpret_cast<uint4*>(&xn[4 * s_20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_114[_blk] = _vptr_114[_blk];
                        }
                    }
                }
                long long lk_at_1 = (long long)pn_tok_1 * key_stride_p0 + (long long)col;
                #pragma unroll
                for (int s_21 = 0; s_21 < 4; s_21++) {
                    {
                        const uint4* _vptr_115 = reinterpret_cast<const uint4*>(key + (lk_at_1 + (long long)s_21 * key_stride_p1) + 0);
                        uint4* _vdst_115 = reinterpret_cast<uint4*>(&kn[4 * s_21]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_115[_blk] = _vptr_115[_blk];
                        }
                    }
                }
                {
                    const uint4* _vptr_116 = reinterpret_cast<const uint4*>(value + ((long long)pn_tok_1 * value_stride_p0 + (long long)col) + 0);
                    uint4* _vdst_116 = reinterpret_cast<uint4*>(&vn[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_116[_blk] = _vptr_116[_blk];
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
            for (int s_22 = 0; s_22 < 4; s_22++) {
                reinterpret_cast<int4*>(out + (to_base + (long long)(s_22 * 5120)))[0] = reinterpret_cast<int4*>(xw + (4 * s_22))[0];
            }
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 16; i_8++) {
            xw[i_8] = xn[i_8];
            kw[i_8] = kn[i_8];
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 4; i_9++) {
            vw[i_9] = vn[i_9];
        }
        keep = keep_n;
    }
}

} // extern "C"
