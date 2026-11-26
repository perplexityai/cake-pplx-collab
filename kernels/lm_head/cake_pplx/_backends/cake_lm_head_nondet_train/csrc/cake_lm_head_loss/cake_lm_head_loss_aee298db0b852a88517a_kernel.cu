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

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_loss_aee298db0b852a88517a(float* __restrict__ PU, float* __restrict__ PMS, __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ Z, long long* __restrict__ labels, float* __restrict__ DXA, float* __restrict__ ROWS, float* __restrict__ LOGP, float* __restrict__ LOSS, long long* __restrict__ LAB, int parts, int T, int V, int H, int with_u, float loss_div)
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
    float* stg = reinterpret_cast<float*>(smem_raw + 0);
    const int stg_addr = smem + 0;
    float* ebuf = reinterpret_cast<float*>(smem_raw + 32768);
    const int ebuf_addr = smem + 32768;
    float* stat = reinterpret_cast<float*>(smem_raw + 36864);
    const int stat_addr = smem + 36864;

    // === Task calls (declaration order) ===
    float d = __fdiv_rn(-1.0f, loss_div);
    int tiles = H / 32;
    int tc = bid / tiles;
    int h0 = bid % tiles * 32;
    int live = 0;
    if (with_u != 0) {
        if (tc < T) {
            live = 1;
        }
    }
    long long yu = -1ll;
    if (live == 1) {
        if (warp == 7) {
            yu = labels[tc];
        }
    }
    int q = tid / 8;
    int c4 = tid % 8;
    float buf[32] = {0};
    if (live == 1) {
        if (q < parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(PU + ((unsigned long long)(q * T + tc) * (unsigned long long)H + (unsigned long long)(h0 + 4 * c4)) + 0);
                buf[0 + 0] = _v4.x;
                buf[0 + 1] = _v4.y;
                buf[0 + 2] = _v4.z;
                buf[0 + 3] = _v4.w;
            }
        }
        if (q + 32 < parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(PU + ((unsigned long long)((q + 32) * T + tc) * (unsigned long long)H + (unsigned long long)(h0 + 4 * c4)) + 0);
                buf[4 + 0] = _v4.x;
                buf[4 + 1] = _v4.y;
                buf[4 + 2] = _v4.z;
                buf[4 + 3] = _v4.w;
            }
        }
        if (q + 64 < parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(PU + ((unsigned long long)((q + 64) * T + tc) * (unsigned long long)H + (unsigned long long)(h0 + 4 * c4)) + 0);
                buf[8 + 0] = _v4.x;
                buf[8 + 1] = _v4.y;
                buf[8 + 2] = _v4.z;
                buf[8 + 3] = _v4.w;
            }
        }
        if (q + 96 < parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(PU + ((unsigned long long)((q + 96) * T + tc) * (unsigned long long)H + (unsigned long long)(h0 + 4 * c4)) + 0);
                buf[12 + 0] = _v4.x;
                buf[12 + 1] = _v4.y;
                buf[12 + 2] = _v4.z;
                buf[12 + 3] = _v4.w;
            }
        }
        if (q + 128 < parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(PU + ((unsigned long long)((q + 128) * T + tc) * (unsigned long long)H + (unsigned long long)(h0 + 4 * c4)) + 0);
                buf[16 + 0] = _v4.x;
                buf[16 + 1] = _v4.y;
                buf[16 + 2] = _v4.z;
                buf[16 + 3] = _v4.w;
            }
        }
        if (q + 160 < parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(PU + ((unsigned long long)((q + 160) * T + tc) * (unsigned long long)H + (unsigned long long)(h0 + 4 * c4)) + 0);
                buf[20 + 0] = _v4.x;
                buf[20 + 1] = _v4.y;
                buf[20 + 2] = _v4.z;
                buf[20 + 3] = _v4.w;
            }
        }
        if (q + 192 < parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(PU + ((unsigned long long)((q + 192) * T + tc) * (unsigned long long)H + (unsigned long long)(h0 + 4 * c4)) + 0);
                buf[24 + 0] = _v4.x;
                buf[24 + 1] = _v4.y;
                buf[24 + 2] = _v4.z;
                buf[24 + 3] = _v4.w;
            }
        }
        if (q + 224 < parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(PU + ((unsigned long long)((q + 224) * T + tc) * (unsigned long long)H + (unsigned long long)(h0 + 4 * c4)) + 0);
                buf[28 + 0] = _v4.x;
                buf[28 + 1] = _v4.y;
                buf[28 + 2] = _v4.z;
                buf[28 + 3] = _v4.w;
            }
        }
    }
    float wy = 0.0f;
    if (live == 1) {
        if (warp == 7) {
            if (yu >= 0ll) {
                if (yu < (long long)V) {
                    wy = (float)W[(unsigned long long)yu * (unsigned long long)H + (unsigned long long)(h0 + lane)];
                }
            }
        }
    }
    float zv[4] = {0};
    float ok[4] = {0};
    if (bid == 0) {
        if (warp == 6) {
            if (lane == 0) {
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    if (t < T) {
                        long long yl = labels[t];
                        LAB[t] = yl;
                        if (yl >= 0ll) {
                            long long yc = yl;
                            if (yc >= (long long)V) {
                                yc = (long long)(V - 1);
                            }
                            zv[t] = (float)Z[(unsigned long long)t * (unsigned long long)V + (unsigned long long)yc];
                            ok[t] = 1.0f;
                        }
                    }
                }
            }
        }
    }
    if (warp < T) {
        int tw = warp;
        float pm[8] = {0};
        float ps[8] = {0};
        if (lane < parts) {
            pm[0] = PMS[(lane * T + tw) * 2];
            ps[0] = PMS[(lane * T + tw) * 2 + 1];
        }
        if (lane + 32 < parts) {
            pm[1] = PMS[((lane + 32) * T + tw) * 2];
            ps[1] = PMS[((lane + 32) * T + tw) * 2 + 1];
        }
        if (lane + 64 < parts) {
            pm[2] = PMS[((lane + 64) * T + tw) * 2];
            ps[2] = PMS[((lane + 64) * T + tw) * 2 + 1];
        }
        if (lane + 96 < parts) {
            pm[3] = PMS[((lane + 96) * T + tw) * 2];
            ps[3] = PMS[((lane + 96) * T + tw) * 2 + 1];
        }
        if (lane + 128 < parts) {
            pm[4] = PMS[((lane + 128) * T + tw) * 2];
            ps[4] = PMS[((lane + 128) * T + tw) * 2 + 1];
        }
        if (lane + 160 < parts) {
            pm[5] = PMS[((lane + 160) * T + tw) * 2];
            ps[5] = PMS[((lane + 160) * T + tw) * 2 + 1];
        }
        if (lane + 192 < parts) {
            pm[6] = PMS[((lane + 192) * T + tw) * 2];
            ps[6] = PMS[((lane + 192) * T + tw) * 2 + 1];
        }
        if (lane + 224 < parts) {
            pm[7] = PMS[((lane + 224) * T + tw) * 2];
            ps[7] = PMS[((lane + 224) * T + tw) * 2 + 1];
        }
        float mx = -CUDART_INF_F;
        if (lane < parts) {
            mx = max_noftz(mx, pm[0]);
        }
        if (lane + 32 < parts) {
            mx = max_noftz(mx, pm[1]);
        }
        if (lane + 64 < parts) {
            mx = max_noftz(mx, pm[2]);
        }
        if (lane + 96 < parts) {
            mx = max_noftz(mx, pm[3]);
        }
        if (lane + 128 < parts) {
            mx = max_noftz(mx, pm[4]);
        }
        if (lane + 160 < parts) {
            mx = max_noftz(mx, pm[5]);
        }
        if (lane + 192 < parts) {
            mx = max_noftz(mx, pm[6]);
        }
        if (lane + 224 < parts) {
            mx = max_noftz(mx, pm[7]);
        }
        float _warp_reduce_8 = mx;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_8 = max_noftz(_warp_reduce_8, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_8, offset));
        mx = _warp_reduce_8;
        float part = 0.0f;
        if (lane < parts) {
            float e_0 = 0.0f;
            if (pm[0] != -CUDART_INF_F) {
                float _exp_9 = expf(pm[0] - mx);
                e_0 = _exp_9;
            }
            ebuf[tw * 256 + lane] = e_0;
            part = __fmaf_rn(ps[0], e_0, part);
        }
        if (lane + 32 < parts) {
            float e_1 = 0.0f;
            if (pm[1] != -CUDART_INF_F) {
                float _exp_10 = expf(pm[1] - mx);
                e_1 = _exp_10;
            }
            ebuf[tw * 256 + lane + 32] = e_1;
            part = __fmaf_rn(ps[1], e_1, part);
        }
        if (lane + 64 < parts) {
            float e_2 = 0.0f;
            if (pm[2] != -CUDART_INF_F) {
                float _exp_11 = expf(pm[2] - mx);
                e_2 = _exp_11;
            }
            ebuf[tw * 256 + lane + 64] = e_2;
            part = __fmaf_rn(ps[2], e_2, part);
        }
        if (lane + 96 < parts) {
            float e_3 = 0.0f;
            if (pm[3] != -CUDART_INF_F) {
                float _exp_12 = expf(pm[3] - mx);
                e_3 = _exp_12;
            }
            ebuf[tw * 256 + lane + 96] = e_3;
            part = __fmaf_rn(ps[3], e_3, part);
        }
        if (lane + 128 < parts) {
            float e_4 = 0.0f;
            if (pm[4] != -CUDART_INF_F) {
                float _exp_13 = expf(pm[4] - mx);
                e_4 = _exp_13;
            }
            ebuf[tw * 256 + lane + 128] = e_4;
            part = __fmaf_rn(ps[4], e_4, part);
        }
        if (lane + 160 < parts) {
            float e_5 = 0.0f;
            if (pm[5] != -CUDART_INF_F) {
                float _exp_14 = expf(pm[5] - mx);
                e_5 = _exp_14;
            }
            ebuf[tw * 256 + lane + 160] = e_5;
            part = __fmaf_rn(ps[5], e_5, part);
        }
        if (lane + 192 < parts) {
            float e_6 = 0.0f;
            if (pm[6] != -CUDART_INF_F) {
                float _exp_15 = expf(pm[6] - mx);
                e_6 = _exp_15;
            }
            ebuf[tw * 256 + lane + 192] = e_6;
            part = __fmaf_rn(ps[6], e_6, part);
        }
        if (lane + 224 < parts) {
            float e_7 = 0.0f;
            if (pm[7] != -CUDART_INF_F) {
                float _exp_16 = expf(pm[7] - mx);
                e_7 = _exp_16;
            }
            ebuf[tw * 256 + lane + 224] = e_7;
            part = __fmaf_rn(ps[7], e_7, part);
        }
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, part, 16);
        part = part + _shfl_xor_sync_17;
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, part, 8);
        part = part + _shfl_xor_sync_18;
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, part, 4);
        part = part + _shfl_xor_sync_19;
        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, part, 2);
        part = part + _shfl_xor_sync_20;
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, part, 1);
        part = part + _shfl_xor_sync_21;
        if (lane == 0) {
            stat[2 * tw] = mx;
            stat[2 * tw + 1] = part;
        }
    }
    if (live == 1) {
        if (q < parts) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)((q * 32 + 4 * c4) * 4)), "f"(buf[0]), "f"(buf[1]), "f"(buf[2]), "f"(buf[3]) : "memory");
        }
        if (q + 32 < parts) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(((q + 32) * 32 + 4 * c4) * 4)), "f"(buf[4]), "f"(buf[5]), "f"(buf[6]), "f"(buf[7]) : "memory");
        }
        if (q + 64 < parts) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(((q + 64) * 32 + 4 * c4) * 4)), "f"(buf[8]), "f"(buf[9]), "f"(buf[10]), "f"(buf[11]) : "memory");
        }
        if (q + 96 < parts) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(((q + 96) * 32 + 4 * c4) * 4)), "f"(buf[12]), "f"(buf[13]), "f"(buf[14]), "f"(buf[15]) : "memory");
        }
        if (q + 128 < parts) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(((q + 128) * 32 + 4 * c4) * 4)), "f"(buf[16]), "f"(buf[17]), "f"(buf[18]), "f"(buf[19]) : "memory");
        }
        if (q + 160 < parts) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(((q + 160) * 32 + 4 * c4) * 4)), "f"(buf[20]), "f"(buf[21]), "f"(buf[22]), "f"(buf[23]) : "memory");
        }
        if (q + 192 < parts) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(((q + 192) * 32 + 4 * c4) * 4)), "f"(buf[24]), "f"(buf[25]), "f"(buf[26]), "f"(buf[27]) : "memory");
        }
        if (q + 224 < parts) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stg_addr + (unsigned int)(((q + 224) * 32 + 4 * c4) * 4)), "f"(buf[28]), "f"(buf[29]), "f"(buf[30]), "f"(buf[31]) : "memory");
        }
    }
    __syncthreads();
    if (live == 1) {
        if (warp == 7) {
            float out = 0.0f;
            if (yu >= 0ll) {
                float acc = 0.0f;
                #pragma unroll 8
                for (int g = 0; g < parts; g++) {
                    acc = __fmaf_rn(stg[g * 32 + lane], ebuf[tc * 256 + g], acc);
                }
                float _fdiv_22 = __fdiv_rn(acc, stat[2 * tc + 1]);
                out = d * (wy - _fdiv_22);
            }
            DXA[(unsigned long long)tc * (unsigned long long)H + (unsigned long long)(h0 + lane)] = out;
        }
    }
    if (bid == 0) {
        if (warp == 6) {
            if (lane == 0) {
                float terms[4];
                #pragma unroll
                for (int t_1 = 0; t_1 < 4; t_1++) {
                    terms[t_1] = 0.0f;
                    if (t_1 < T) {
                        float _log_23 = logf(stat[2 * t_1 + 1]);
                        float lse = stat[2 * t_1] + _log_23;
                        float logp = 0.0f;
                        float dt = 0.0f;
                        if (ok[t_1] != 0.0f) {
                            logp = zv[t_1] - lse;
                            dt = d;
                        }
                        ROWS[2 * t_1] = lse;
                        ROWS[2 * t_1 + 1] = dt;
                        LOGP[t_1] = logp;
                        terms[t_1] = logp;
                    }
                }
                float s02;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s02) : "f"(terms[0]), "f"(terms[2]));
                float s13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s13) : "f"(terms[1]), "f"(terms[3]));
                float _add_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(s02), "f"(s13));
                float fin = __fdiv_rn(_add_24, loss_div);
                float _sub_25;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(0.0f), "f"(fin));
                LOSS[0] = _sub_25;
            }
        }
    }
}

} // extern "C"
