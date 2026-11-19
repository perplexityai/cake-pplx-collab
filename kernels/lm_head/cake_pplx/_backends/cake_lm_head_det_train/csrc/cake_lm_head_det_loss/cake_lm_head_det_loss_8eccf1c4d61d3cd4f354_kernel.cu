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

#define LAUNCH_MIN_BLOCKS 8

#include <math_constants.h>


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_lm_head_det_loss_8eccf1c4d61d3cd4f354(__nv_bfloat16* __restrict__ z, long long* __restrict__ labels, float* __restrict__ lse, float* __restrict__ d, float* __restrict__ rlab, int row0, int d_off, int V)
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
    float* res = reinterpret_cast<float*>(smem_raw + 0);
    const int res_addr = smem + 0;

    // === Task calls (declaration order) ===
    int r = blockIdx.y;
    int t = row0 + r;
    long long lbl = labels[t];
    float lse_t = lse[t];
    float d_raw = d[d_off + r];
    unsigned long long row_base = (unsigned long long)r * (unsigned long long)V;
    int num_vecs = V / 8;
    int vi0 = blockIdx.x * 1024 + tid;
    float rg_s = 0.0f;
    float rg_r = 0.0f;
    if (lbl < 0ll) {
        float zero_out[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            zero_out[j] = 0.0f;
        }
        #pragma unroll
        for (int kz = 0; kz < 8; kz++) {
            int vi_z = vi0 + kz * 128;
            if (vi_z < num_vecs) {
                unsigned int _packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero_out[_lp*2 + 0], zero_out[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero_out[_lp*2 + 2], zero_out[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero_out[_lp*2 + 4], zero_out[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero_out[_lp*2 + 6], zero_out[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(z + (row_base + (unsigned long long)vi_z * 8ull))[0] = reinterpret_cast<int4*>(_packed_0)[0];
            }
        }
    } else {
        int lbl32 = (int)lbl;
        int lbl_vec = lbl32 / 8;
        float nlse2 = (-lse_t) * 1.442695f;
        unsigned int zero32 = 0;
        unsigned int zw[32];
        float zf[64];
        #pragma unroll
        for (int kl = 0; kl < 8; kl++) {
            int vi_l = vi0 + kl * 128;
            if (vi_l < num_vecs) {
                {
                    uint4 _uv4_0 = *reinterpret_cast<const uint4*>(z + (row_base + (unsigned long long)vi_l * 8ull) + 0);
                    zw[kl * 4 + 0] = _uv4_0.x;
                    zw[kl * 4 + 1] = _uv4_0.y;
                    zw[kl * 4 + 2] = _uv4_0.z;
                    zw[kl * 4 + 3] = _uv4_0.w;
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int vi = vi0 + k * 128;
            if (vi < num_vecs) {
                float out[8];
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    uint32_t _prmt_1;
                    asm("prmt.b32 %0, %1, %2, 0x1044;" : "=r"(_prmt_1) : "r"(zw[k * 4 + w]), "r"(zero32));
                    float exp = expf(__uint_as_float(_prmt_1) - lse_t);
                    uint32_t _prmt_2;
                    asm("prmt.b32 %0, %1, %2, 0x3244;" : "=r"(_prmt_2) : "r"(zw[k * 4 + w]), "r"(zero32));
                    float exp_0 = expf(__uint_as_float(_prmt_2) - lse_t);
                    out[2 * w] = d_raw * (-exp);
                    out[2 * w + 1] = d_raw * (-exp_0);
                }
                if (vi == lbl_vec) {
                    int col0 = vi * 8;
                    #pragma unroll
                    for (int wl = 0; wl < 4; wl++) {
                        if (col0 + 2 * wl == lbl32) {
                            uint32_t _prmt_5;
                            asm("prmt.b32 %0, %1, %2, 0x1044;" : "=r"(_prmt_5) : "r"(zw[k * 4 + wl]), "r"(zero32));
                            float _exp_4 = expf(__uint_as_float(_prmt_5) - lse_t);
                            float _sub_3;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(1.0f), "f"(_exp_4));
                            out[2 * wl] = mul_f32_rn_noftz(d_raw, _sub_3);
                            rg_s = out[2 * wl];
                        }
                        if (col0 + 2 * wl + 1 == lbl32) {
                            uint32_t _prmt_8;
                            asm("prmt.b32 %0, %1, %2, 0x3244;" : "=r"(_prmt_8) : "r"(zw[k * 4 + wl]), "r"(zero32));
                            float _exp_7 = expf(__uint_as_float(_prmt_8) - lse_t);
                            float _sub_6;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(1.0f), "f"(_exp_7));
                            out[2 * wl + 1] = mul_f32_rn_noftz(d_raw, _sub_6);
                            rg_s = out[2 * wl + 1];
                        }
                    }
                }
                unsigned int _packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(z + (row_base + (unsigned long long)vi * 8ull))[0] = reinterpret_cast<int4*>(_packed_1)[0];
            }
        }
        float lr_hi = (float)(__nv_bfloat16)rg_s;
        float _abs_9;
        asm("abs.f32 %0, %1;" : "=f"(_abs_9) : "f"(lr_hi));
        float _sub_10;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(rg_s), "f"(lr_hi));
        float lr_r = ((_abs_9 <= 3.4028235e+38f) ? _sub_10 : 0.0f);
        rg_r = lr_r;
    }
    res[tid] = rg_r;
    __syncthreads();
    if (tid == 0) {
        if (lbl >= 0ll) {
            if (lbl < (long long)num_vecs * 8ll) {
                int own_vec = (int)lbl >> 3;
                int first_vec = blockIdx.x * 1024;
                if (own_vec >= first_vec) {
                    if (own_vec < first_vec + 1024) {
                        rlab[t] = res[own_vec & 127];
                    }
                }
            }
        }
    }
}

} // extern "C"
