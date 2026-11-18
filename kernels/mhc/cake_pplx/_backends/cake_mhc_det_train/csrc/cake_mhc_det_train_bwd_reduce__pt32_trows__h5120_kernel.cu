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

extern "C" {

__global__ __launch_bounds__(32) void
kernel_cake_mhc_det_train_bwd_reduce__pt32_trows__h5120(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_rows, int accumulate)
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
    float* vw = reinterpret_cast<float*>(smem_raw + 0);
    const int vw_addr = smem + 0;

    // === Task calls (declaration order) ===
    int b = bid;
    int tid_0 = tid;
    if (b >= 27) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        long long e0 = (long long)((b - 27) * 32 + tid_0) * 4ll;
        float out[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            out[k] = 0.0f;
        }
        float _vec_load_0[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + e0 + 0);
            _vec_load_0[0 + 0] = _v4.x;
            _vec_load_0[0 + 1] = _v4.y;
            _vec_load_0[0 + 2] = _v4.z;
            _vec_load_0[0 + 3] = _v4.w;
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 4; k_1++) {
            out[k_1] = ((_vec_load_0[k_1] != 0.0f) ? _vec_load_0[k_1] : 0.0f);
        }
        #pragma unroll 1
        for (int p = 1; p < n_parts; p++) {
            float vec_load[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + ((long long)p * 491520ll + e0) + 0);
                vec_load[0 + 0] = _v4.x;
                vec_load[0 + 1] = _v4.y;
                vec_load[0 + 2] = _v4.z;
                vec_load[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; k_2++) {
                float _add_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(out[k_2]), "f"(vec_load[k_2]));
                out[k_2] = _add_2;
            }
        }
        if (accumulate != 0) {
            float _vec_load_2[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn + e0 + 0);
                _vec_load_2[0 + 0] = _v4.x;
                _vec_load_2[0 + 1] = _v4.y;
                _vec_load_2[0 + 2] = _v4.z;
                _vec_load_2[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_3 = 0; k_3 < 4; k_3++) {
                float _add_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(out[k_3]), "f"(_vec_load_2[k_3]));
                out[k_3] = _add_4;
            }
        }
        {
            float4 _v4 = make_float4(out[0 + 0], out[0 + 1], out[0 + 2], out[0 + 3]);
            *reinterpret_cast<float4*>((dfn + e0) + 0) = _v4;
        }
    } else {
        int lane_1 = tid_0 & 31;
        int wrp = tid_0 >> 5;
        float hh[8];
        #pragma unroll
        for (int k_4 = 0; k_4 < 8; k_4++) {
            hh[k_4] = 0.0f;
        }
        int nj = n_rows + 255 >> 8;
        #pragma unroll 1
        for (int jb = 0; jb < nj; jb += 4) {
            int last_row = n_rows - 1;
            long long cbase = (long long)b * (long long)n_rows;
            #pragma unroll
            for (int k_5 = 0; k_5 < 8; k_5++) {
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    int row = (jb + u) * 256 + (k_5 * 32 + tid_0);
                    int rowc = ((row < n_rows) ? row : last_row);
                    float xv[1];
                    {
                        uint32_t _scalar_bits_5;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_5) : "l"((const void*)(param_partial + (cbase + (long long)rowc) + (0))) : "memory");
                        xv[0] = __uint_as_float(_scalar_bits_5);
                    }
                    float xr = xv[0];
                    float _add_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(hh[k_5]), "f"(((row < n_rows) ? xr : 0.0f)));
                    hh[k_5] = _add_6;
                }
            }
        }
        #pragma unroll
        for (int k_6 = 0; k_6 < 8; k_6++) {
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, hh[k_6], 16);
            float _add_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(hh[k_6]), "f"(_shfl_xor_sync_8));
            hh[k_6] = _add_7;
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, hh[k_6], 8);
            float _add_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(hh[k_6]), "f"(_shfl_xor_sync_10));
            hh[k_6] = _add_9;
            float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, hh[k_6], 4);
            float _add_11;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(hh[k_6]), "f"(_shfl_xor_sync_12));
            hh[k_6] = _add_11;
            float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, hh[k_6], 2);
            float _add_13;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(hh[k_6]), "f"(_shfl_xor_sync_14));
            hh[k_6] = _add_13;
            float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, hh[k_6], 1);
            float _add_15;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(hh[k_6]), "f"(_shfl_xor_sync_16));
            hh[k_6] = _add_15;
            if (lane_1 == 0) {
                vw[k_6 + wrp] = hh[k_6];
            }
        }
        __syncthreads();
        if (tid_0 == 0) {
            float fs[1];
            fs[0] = 0.0f;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                float _add_17;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(fs[0]), "f"(vw[m]));
                fs[0] = _add_17;
            }
            long long col64 = (long long)b;
            if (accumulate != 0) {
                if (b < 24) {
                    float _add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(fs[0]), "f"(dbase[col64]));
                    fs[0] = _add_18;
                } else {
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(fs[0]), "f"(dscale[col64 - 24ll]));
                    fs[0] = _add_19;
                }
            }
            if (b < 24) {
                *(reinterpret_cast<float*>(dbase + col64) + (0)) = fs[0];
            } else {
                *(reinterpret_cast<float*>(dscale + (col64 - 24ll)) + (0)) = fs[0];
            }
        }
    }
}

} // extern "C"
