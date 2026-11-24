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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mhc_det_train_bwd_reduce__s27_wcvt_fadd_fshfl_trows__h7168(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_rows, int accumulate)
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
        long long e0 = (long long)((b - 27) * 256 + tid_0) * 4ll;
        float out[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            out[k] = 0.0f;
        }
        #pragma unroll 2
        for (int p = 0; p < n_parts; p++) {
            float vec_load[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + ((long long)p * 688128ll + e0) + 0);
                vec_load[0 + 0] = _v4.x;
                vec_load[0 + 1] = _v4.y;
                vec_load[0 + 2] = _v4.z;
                vec_load[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_1 = 0; k_1 < 4; k_1++) {
                float _add_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(out[k_1]), "f"(vec_load[k_1]));
                out[k_1] = _add_1;
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
            for (int k_2 = 0; k_2 < 4; k_2++) {
                float _add_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(out[k_2]), "f"(_vec_load_2[k_2]));
                out[k_2] = _add_3;
            }
        }
        {
            float4 _v4 = make_float4(out[0 + 0], out[0 + 1], out[0 + 2], out[0 + 3]);
            *reinterpret_cast<float4*>((dfn + e0) + 0) = _v4;
        }
    } else {
        int lane_1 = tid_0 & 31;
        int wrp = tid_0 >> 5;
        float hh[1];
        #pragma unroll
        for (int k_3 = 0; k_3 < 1; k_3++) {
            hh[k_3] = 0.0f;
        }
        int nj = n_rows + 255 >> 8;
        if (nj <= 2) {
            int last_row = n_rows - 1;
            long long cbase = (long long)b * (long long)n_rows;
            #pragma unroll
            for (int k_4 = 0; k_4 < 1; k_4++) {
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    int row = u * 256 + (k_4 * 256 + tid_0);
                    int rowc = ((row < n_rows) ? row : last_row);
                    float xv[1];
                    {
                        uint32_t _scalar_bits_4;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_4) : "l"((const void*)(param_partial + (cbase + (long long)rowc) + (0))) : "memory");
                        xv[0] = __uint_as_float(_scalar_bits_4);
                    }
                    float xr = xv[0];
                    float _add_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(hh[k_4]), "f"(((row < n_rows) ? xr : 0.0f)));
                    hh[k_4] = _add_5;
                }
            }
        } else if (nj <= 8) {
            int last_row_1 = n_rows - 1;
            long long cbase_1 = (long long)b * (long long)n_rows;
            #pragma unroll
            for (int k_5 = 0; k_5 < 1; k_5++) {
                #pragma unroll
                for (int u_1 = 0; u_1 < 8; u_1++) {
                    int row_1 = u_1 * 256 + (k_5 * 256 + tid_0);
                    int rowc_1 = ((row_1 < n_rows) ? row_1 : last_row_1);
                    float xv_1[1];
                    {
                        uint32_t _scalar_bits_6;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_6) : "l"((const void*)(param_partial + (cbase_1 + (long long)rowc_1) + (0))) : "memory");
                        xv_1[0] = __uint_as_float(_scalar_bits_6);
                    }
                    float xr_1 = xv_1[0];
                    float _add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(hh[k_5]), "f"(((row_1 < n_rows) ? xr_1 : 0.0f)));
                    hh[k_5] = _add_7;
                }
            }
        } else {
            #pragma unroll 1
            for (int jb = 0; jb < nj; jb += 32) {
                int last_row_2 = n_rows - 1;
                long long cbase_2 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_6 = 0; k_6 < 1; k_6++) {
                    #pragma unroll
                    for (int u_2 = 0; u_2 < 32; u_2++) {
                        int row_2 = (jb + u_2) * 256 + (k_6 * 256 + tid_0);
                        int rowc_2 = ((row_2 < n_rows) ? row_2 : last_row_2);
                        float xv_2[1];
                        {
                            uint32_t _scalar_bits_8;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_8) : "l"((const void*)(param_partial + (cbase_2 + (long long)rowc_2) + (0))) : "memory");
                            xv_2[0] = __uint_as_float(_scalar_bits_8);
                        }
                        float xr_2 = xv_2[0];
                        float _add_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(hh[k_6]), "f"(((row_2 < n_rows) ? xr_2 : 0.0f)));
                        hh[k_6] = _add_9;
                    }
                }
            }
        }
        #pragma unroll
        for (int k_7 = 0; k_7 < 1; k_7++) {
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 16);
            float _add_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(hh[k_7]), "f"(_shfl_xor_sync_11));
            hh[k_7] = _add_10;
            float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 8);
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(hh[k_7]), "f"(_shfl_xor_sync_13));
            hh[k_7] = _add_12;
            float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 4);
            float _add_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(hh[k_7]), "f"(_shfl_xor_sync_15));
            hh[k_7] = _add_14;
            float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 2);
            float _add_16;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(hh[k_7]), "f"(_shfl_xor_sync_17));
            hh[k_7] = _add_16;
            float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 1);
            float _add_18;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(hh[k_7]), "f"(_shfl_xor_sync_19));
            hh[k_7] = _add_18;
            if (lane_1 == 0) {
                vw[k_7 * 8 + wrp] = hh[k_7];
            }
        }
        __syncthreads();
        if (tid_0 == 0) {
            float fs[1];
            fs[0] = 0.0f;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                float _add_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(fs[0]), "f"(vw[m]));
                fs[0] = _add_20;
            }
            long long col64 = (long long)b;
            if (accumulate != 0) {
                if (b < 24) {
                    float _add_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(fs[0]), "f"(dbase[col64]));
                    fs[0] = _add_21;
                } else {
                    float _add_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(fs[0]), "f"(dscale[col64 - 24ll]));
                    fs[0] = _add_22;
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
