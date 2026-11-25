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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mhc_det_train_bwd_reduce__wpu4_fadd_trows_pair(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_rows, int accumulate)
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int b = bid;
    int tid_0 = tid;
    if (b >= 27) {
        long long e0 = (long long)((b - 27) * 256 + tid_0) * 4ll;
        float out[4];
        float hi[4];
        float lo[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            hi[k] = 0.0f;
            lo[k] = 0.0f;
        }
        #pragma unroll 4
        for (int p = 0; p < n_parts; p++) {
            float vec_load[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + ((long long)p * 393216ll + e0) + 0);
                vec_load[0 + 0] = _v4.x;
                vec_load[0 + 1] = _v4.y;
                vec_load[0 + 2] = _v4.z;
                vec_load[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_1 = 0; k_1 < 4; k_1++) {
                float add;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(hi[k_1]), "f"(vec_load[k_1]));
                float sub;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(add), "f"(hi[k_1]));
                float _sub_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(add), "f"(sub));
                float _sub_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(hi[k_1]), "f"(_sub_4));
                float _sub_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(vec_load[k_1]), "f"(sub));
                float _add_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(_sub_3), "f"(_sub_5));
                float _add_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(lo[k_1]), "f"(_add_2));
                lo[k_1] = _add_1;
                hi[k_1] = add;
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
                float add_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(hi[k_2]), "f"(_vec_load_2[k_2]));
                float sub_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(add_1), "f"(hi[k_2]));
                float _sub_10;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(add_1), "f"(sub_1));
                float _sub_9;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(hi[k_2]), "f"(_sub_10));
                float _sub_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(_vec_load_2[k_2]), "f"(sub_1));
                float _add_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(_sub_9), "f"(_sub_11));
                float _add_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(lo[k_2]), "f"(_add_8));
                lo[k_2] = _add_7;
                hi[k_2] = add_1;
            }
        }
        #pragma unroll
        for (int k_3 = 0; k_3 < 4; k_3++) {
            bool nonfin = (__as_u32(hi[k_3]) & 2139095040) == 2139095040;
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(hi[k_3]), "f"(lo[k_3]));
            out[k_3] = ((nonfin) ? hi[k_3] : _add_12);
        }
        {
            float4 _v4 = make_float4(out[0 + 0], out[0 + 1], out[0 + 2], out[0 + 3]);
            *reinterpret_cast<float4*>((dfn + e0) + 0) = _v4;
        }
    } else {
        int lane_1 = tid_0 & 31;
        int wrp = tid_0 >> 5;
        float hh[1];
        float ll[1];
        #pragma unroll
        for (int k_4 = 0; k_4 < 1; k_4++) {
            hh[k_4] = 0.0f;
            ll[k_4] = 0.0f;
        }
        int nj = n_rows + 255 >> 8;
        if (nj <= 2) {
            int last_row = n_rows - 1;
            long long cbase = (long long)b * (long long)n_rows;
            #pragma unroll
            for (int k_5 = 0; k_5 < 1; k_5++) {
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    int row = u * 256 + (k_5 * 256 + tid_0);
                    int rowc = ((row < n_rows) ? row : last_row);
                    float xr = param_partial[cbase + (long long)rowc];
                    float add_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(hh[k_5]), "f"(((row < n_rows) ? xr : 0.0f)));
                    float sub_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(add_2), "f"(hh[k_5]));
                    float _sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(add_2), "f"(sub_2));
                    float _sub_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(hh[k_5]), "f"(_sub_16));
                    float _sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(((row < n_rows) ? xr : 0.0f)), "f"(sub_2));
                    float _add_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(_sub_15), "f"(_sub_17));
                    float _add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(ll[k_5]), "f"(_add_14));
                    ll[k_5] = _add_13;
                    hh[k_5] = add_2;
                }
            }
        } else if (nj <= 8) {
            int last_row_1 = n_rows - 1;
            long long cbase_1 = (long long)b * (long long)n_rows;
            #pragma unroll
            for (int k_6 = 0; k_6 < 1; k_6++) {
                #pragma unroll
                for (int u_1 = 0; u_1 < 8; u_1++) {
                    int row_1 = u_1 * 256 + (k_6 * 256 + tid_0);
                    int rowc_1 = ((row_1 < n_rows) ? row_1 : last_row_1);
                    float xr_1 = param_partial[cbase_1 + (long long)rowc_1];
                    float add_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(hh[k_6]), "f"(((row_1 < n_rows) ? xr_1 : 0.0f)));
                    float sub_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(add_3), "f"(hh[k_6]));
                    float _sub_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(add_3), "f"(sub_3));
                    float _sub_20;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(hh[k_6]), "f"(_sub_21));
                    float _sub_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(((row_1 < n_rows) ? xr_1 : 0.0f)), "f"(sub_3));
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(_sub_20), "f"(_sub_22));
                    float _add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(ll[k_6]), "f"(_add_19));
                    ll[k_6] = _add_18;
                    hh[k_6] = add_3;
                }
            }
        } else {
            #pragma unroll 1
            for (int jb = 0; jb < nj; jb += 32) {
                int last_row_2 = n_rows - 1;
                long long cbase_2 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_7 = 0; k_7 < 1; k_7++) {
                    #pragma unroll
                    for (int u_2 = 0; u_2 < 32; u_2++) {
                        int row_2 = (jb + u_2) * 256 + (k_7 * 256 + tid_0);
                        int rowc_2 = ((row_2 < n_rows) ? row_2 : last_row_2);
                        float xr_2 = param_partial[cbase_2 + (long long)rowc_2];
                        float add_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(hh[k_7]), "f"(((row_2 < n_rows) ? xr_2 : 0.0f)));
                        float sub_4;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_4) : "f"(add_4), "f"(hh[k_7]));
                        float _sub_26;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(add_4), "f"(sub_4));
                        float _sub_25;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(hh[k_7]), "f"(_sub_26));
                        float _sub_27;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(((row_2 < n_rows) ? xr_2 : 0.0f)), "f"(sub_4));
                        float _add_24;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(_sub_25), "f"(_sub_27));
                        float _add_23;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(ll[k_7]), "f"(_add_24));
                        ll[k_7] = _add_23;
                        hh[k_7] = add_4;
                    }
                }
            }
        }
        #pragma unroll
        for (int k_8 = 0; k_8 < 1; k_8++) {
            float oh = __shfl_xor_sync(0xFFFFFFFF, hh[k_8], 16);
            float ol = __shfl_xor_sync(0xFFFFFFFF, ll[k_8], 16);
            float n;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n) : "f"(hh[k_8]), "f"(oh));
            float bp;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp) : "f"(n), "f"(hh[k_8]));
            float ap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap) : "f"(n), "f"(bp));
            float _sub_28;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(hh[k_8]), "f"(ap));
            float _sub_29;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(oh), "f"(bp));
            float e;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_28), "f"(_sub_29));
            float _add_31;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(ll[k_8]), "f"(ol));
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(_add_31), "f"(e));
            ll[k_8] = _add_30;
            hh[k_8] = n;
            float oh_0 = __shfl_xor_sync(0xFFFFFFFF, hh[k_8], 8);
            float ol_0 = __shfl_xor_sync(0xFFFFFFFF, ll[k_8], 8);
            float n_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_0) : "f"(hh[k_8]), "f"(oh_0));
            float bp_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_0) : "f"(n_0), "f"(hh[k_8]));
            float ap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_0) : "f"(n_0), "f"(bp_0));
            float _sub_32;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(hh[k_8]), "f"(ap_0));
            float _sub_33;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(oh_0), "f"(bp_0));
            float e_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_32), "f"(_sub_33));
            float _add_35;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(ll[k_8]), "f"(ol_0));
            float _add_34;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(_add_35), "f"(e_0));
            ll[k_8] = _add_34;
            hh[k_8] = n_0;
            float oh_1 = __shfl_xor_sync(0xFFFFFFFF, hh[k_8], 4);
            float ol_1 = __shfl_xor_sync(0xFFFFFFFF, ll[k_8], 4);
            float n_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_1) : "f"(hh[k_8]), "f"(oh_1));
            float bp_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_1) : "f"(n_1), "f"(hh[k_8]));
            float ap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_1) : "f"(n_1), "f"(bp_1));
            float _sub_36;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(hh[k_8]), "f"(ap_1));
            float _sub_37;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(oh_1), "f"(bp_1));
            float e_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_36), "f"(_sub_37));
            float _add_39;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(ll[k_8]), "f"(ol_1));
            float _add_38;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(_add_39), "f"(e_1));
            ll[k_8] = _add_38;
            hh[k_8] = n_1;
            float oh_2 = __shfl_xor_sync(0xFFFFFFFF, hh[k_8], 2);
            float ol_2 = __shfl_xor_sync(0xFFFFFFFF, ll[k_8], 2);
            float n_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_2) : "f"(hh[k_8]), "f"(oh_2));
            float bp_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_2) : "f"(n_2), "f"(hh[k_8]));
            float ap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_2) : "f"(n_2), "f"(bp_2));
            float _sub_40;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(hh[k_8]), "f"(ap_2));
            float _sub_41;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(oh_2), "f"(bp_2));
            float e_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_40), "f"(_sub_41));
            float _add_43;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(ll[k_8]), "f"(ol_2));
            float _add_42;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_add_43), "f"(e_2));
            ll[k_8] = _add_42;
            hh[k_8] = n_2;
            float oh_3 = __shfl_xor_sync(0xFFFFFFFF, hh[k_8], 1);
            float ol_3 = __shfl_xor_sync(0xFFFFFFFF, ll[k_8], 1);
            float n_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_3) : "f"(hh[k_8]), "f"(oh_3));
            float bp_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_3) : "f"(n_3), "f"(hh[k_8]));
            float ap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_3) : "f"(n_3), "f"(bp_3));
            float _sub_44;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(hh[k_8]), "f"(ap_3));
            float _sub_45;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(oh_3), "f"(bp_3));
            float e_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_44), "f"(_sub_45));
            float _add_47;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(ll[k_8]), "f"(ol_3));
            float _add_46;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(_add_47), "f"(e_3));
            ll[k_8] = _add_46;
            hh[k_8] = n_3;
            if (lane_1 == 0) {
                int vm = (k_8 * 8 + wrp) * 2;
                vw[vm] = hh[k_8];
                vw[vm + 1] = ll[k_8];
            }
        }
        __syncthreads();
        if (tid_0 == 0) {
            float fh[1];
            float fl[1];
            fh[0] = 0.0f;
            fl[0] = 0.0f;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                float add_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(fh[0]), "f"(vw[m * 2]));
                float sub_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_5) : "f"(add_5), "f"(fh[0]));
                float _sub_51;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(add_5), "f"(sub_5));
                float _sub_50;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(fh[0]), "f"(_sub_51));
                float _sub_52;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(vw[m * 2]), "f"(sub_5));
                float _add_49;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_sub_50), "f"(_sub_52));
                float _add_48;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(fl[0]), "f"(_add_49));
                fl[0] = _add_48;
                fh[0] = add_5;
                float _add_53;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(fl[0]), "f"(vw[m * 2 + 1]));
                fl[0] = _add_53;
            }
            long long col64 = (long long)b;
            if (accumulate != 0) {
                if (b < 24) {
                    float cbv = dbase[col64];
                    float add_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(fh[0]), "f"(cbv));
                    float sub_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_6) : "f"(add_6), "f"(fh[0]));
                    float _sub_57;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(add_6), "f"(sub_6));
                    float _sub_56;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(fh[0]), "f"(_sub_57));
                    float _sub_58;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(cbv), "f"(sub_6));
                    float _add_55;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_sub_56), "f"(_sub_58));
                    float _add_54;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(fl[0]), "f"(_add_55));
                    fl[0] = _add_54;
                    fh[0] = add_6;
                } else {
                    float csv = dscale[col64 - 24ll];
                    float add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(fh[0]), "f"(csv));
                    float sub_7;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_7) : "f"(add_7), "f"(fh[0]));
                    float _sub_62;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(add_7), "f"(sub_7));
                    float _sub_61;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(fh[0]), "f"(_sub_62));
                    float _sub_63;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(csv), "f"(sub_7));
                    float _add_60;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_sub_61), "f"(_sub_63));
                    float _add_59;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(fl[0]), "f"(_add_60));
                    fl[0] = _add_59;
                    fh[0] = add_7;
                }
            }
            bool nonfin_1 = (__as_u32(fh[0]) & 2139095040) == 2139095040;
            float _add_64;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(fh[0]), "f"(fl[0]));
            float res = ((nonfin_1) ? fh[0] : _add_64);
            if (b < 24) {
                *(reinterpret_cast<float*>(dbase + col64) + (0)) = res;
            } else {
                *(reinterpret_cast<float*>(dscale + (col64 - 24ll)) + (0)) = res;
            }
        }
    }
}

} // extern "C"
