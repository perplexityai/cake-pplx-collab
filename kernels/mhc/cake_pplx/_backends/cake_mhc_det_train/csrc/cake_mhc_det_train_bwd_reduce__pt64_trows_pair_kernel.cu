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

__global__ __launch_bounds__(64) void
kernel_cake_mhc_det_train_bwd_reduce__pt64_trows_pair(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_rows, int accumulate)
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
        long long e0 = (long long)((b - 27) * 64 + tid_0) * 4ll;
        float out[4];
        float hi[4];
        float lo[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            hi[k] = 0.0f;
            lo[k] = 0.0f;
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
            hi[k_1] = ((_vec_load_0[k_1] != 0.0f) ? _vec_load_0[k_1] : 0.0f);
        }
        #pragma unroll 1
        for (int p = 1; p < n_parts; p++) {
            float vec_load[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + ((long long)p * 393216ll + e0) + 0);
                vec_load[0 + 0] = _v4.x;
                vec_load[0 + 1] = _v4.y;
                vec_load[0 + 2] = _v4.z;
                vec_load[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; k_2++) {
                float add;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(hi[k_2]), "f"(vec_load[k_2]));
                float sub;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(add), "f"(hi[k_2]));
                float _sub_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(add), "f"(sub));
                float _sub_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(hi[k_2]), "f"(_sub_5));
                float _sub_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(vec_load[k_2]), "f"(sub));
                float _add_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_sub_4), "f"(_sub_6));
                float _add_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(lo[k_2]), "f"(_add_3));
                lo[k_2] = _add_2;
                hi[k_2] = add;
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
                float add_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(hi[k_3]), "f"(_vec_load_2[k_3]));
                float sub_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(add_1), "f"(hi[k_3]));
                float _sub_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(add_1), "f"(sub_1));
                float _sub_10;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(hi[k_3]), "f"(_sub_11));
                float _sub_12;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(_vec_load_2[k_3]), "f"(sub_1));
                float _add_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(_sub_10), "f"(_sub_12));
                float _add_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(lo[k_3]), "f"(_add_9));
                lo[k_3] = _add_8;
                hi[k_3] = add_1;
            }
        }
        #pragma unroll
        for (int k_4 = 0; k_4 < 4; k_4++) {
            bool nonfin = (__as_u32(hi[k_4]) & 2139095040) == 2139095040;
            float _add_13;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(hi[k_4]), "f"(lo[k_4]));
            out[k_4] = ((nonfin) ? hi[k_4] : _add_13);
        }
        {
            float4 _v4 = make_float4(out[0 + 0], out[0 + 1], out[0 + 2], out[0 + 3]);
            *reinterpret_cast<float4*>((dfn + e0) + 0) = _v4;
        }
    } else {
        int lane_1 = tid_0 & 31;
        int wrp = tid_0 >> 5;
        float hh[4];
        float ll[4];
        #pragma unroll
        for (int k_5 = 0; k_5 < 4; k_5++) {
            hh[k_5] = 0.0f;
            ll[k_5] = 0.0f;
        }
        int nj = n_rows + 255 >> 8;
        #pragma unroll 1
        for (int jb = 0; jb < nj; jb += 8) {
            int last_row = n_rows - 1;
            long long cbase = (long long)b * (long long)n_rows;
            #pragma unroll
            for (int k_6 = 0; k_6 < 4; k_6++) {
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    int row = (jb + u) * 256 + (k_6 * 64 + tid_0);
                    int rowc = ((row < n_rows) ? row : last_row);
                    float xr = param_partial[cbase + (long long)rowc];
                    float add_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(hh[k_6]), "f"(((row < n_rows) ? xr : 0.0f)));
                    float sub_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(add_2), "f"(hh[k_6]));
                    float _sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(add_2), "f"(sub_2));
                    float _sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(hh[k_6]), "f"(_sub_17));
                    float _sub_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(((row < n_rows) ? xr : 0.0f)), "f"(sub_2));
                    float _add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_sub_16), "f"(_sub_18));
                    float _add_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(ll[k_6]), "f"(_add_15));
                    ll[k_6] = _add_14;
                    hh[k_6] = add_2;
                }
            }
        }
        #pragma unroll
        for (int k_7 = 0; k_7 < 4; k_7++) {
            float oh = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 16);
            float ol = __shfl_xor_sync(0xFFFFFFFF, ll[k_7], 16);
            float n;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n) : "f"(hh[k_7]), "f"(oh));
            float bp;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp) : "f"(n), "f"(hh[k_7]));
            float ap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap) : "f"(n), "f"(bp));
            float _sub_19;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(hh[k_7]), "f"(ap));
            float _sub_20;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(oh), "f"(bp));
            float e;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_19), "f"(_sub_20));
            float _add_22;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(ll[k_7]), "f"(ol));
            float _add_21;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_22), "f"(e));
            ll[k_7] = _add_21;
            hh[k_7] = n;
            float oh_0 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 8);
            float ol_0 = __shfl_xor_sync(0xFFFFFFFF, ll[k_7], 8);
            float n_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_0) : "f"(hh[k_7]), "f"(oh_0));
            float bp_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_0) : "f"(n_0), "f"(hh[k_7]));
            float ap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_0) : "f"(n_0), "f"(bp_0));
            float _sub_23;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(hh[k_7]), "f"(ap_0));
            float _sub_24;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(oh_0), "f"(bp_0));
            float e_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_23), "f"(_sub_24));
            float _add_26;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(ll[k_7]), "f"(ol_0));
            float _add_25;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_add_26), "f"(e_0));
            ll[k_7] = _add_25;
            hh[k_7] = n_0;
            float oh_1 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 4);
            float ol_1 = __shfl_xor_sync(0xFFFFFFFF, ll[k_7], 4);
            float n_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_1) : "f"(hh[k_7]), "f"(oh_1));
            float bp_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_1) : "f"(n_1), "f"(hh[k_7]));
            float ap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_1) : "f"(n_1), "f"(bp_1));
            float _sub_27;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(hh[k_7]), "f"(ap_1));
            float _sub_28;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(oh_1), "f"(bp_1));
            float e_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_27), "f"(_sub_28));
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(ll[k_7]), "f"(ol_1));
            float _add_29;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_30), "f"(e_1));
            ll[k_7] = _add_29;
            hh[k_7] = n_1;
            float oh_2 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 2);
            float ol_2 = __shfl_xor_sync(0xFFFFFFFF, ll[k_7], 2);
            float n_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_2) : "f"(hh[k_7]), "f"(oh_2));
            float bp_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_2) : "f"(n_2), "f"(hh[k_7]));
            float ap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_2) : "f"(n_2), "f"(bp_2));
            float _sub_31;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(hh[k_7]), "f"(ap_2));
            float _sub_32;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(oh_2), "f"(bp_2));
            float e_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_31), "f"(_sub_32));
            float _add_34;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(ll[k_7]), "f"(ol_2));
            float _add_33;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_add_34), "f"(e_2));
            ll[k_7] = _add_33;
            hh[k_7] = n_2;
            float oh_3 = __shfl_xor_sync(0xFFFFFFFF, hh[k_7], 1);
            float ol_3 = __shfl_xor_sync(0xFFFFFFFF, ll[k_7], 1);
            float n_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_3) : "f"(hh[k_7]), "f"(oh_3));
            float bp_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_3) : "f"(n_3), "f"(hh[k_7]));
            float ap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_3) : "f"(n_3), "f"(bp_3));
            float _sub_35;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(hh[k_7]), "f"(ap_3));
            float _sub_36;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(oh_3), "f"(bp_3));
            float e_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_35), "f"(_sub_36));
            float _add_38;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(ll[k_7]), "f"(ol_3));
            float _add_37;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_38), "f"(e_3));
            ll[k_7] = _add_37;
            hh[k_7] = n_3;
            if (lane_1 == 0) {
                int vm = (k_7 * 2 + wrp) * 2;
                vw[vm] = hh[k_7];
                vw[vm + 1] = ll[k_7];
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
                float add_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(fh[0]), "f"(vw[m * 2]));
                float sub_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(add_3), "f"(fh[0]));
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(add_3), "f"(sub_3));
                float _sub_41;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(fh[0]), "f"(_sub_42));
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(vw[m * 2]), "f"(sub_3));
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_sub_41), "f"(_sub_43));
                float _add_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(fl[0]), "f"(_add_40));
                fl[0] = _add_39;
                fh[0] = add_3;
                float _add_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(fl[0]), "f"(vw[m * 2 + 1]));
                fl[0] = _add_44;
            }
            long long col64 = (long long)b;
            if (accumulate != 0) {
                if (b < 24) {
                    float cbv = dbase[col64];
                    float add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(fh[0]), "f"(cbv));
                    float sub_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_4) : "f"(add_4), "f"(fh[0]));
                    float _sub_48;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(add_4), "f"(sub_4));
                    float _sub_47;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(fh[0]), "f"(_sub_48));
                    float _sub_49;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(cbv), "f"(sub_4));
                    float _add_46;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(_sub_47), "f"(_sub_49));
                    float _add_45;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(fl[0]), "f"(_add_46));
                    fl[0] = _add_45;
                    fh[0] = add_4;
                } else {
                    float csv = dscale[col64 - 24ll];
                    float add_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(fh[0]), "f"(csv));
                    float sub_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_5) : "f"(add_5), "f"(fh[0]));
                    float _sub_53;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(add_5), "f"(sub_5));
                    float _sub_52;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(fh[0]), "f"(_sub_53));
                    float _sub_54;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(csv), "f"(sub_5));
                    float _add_51;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(_sub_52), "f"(_sub_54));
                    float _add_50;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(fl[0]), "f"(_add_51));
                    fl[0] = _add_50;
                    fh[0] = add_5;
                }
            }
            bool nonfin_1 = (__as_u32(fh[0]) & 2139095040) == 2139095040;
            float _add_55;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(fh[0]), "f"(fl[0]));
            float res = ((nonfin_1) ? fh[0] : _add_55);
            if (b < 24) {
                *(reinterpret_cast<float*>(dbase + col64) + (0)) = res;
            } else {
                *(reinterpret_cast<float*>(dscale + (col64 - 24ll)) + (0)) = res;
            }
        }
    }
}

} // extern "C"
