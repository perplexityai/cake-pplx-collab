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
kernel_cake_mhc_det_train_bwd_reduce__pt64_pair(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_chunks, int accumulate)
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
    float* sh = reinterpret_cast<float*>(smem_raw + 0);
    const int sh_addr = smem + 0;
    float* sl = reinterpret_cast<float*>(smem_raw + 256);
    const int sl_addr = smem + 256;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int b = bid;
    int tid_0 = tid;
    if (b < 1536) {
        long long e0 = (long long)(b * 64 + tid_0) * 4ll;
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
        int n_tiles = (n_chunks + 511) / 512;
        int col = b - 1536;
        long long col64 = (long long)col;
        long long scol64 = col64 - 24ll;
        int lane_1 = tid_0 & 31;
        int wrp = tid_0 >> 5;
        float ph[1];
        ph[0] = 0.0f;
        float pl[1];
        pl[0] = 0.0f;
        #pragma unroll 1
        for (int ti = 0; ti < n_tiles; ti++) {
            int c0 = ti * 512;
            int rem = n_chunks - c0;
            int rows = ((rem < 512) ? rem : 512);
            int last = rows - 1;
            long long c064 = (long long)c0;
            #pragma unroll
            for (int k_5 = 0; k_5 < 8; k_5++) {
                int row = k_5 * 64 + tid_0;
                int inb = (int)(row < rows);
                int rowc = ((inb != 0) ? row : last);
                float xv = param_partial[(c064 + (long long)rowc) * 32ll + col64];
                float add_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(ph[0]), "f"(((inb != 0) ? xv : 0.0f)));
                float sub_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(add_2), "f"(ph[0]));
                float _sub_17;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(add_2), "f"(sub_2));
                float _sub_16;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(ph[0]), "f"(_sub_17));
                float _sub_18;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(((inb != 0) ? xv : 0.0f)), "f"(sub_2));
                float _add_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_sub_16), "f"(_sub_18));
                float _add_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(pl[0]), "f"(_add_15));
                pl[0] = _add_14;
                ph[0] = add_2;
            }
        }
        sh[tid_0] = ph[0];
        sl[tid_0] = pl[0];
        __syncthreads();
        #pragma unroll
        for (int lv = 0; lv < 6; lv++) {
            int half = 64 >> lv + 1;
            if (tid_0 < half) {
                float mh = sh[tid_0];
                float oh = sh[tid_0 + half];
                float mn;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(mn) : "f"(mh), "f"(oh));
                float bp;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp) : "f"(mn), "f"(mh));
                float ap;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap) : "f"(mn), "f"(bp));
                float _sub_19;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(mh), "f"(ap));
                float _sub_20;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(oh), "f"(bp));
                float me;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(me) : "f"(_sub_19), "f"(_sub_20));
                float _add_22;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(sl[tid_0]), "f"(sl[tid_0 + half]));
                float _add_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_22), "f"(me));
                sl[tid_0] = _add_21;
                sh[tid_0] = mn;
            }
            __syncthreads();
        }
        if (tid_0 == 0) {
            float fh[1];
            float fl[1];
            float res[1];
            fh[0] = sh[0];
            fl[0] = sl[0];
            if (accumulate != 0) {
                if (col < 24) {
                    float add_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(fh[0]), "f"(dbase[col64]));
                    float sub_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(add_3), "f"(fh[0]));
                    float _sub_26;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(add_3), "f"(sub_3));
                    float _sub_25;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(fh[0]), "f"(_sub_26));
                    float _sub_27;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(dbase[col64]), "f"(sub_3));
                    float _add_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(_sub_25), "f"(_sub_27));
                    float _add_23;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(fl[0]), "f"(_add_24));
                    fl[0] = _add_23;
                    fh[0] = add_3;
                } else {
                    float add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(fh[0]), "f"(dscale[scol64]));
                    float sub_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_4) : "f"(add_4), "f"(fh[0]));
                    float _sub_31;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(add_4), "f"(sub_4));
                    float _sub_30;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(fh[0]), "f"(_sub_31));
                    float _sub_32;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(dscale[scol64]), "f"(sub_4));
                    float _add_29;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_sub_30), "f"(_sub_32));
                    float _add_28;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(fl[0]), "f"(_add_29));
                    fl[0] = _add_28;
                    fh[0] = add_4;
                }
            }
            bool nonfin_1 = (__as_u32(fh[0]) & 2139095040) == 2139095040;
            float _add_33;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(fh[0]), "f"(fl[0]));
            res[0] = ((nonfin_1) ? fh[0] : _add_33);
            if (col < 24) {
                *(reinterpret_cast<float*>(dbase + col64) + (0)) = res[0];
            } else {
                *(reinterpret_cast<float*>(dscale + scol64) + (0)) = res[0];
            }
        }
    }
}

} // extern "C"
