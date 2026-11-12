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

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_det_loss_e03bef31303e0675324f(float* __restrict__ term, float* __restrict__ loss_acc, float* __restrict__ loss_out, int rows_c, int first_chunk, int last_chunk, int mode, float loss_div)
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
    float* part_s = reinterpret_cast<float*>(smem_raw + 0);
    const int part_s_addr = smem + 0;
    float* part_c = reinterpret_cast<float*>(smem_raw + 128);
    const int part_c_addr = smem + 128;

    // === Task calls (declaration order) ===
    float zero = 0.0f;
    float s = 0.0f;
    float c = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < rows_c; i += 256) {
        float x = term[i];
        float nm_t;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t) : "f"(s), "f"(x));
        float _sub_0;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(s), "f"(nm_t));
        float nm_es;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es) : "f"(_sub_0), "f"(x));
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(x), "f"(nm_t));
        float nm_ex;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex) : "f"(_sub_1), "f"(s));
        float _abs_3;
        asm("abs.f32 %0, %1;" : "=f"(_abs_3) : "f"(s));
        float _abs_4;
        asm("abs.f32 %0, %1;" : "=f"(_abs_4) : "f"(x));
        uint32_t _setp_2_bits;
        asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_2_bits) : "f"(_abs_3), "f"(_abs_4));
        bool _setp_2 = _setp_2_bits != 0;
        float nm_err = ((_setp_2) ? nm_es : nm_ex);
        float _add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(c), "f"(nm_err));
        c = _add_5;
        s = nm_t;
    }
    float wm_s_0 = __shfl_xor_sync(0xFFFFFFFF, s, 16);
    float wm_c_0 = __shfl_xor_sync(0xFFFFFFFF, c, 16);
    float ts_s;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(s), "f"(wm_s_0));
    float ts_bb;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(s));
    float _sub_7;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(ts_s), "f"(ts_bb));
    float _sub_6;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(s), "f"(_sub_7));
    float _sub_8;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(wm_s_0), "f"(ts_bb));
    float ts_e;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_6), "f"(_sub_8));
    float _add_9;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(c), "f"(wm_c_0));
    float pm_c;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c) : "f"(_add_9), "f"(ts_e));
    float ts_s_0;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(ts_s), "f"(pm_c));
    float ts_bb_0;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_0), "f"(ts_s));
    float _sub_11;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(ts_s_0), "f"(ts_bb_0));
    float _sub_10;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(ts_s), "f"(_sub_11));
    float _sub_12;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(pm_c), "f"(ts_bb_0));
    float ts_e_0;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_10), "f"(_sub_12));
    s = ts_s_0;
    c = ts_e_0;
    float wm_s_1 = __shfl_xor_sync(0xFFFFFFFF, s, 8);
    float wm_c_1 = __shfl_xor_sync(0xFFFFFFFF, c, 8);
    float ts_s_1;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(s), "f"(wm_s_1));
    float ts_bb_1;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_1), "f"(s));
    float _sub_14;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(ts_s_1), "f"(ts_bb_1));
    float _sub_13;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(s), "f"(_sub_14));
    float _sub_15;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(wm_s_1), "f"(ts_bb_1));
    float ts_e_1;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_13), "f"(_sub_15));
    float _add_16;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(c), "f"(wm_c_1));
    float pm_c_0;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_0) : "f"(_add_16), "f"(ts_e_1));
    float ts_s_2;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(ts_s_1), "f"(pm_c_0));
    float ts_bb_2;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_2), "f"(ts_s_1));
    float _sub_18;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(ts_s_2), "f"(ts_bb_2));
    float _sub_17;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(ts_s_1), "f"(_sub_18));
    float _sub_19;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(pm_c_0), "f"(ts_bb_2));
    float ts_e_2;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_17), "f"(_sub_19));
    s = ts_s_2;
    c = ts_e_2;
    float wm_s_2 = __shfl_xor_sync(0xFFFFFFFF, s, 4);
    float wm_c_2 = __shfl_xor_sync(0xFFFFFFFF, c, 4);
    float ts_s_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3) : "f"(s), "f"(wm_s_2));
    float ts_bb_3;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3) : "f"(ts_s_3), "f"(s));
    float _sub_21;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(ts_s_3), "f"(ts_bb_3));
    float _sub_20;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(s), "f"(_sub_21));
    float _sub_22;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(wm_s_2), "f"(ts_bb_3));
    float ts_e_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_20), "f"(_sub_22));
    float _add_23;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(c), "f"(wm_c_2));
    float pm_c_1;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_1) : "f"(_add_23), "f"(ts_e_3));
    float ts_s_4;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_4) : "f"(ts_s_3), "f"(pm_c_1));
    float ts_bb_4;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_4) : "f"(ts_s_4), "f"(ts_s_3));
    float _sub_25;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(ts_s_4), "f"(ts_bb_4));
    float _sub_24;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(ts_s_3), "f"(_sub_25));
    float _sub_26;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(pm_c_1), "f"(ts_bb_4));
    float ts_e_4;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_24), "f"(_sub_26));
    s = ts_s_4;
    c = ts_e_4;
    float wm_s_3 = __shfl_xor_sync(0xFFFFFFFF, s, 2);
    float wm_c_3 = __shfl_xor_sync(0xFFFFFFFF, c, 2);
    float ts_s_5;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_5) : "f"(s), "f"(wm_s_3));
    float ts_bb_5;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_5) : "f"(ts_s_5), "f"(s));
    float _sub_28;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(ts_s_5), "f"(ts_bb_5));
    float _sub_27;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(s), "f"(_sub_28));
    float _sub_29;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(wm_s_3), "f"(ts_bb_5));
    float ts_e_5;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_5) : "f"(_sub_27), "f"(_sub_29));
    float _add_30;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(c), "f"(wm_c_3));
    float pm_c_2;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_2) : "f"(_add_30), "f"(ts_e_5));
    float ts_s_6;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_6) : "f"(ts_s_5), "f"(pm_c_2));
    float ts_bb_6;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6) : "f"(ts_s_6), "f"(ts_s_5));
    float _sub_32;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(ts_s_6), "f"(ts_bb_6));
    float _sub_31;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(ts_s_5), "f"(_sub_32));
    float _sub_33;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(pm_c_2), "f"(ts_bb_6));
    float ts_e_6;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6) : "f"(_sub_31), "f"(_sub_33));
    s = ts_s_6;
    c = ts_e_6;
    float wm_s_4 = __shfl_xor_sync(0xFFFFFFFF, s, 1);
    float wm_c_4 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
    float ts_s_7;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7) : "f"(s), "f"(wm_s_4));
    float ts_bb_7;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7) : "f"(ts_s_7), "f"(s));
    float _sub_35;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(ts_s_7), "f"(ts_bb_7));
    float _sub_34;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(s), "f"(_sub_35));
    float _sub_36;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(wm_s_4), "f"(ts_bb_7));
    float ts_e_7;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7) : "f"(_sub_34), "f"(_sub_36));
    float _add_37;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(c), "f"(wm_c_4));
    float pm_c_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_3) : "f"(_add_37), "f"(ts_e_7));
    float ts_s_8;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8) : "f"(ts_s_7), "f"(pm_c_3));
    float ts_bb_8;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8) : "f"(ts_s_8), "f"(ts_s_7));
    float _sub_39;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(ts_s_8), "f"(ts_bb_8));
    float _sub_38;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(ts_s_7), "f"(_sub_39));
    float _sub_40;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(pm_c_3), "f"(ts_bb_8));
    float ts_e_8;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8) : "f"(_sub_38), "f"(_sub_40));
    s = ts_s_8;
    c = ts_e_8;
    if (lane == 0) {
        part_s[warp] = s;
        part_c[warp] = c;
    }
    __syncthreads();
    if (warp == 0) {
        float p = ((lane < 8) ? part_s[lane] : zero);
        float q = ((lane < 8) ? part_c[lane] : zero);
        float wm_s_5 = __shfl_xor_sync(0xFFFFFFFF, p, 16);
        float wm_c_5 = __shfl_xor_sync(0xFFFFFFFF, q, 16);
        float ts_s_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_9) : "f"(p), "f"(wm_s_5));
        float ts_bb_9;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_9) : "f"(ts_s_9), "f"(p));
        float _sub_42;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(ts_s_9), "f"(ts_bb_9));
        float _sub_41;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(p), "f"(_sub_42));
        float _sub_43;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(wm_s_5), "f"(ts_bb_9));
        float ts_e_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9) : "f"(_sub_41), "f"(_sub_43));
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(q), "f"(wm_c_5));
        float pm_c_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_4) : "f"(_add_44), "f"(ts_e_9));
        float ts_s_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_10) : "f"(ts_s_9), "f"(pm_c_4));
        float ts_bb_10;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_10) : "f"(ts_s_10), "f"(ts_s_9));
        float _sub_46;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(ts_s_10), "f"(ts_bb_10));
        float _sub_45;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(ts_s_9), "f"(_sub_46));
        float _sub_47;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(pm_c_4), "f"(ts_bb_10));
        float ts_e_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_10) : "f"(_sub_45), "f"(_sub_47));
        p = ts_s_10;
        q = ts_e_10;
        float wm_s_6 = __shfl_xor_sync(0xFFFFFFFF, p, 8);
        float wm_c_6 = __shfl_xor_sync(0xFFFFFFFF, q, 8);
        float ts_s_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_11) : "f"(p), "f"(wm_s_6));
        float ts_bb_11;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_11) : "f"(ts_s_11), "f"(p));
        float _sub_49;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(ts_s_11), "f"(ts_bb_11));
        float _sub_48;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(p), "f"(_sub_49));
        float _sub_50;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(wm_s_6), "f"(ts_bb_11));
        float ts_e_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_11) : "f"(_sub_48), "f"(_sub_50));
        float _add_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(q), "f"(wm_c_6));
        float pm_c_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_5) : "f"(_add_51), "f"(ts_e_11));
        float ts_s_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_12) : "f"(ts_s_11), "f"(pm_c_5));
        float ts_bb_12;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_12) : "f"(ts_s_12), "f"(ts_s_11));
        float _sub_53;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(ts_s_12), "f"(ts_bb_12));
        float _sub_52;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(ts_s_11), "f"(_sub_53));
        float _sub_54;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(pm_c_5), "f"(ts_bb_12));
        float ts_e_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_12) : "f"(_sub_52), "f"(_sub_54));
        p = ts_s_12;
        q = ts_e_12;
        float wm_s_7 = __shfl_xor_sync(0xFFFFFFFF, p, 4);
        float wm_c_7 = __shfl_xor_sync(0xFFFFFFFF, q, 4);
        float ts_s_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_13) : "f"(p), "f"(wm_s_7));
        float ts_bb_13;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_13) : "f"(ts_s_13), "f"(p));
        float _sub_56;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(ts_s_13), "f"(ts_bb_13));
        float _sub_55;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(p), "f"(_sub_56));
        float _sub_57;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(wm_s_7), "f"(ts_bb_13));
        float ts_e_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_13) : "f"(_sub_55), "f"(_sub_57));
        float _add_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(q), "f"(wm_c_7));
        float pm_c_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_6) : "f"(_add_58), "f"(ts_e_13));
        float ts_s_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_14) : "f"(ts_s_13), "f"(pm_c_6));
        float ts_bb_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_14) : "f"(ts_s_14), "f"(ts_s_13));
        float _sub_60;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(ts_s_14), "f"(ts_bb_14));
        float _sub_59;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(ts_s_13), "f"(_sub_60));
        float _sub_61;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(pm_c_6), "f"(ts_bb_14));
        float ts_e_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_14) : "f"(_sub_59), "f"(_sub_61));
        p = ts_s_14;
        q = ts_e_14;
        float wm_s_8 = __shfl_xor_sync(0xFFFFFFFF, p, 2);
        float wm_c_8 = __shfl_xor_sync(0xFFFFFFFF, q, 2);
        float ts_s_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_15) : "f"(p), "f"(wm_s_8));
        float ts_bb_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_15) : "f"(ts_s_15), "f"(p));
        float _sub_63;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(ts_s_15), "f"(ts_bb_15));
        float _sub_62;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(p), "f"(_sub_63));
        float _sub_64;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(wm_s_8), "f"(ts_bb_15));
        float ts_e_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_15) : "f"(_sub_62), "f"(_sub_64));
        float _add_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(q), "f"(wm_c_8));
        float pm_c_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_7) : "f"(_add_65), "f"(ts_e_15));
        float ts_s_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_16) : "f"(ts_s_15), "f"(pm_c_7));
        float ts_bb_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_16) : "f"(ts_s_16), "f"(ts_s_15));
        float _sub_67;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(ts_s_16), "f"(ts_bb_16));
        float _sub_66;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(ts_s_15), "f"(_sub_67));
        float _sub_68;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(pm_c_7), "f"(ts_bb_16));
        float ts_e_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_16) : "f"(_sub_66), "f"(_sub_68));
        p = ts_s_16;
        q = ts_e_16;
        float wm_s_9 = __shfl_xor_sync(0xFFFFFFFF, p, 1);
        float wm_c_9 = __shfl_xor_sync(0xFFFFFFFF, q, 1);
        float ts_s_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_17) : "f"(p), "f"(wm_s_9));
        float ts_bb_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_17) : "f"(ts_s_17), "f"(p));
        float _sub_70;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(ts_s_17), "f"(ts_bb_17));
        float _sub_69;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(p), "f"(_sub_70));
        float _sub_71;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(wm_s_9), "f"(ts_bb_17));
        float ts_e_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_17) : "f"(_sub_69), "f"(_sub_71));
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(q), "f"(wm_c_9));
        float pm_c_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_8) : "f"(_add_72), "f"(ts_e_17));
        float ts_s_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_18) : "f"(ts_s_17), "f"(pm_c_8));
        float ts_bb_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_18) : "f"(ts_s_18), "f"(ts_s_17));
        float _sub_74;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(ts_s_18), "f"(ts_bb_18));
        float _sub_73;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(ts_s_17), "f"(_sub_74));
        float _sub_75;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(pm_c_8), "f"(ts_bb_18));
        float ts_e_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_18) : "f"(_sub_73), "f"(_sub_75));
        p = ts_s_18;
        q = ts_e_18;
        if (lane == 0) {
            if (first_chunk == 0) {
                float hi = loss_acc[0];
                float lo = loss_acc[1];
                float ts_s_19;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_19) : "f"(p), "f"(hi));
                float ts_bb_19;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_19) : "f"(ts_s_19), "f"(p));
                float _sub_77;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(ts_s_19), "f"(ts_bb_19));
                float _sub_76;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(p), "f"(_sub_77));
                float _sub_78;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(hi), "f"(ts_bb_19));
                float ts_e_19;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_19) : "f"(_sub_76), "f"(_sub_78));
                float _add_79;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(q), "f"(lo));
                float pm_c_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_9) : "f"(_add_79), "f"(ts_e_19));
                float ts_s_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_20) : "f"(ts_s_19), "f"(pm_c_9));
                float ts_bb_20;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_20) : "f"(ts_s_20), "f"(ts_s_19));
                float _sub_81;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(ts_s_20), "f"(ts_bb_20));
                float _sub_80;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(ts_s_19), "f"(_sub_81));
                float _sub_82;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(pm_c_9), "f"(ts_bb_20));
                float ts_e_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_20) : "f"(_sub_80), "f"(_sub_82));
                p = ts_s_20;
                q = ts_e_20;
            }
            loss_acc[0] = p;
            loss_acc[1] = q;
            if (last_chunk != 0) {
                float fin;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(fin) : "f"(p), "f"(q));
                if (mode == 0) {
                    float pd_q = __fdiv_rn(p, loss_div);
                    float _neg_83;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_83) : "f"(pd_q));
                    float pd_r;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pd_r) : "f"(_neg_83), "f"(loss_div), "f"(p));
                    float _add_84;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(pd_r), "f"(q));
                    float pd_e = __fdiv_rn(_add_84, loss_div);
                    float quo;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(quo) : "f"(pd_q), "f"(pd_e));
                    fin = quo;
                }
                float _sub_85;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(zero), "f"(fin));
                loss_out[0] = _sub_85;
            }
        }
    }
}

} // extern "C"
