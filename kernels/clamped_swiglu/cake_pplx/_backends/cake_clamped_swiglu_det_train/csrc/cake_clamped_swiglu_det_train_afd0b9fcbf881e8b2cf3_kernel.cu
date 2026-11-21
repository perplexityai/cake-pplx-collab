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

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_clamped_swiglu_det_train_afd0b9fcbf881e8b2cf3(__nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ up, float* __restrict__ w, __nv_bfloat16* __restrict__ h, int rows, long long gate_stride, long long up_stride, long long h_stride, float limit, int weighted)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int lane_1 = tid & 31;
    long long warp_1 = tid >> 5;
    long long tasks = (long long)rows * 9ll;
    long long nblocks = (tasks + 3ll) / 4ll;
    #pragma unroll 1
    for (long long blk = (long long)bid; blk < nblocks; blk += (long long)num_bids) {
        long long task = blk * 4ll + warp_1;
        long long _min_0 = ((task / 9ll) < ((long long)rows) ? (task / 9ll) : ((long long)rows));
        int row = (int)_min_0;
        int run = (int)(task % 9ll);
        int src = ((row) < (rows - 1) ? (row) : (rows - 1));
        long long goff = (long long)src * gate_stride;
        long long uoff = (long long)src * up_stride;
        long long hoff = (long long)row * h_stride;
        float wv = w[src * weighted];
        float wr = ((weighted != 0) ? wv : 1.0f);
        float2 wr2 = make_float2(wr, wr);
        unsigned int gw[4];
        unsigned int uw[4];
        long long col_0 = (long long)((run * 32 + lane_1) * 8);
        {
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(gate + (goff + col_0) + 0);
            gw[0 + 0] = _uv4_1.x;
            gw[0 + 1] = _uv4_1.y;
            gw[0 + 2] = _uv4_1.z;
            gw[0 + 3] = _uv4_1.w;
        }
        {
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(up + (uoff + col_0) + 0);
            uw[0 + 0] = _uv4_2.x;
            uw[0 + 1] = _uv4_2.y;
            uw[0 + 2] = _uv4_2.z;
            uw[0 + 3] = _uv4_2.w;
        }
        unsigned int hw[4];
        __nv_bfloat162 lowest_0 = __as_bf16x2(gw[0]);
        lowest_0 = __hmin2_nan(lowest_0, __as_bf16x2(gw[1]));
        lowest_0 = __hmin2_nan(lowest_0, __as_bf16x2(gw[2]));
        lowest_0 = __hmin2_nan(lowest_0, __as_bf16x2(gw[3]));
        float2 pair = cake_bf16x2_to_f32x2(lowest_0);
        float lowest;
        asm("min.NaN.f32 %0, %1, %2;" : "=f"(lowest) : "f"(pair.x), "f"(pair.y));
        float _min_4;
        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_4) : "f"(lowest), "f"(limit));
        int _warp_vote_3 = __all_sync(0xFFFFFFFF, _min_4 > -87.0f);
        if (_warp_vote_3 != 0) {
            float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
            float2 v_0 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
            float neg;
            asm("neg.f32 %0, %1;" : "=f"(neg) : "f"(limit));
            float _min_5;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_5) : "f"(x_0.x), "f"(limit));
            float _min_6;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_6) : "f"(x_0.y), "f"(limit));
            float2 gc = make_float2(_min_5, _min_6);
            float _min_8;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_8) : "f"(v_0.x), "f"(limit));
            float _max_7;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(_min_8), "f"(neg));
            float _min_10;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_10) : "f"(v_0.y), "f"(limit));
            float _max_9;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(_min_10), "f"(neg));
            float2 uc = make_float2(_max_7, _max_9);
            float2 _f32x2_mul_b_11 = make_float2(-1.442695f, -1.442695f);
            float2 t;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&gc), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
            float e0 = approx_exp2(t.x);
            float e1 = approx_exp2(t.y);
            float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
            float _rcp_12 = approx_rcp(-nd.x);
            float _rcp_13 = approx_rcp(-nd.y);
            float2 r = make_float2(_rcp_12, _rcp_13);
            float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
            hw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, mul_f32x2_noftz(mul_f32x2_noftz(gc, s), uc))));
            float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
            float2 v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
            float neg_0;
            asm("neg.f32 %0, %1;" : "=f"(neg_0) : "f"(limit));
            float _min_14;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_14) : "f"(x_1.x), "f"(limit));
            float _min_15;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_15) : "f"(x_1.y), "f"(limit));
            float2 gc_0 = make_float2(_min_14, _min_15);
            float _min_17;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_17) : "f"(v_1.x), "f"(limit));
            float _max_16;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(_min_17), "f"(neg_0));
            float _min_19;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_19) : "f"(v_1.y), "f"(limit));
            float _max_18;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_18) : "f"(_min_19), "f"(neg_0));
            float2 uc_0 = make_float2(_max_16, _max_18);
            float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
            float2 t_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&gc_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
            float e0_0 = approx_exp2(t_0.x);
            float e1_0 = approx_exp2(t_0.y);
            float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
            float _rcp_21 = approx_rcp(-nd_0.x);
            float _rcp_22 = approx_rcp(-nd_0.y);
            float2 r_0 = make_float2(_rcp_21, _rcp_22);
            float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
            hw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, mul_f32x2_noftz(mul_f32x2_noftz(gc_0, s_0), uc_0))));
            float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
            float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
            float neg_1;
            asm("neg.f32 %0, %1;" : "=f"(neg_1) : "f"(limit));
            float _min_23;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_23) : "f"(x_2.x), "f"(limit));
            float _min_24;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_24) : "f"(x_2.y), "f"(limit));
            float2 gc_1 = make_float2(_min_23, _min_24);
            float _min_26;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_26) : "f"(v_2.x), "f"(limit));
            float _max_25;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_25) : "f"(_min_26), "f"(neg_1));
            float _min_28;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_28) : "f"(v_2.y), "f"(limit));
            float _max_27;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_27) : "f"(_min_28), "f"(neg_1));
            float2 uc_1 = make_float2(_max_25, _max_27);
            float2 _f32x2_mul_b_29 = make_float2(-1.442695f, -1.442695f);
            float2 t_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&gc_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
            float e0_1 = approx_exp2(t_1.x);
            float e1_1 = approx_exp2(t_1.y);
            float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
            float _rcp_30 = approx_rcp(-nd_1.x);
            float _rcp_31 = approx_rcp(-nd_1.y);
            float2 r_1 = make_float2(_rcp_30, _rcp_31);
            float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
            hw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, mul_f32x2_noftz(mul_f32x2_noftz(gc_1, s_1), uc_1))));
            float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
            float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
            float neg_2;
            asm("neg.f32 %0, %1;" : "=f"(neg_2) : "f"(limit));
            float _min_32;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_32) : "f"(x_3.x), "f"(limit));
            float _min_33;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_33) : "f"(x_3.y), "f"(limit));
            float2 gc_2 = make_float2(_min_32, _min_33);
            float _min_35;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_35) : "f"(v_3.x), "f"(limit));
            float _max_34;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_34) : "f"(_min_35), "f"(neg_2));
            float _min_37;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_37) : "f"(v_3.y), "f"(limit));
            float _max_36;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_36) : "f"(_min_37), "f"(neg_2));
            float2 uc_2 = make_float2(_max_34, _max_36);
            float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
            float2 t_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&gc_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
            float e0_2 = approx_exp2(t_2.x);
            float e1_2 = approx_exp2(t_2.y);
            float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
            float _rcp_39 = approx_rcp(-nd_2.x);
            float _rcp_40 = approx_rcp(-nd_2.y);
            float2 r_2 = make_float2(_rcp_39, _rcp_40);
            float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
            hw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, mul_f32x2_noftz(mul_f32x2_noftz(gc_2, s_2), uc_2))));
            if (row < rows) {
                long long col_1 = (long long)((run * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(h + (hoff + col_1))[0] = reinterpret_cast<int4*>(hw)[0];
            }
        } else {
            float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
            float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
            float neg_3;
            asm("neg.f32 %0, %1;" : "=f"(neg_3) : "f"(limit));
            float _min_41;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_41) : "f"(x_4.x), "f"(limit));
            float _min_42;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_42) : "f"(x_4.y), "f"(limit));
            float2 gc_3 = make_float2(_min_41, _min_42);
            float _min_44;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_44) : "f"(v_4.x), "f"(limit));
            float _max_43;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_43) : "f"(_min_44), "f"(neg_3));
            float _min_46;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_46) : "f"(v_4.y), "f"(limit));
            float _max_45;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_45) : "f"(_min_46), "f"(neg_3));
            float2 uc_3 = make_float2(_max_43, _max_45);
            float2 _f32x2_mul_b_47 = make_float2(-1.442695f, -1.442695f);
            float2 t_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&gc_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
            float _sub_48;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(t_3.x), "f"(126.0f));
            float u = ((t_3.x > 126.0f) ? _sub_48 : t_3.x);
            float z = approx_exp2(u);
            float _fma_49;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(z), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn = ((t_3.x > 126.0f) ? z : _fma_49);
            float r_3 = approx_rcp(dn);
            float _fma_50;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(-dn), "f"(r_3), "f"(1.0f));
            float q;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q) : "f"(r_3), "f"(_fma_50), "f"(r_3));
            float _sub_51;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(t_3.y), "f"(126.0f));
            float u_0 = ((t_3.y > 126.0f) ? _sub_51 : t_3.y);
            float z_0 = approx_exp2(u_0);
            float _fma_52;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(z_0), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_0 = ((t_3.y > 126.0f) ? z_0 : _fma_52);
            float r_3_1 = approx_rcp(dn_0);
            float _fma_53;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(-dn_0), "f"(r_3_1), "f"(1.0f));
            float q_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_0) : "f"(r_3_1), "f"(_fma_53), "f"(r_3_1));
            float _fma_54;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(q), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_55;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(q_0), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_3 = make_float2(_fma_54, _fma_55);
            hw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, mul_f32x2_noftz(mul_f32x2_noftz(gc_3, s_3), uc_3))));
            float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
            float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
            float neg_3_1;
            asm("neg.f32 %0, %1;" : "=f"(neg_3_1) : "f"(limit));
            float _min_56;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_56) : "f"(x_5.x), "f"(limit));
            float _min_57;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_57) : "f"(x_5.y), "f"(limit));
            float2 gc_3_1 = make_float2(_min_56, _min_57);
            float _min_59;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_59) : "f"(v_5.x), "f"(limit));
            float _max_58;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_58) : "f"(_min_59), "f"(neg_3_1));
            float _min_61;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_61) : "f"(v_5.y), "f"(limit));
            float _max_60;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_60) : "f"(_min_61), "f"(neg_3_1));
            float2 uc_3_1 = make_float2(_max_58, _max_60);
            float2 _f32x2_mul_b_62 = make_float2(-1.442695f, -1.442695f);
            float2 t_3_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3_1) : "l"(*(const unsigned long long*)&gc_3_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
            float _sub_63;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(t_3_1.x), "f"(126.0f));
            float u_1 = ((t_3_1.x > 126.0f) ? _sub_63 : t_3_1.x);
            float z_1 = approx_exp2(u_1);
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(z_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_1 = ((t_3_1.x > 126.0f) ? z_1 : _fma_64);
            float r_4 = approx_rcp(dn_1);
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-dn_1), "f"(r_4), "f"(1.0f));
            float q_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_1) : "f"(r_4), "f"(_fma_65), "f"(r_4));
            float _sub_66;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(t_3_1.y), "f"(126.0f));
            float u_2 = ((t_3_1.y > 126.0f) ? _sub_66 : t_3_1.y);
            float z_2 = approx_exp2(u_2);
            float _fma_67;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(z_2), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_2 = ((t_3_1.y > 126.0f) ? z_2 : _fma_67);
            float r_5 = approx_rcp(dn_2);
            float _fma_68;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(-dn_2), "f"(r_5), "f"(1.0f));
            float q_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_2) : "f"(r_5), "f"(_fma_68), "f"(r_5));
            float _fma_69;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(q_1), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_70;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(q_2), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_3_1 = make_float2(_fma_69, _fma_70);
            hw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, mul_f32x2_noftz(mul_f32x2_noftz(gc_3_1, s_3_1), uc_3_1))));
            float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
            float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
            float neg_4;
            asm("neg.f32 %0, %1;" : "=f"(neg_4) : "f"(limit));
            float _min_71;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_71) : "f"(x_6.x), "f"(limit));
            float _min_72;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_72) : "f"(x_6.y), "f"(limit));
            float2 gc_4 = make_float2(_min_71, _min_72);
            float _min_74;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_74) : "f"(v_6.x), "f"(limit));
            float _max_73;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_73) : "f"(_min_74), "f"(neg_4));
            float _min_76;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_76) : "f"(v_6.y), "f"(limit));
            float _max_75;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_75) : "f"(_min_76), "f"(neg_4));
            float2 uc_4 = make_float2(_max_73, _max_75);
            float2 _f32x2_mul_b_77 = make_float2(-1.442695f, -1.442695f);
            float2 t_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&gc_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
            float _sub_78;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(t_4.x), "f"(126.0f));
            float u_3 = ((t_4.x > 126.0f) ? _sub_78 : t_4.x);
            float z_3 = approx_exp2(u_3);
            float _fma_79;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(z_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_3 = ((t_4.x > 126.0f) ? z_3 : _fma_79);
            float r_6 = approx_rcp(dn_3);
            float _fma_80;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(-dn_3), "f"(r_6), "f"(1.0f));
            float q_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_3) : "f"(r_6), "f"(_fma_80), "f"(r_6));
            float _sub_81;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(t_4.y), "f"(126.0f));
            float u_4 = ((t_4.y > 126.0f) ? _sub_81 : t_4.y);
            float z_4 = approx_exp2(u_4);
            float _fma_82;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(z_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_4 = ((t_4.y > 126.0f) ? z_4 : _fma_82);
            float r_7 = approx_rcp(dn_4);
            float _fma_83;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(-dn_4), "f"(r_7), "f"(1.0f));
            float q_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_4) : "f"(r_7), "f"(_fma_83), "f"(r_7));
            float _fma_84;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(q_3), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_85;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(q_4), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_4 = make_float2(_fma_84, _fma_85);
            hw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, mul_f32x2_noftz(mul_f32x2_noftz(gc_4, s_4), uc_4))));
            float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
            float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
            float neg_5;
            asm("neg.f32 %0, %1;" : "=f"(neg_5) : "f"(limit));
            float _min_86;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_86) : "f"(x_7.x), "f"(limit));
            float _min_87;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_87) : "f"(x_7.y), "f"(limit));
            float2 gc_5 = make_float2(_min_86, _min_87);
            float _min_89;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_89) : "f"(v_7.x), "f"(limit));
            float _max_88;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_88) : "f"(_min_89), "f"(neg_5));
            float _min_91;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_91) : "f"(v_7.y), "f"(limit));
            float _max_90;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_90) : "f"(_min_91), "f"(neg_5));
            float2 uc_5 = make_float2(_max_88, _max_90);
            float2 _f32x2_mul_b_92 = make_float2(-1.442695f, -1.442695f);
            float2 t_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&gc_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
            float _sub_93;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(t_5.x), "f"(126.0f));
            float u_5 = ((t_5.x > 126.0f) ? _sub_93 : t_5.x);
            float z_5 = approx_exp2(u_5);
            float _fma_94;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(z_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_5 = ((t_5.x > 126.0f) ? z_5 : _fma_94);
            float r_8 = approx_rcp(dn_5);
            float _fma_95;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(-dn_5), "f"(r_8), "f"(1.0f));
            float q_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_5) : "f"(r_8), "f"(_fma_95), "f"(r_8));
            float _sub_96;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(t_5.y), "f"(126.0f));
            float u_6 = ((t_5.y > 126.0f) ? _sub_96 : t_5.y);
            float z_6 = approx_exp2(u_6);
            float _fma_97;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(z_6), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_6 = ((t_5.y > 126.0f) ? z_6 : _fma_97);
            float r_9 = approx_rcp(dn_6);
            float _fma_98;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(-dn_6), "f"(r_9), "f"(1.0f));
            float q_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_6) : "f"(r_9), "f"(_fma_98), "f"(r_9));
            float _fma_99;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(q_5), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_100;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(q_6), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_5 = make_float2(_fma_99, _fma_100);
            hw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, mul_f32x2_noftz(mul_f32x2_noftz(gc_5, s_5), uc_5))));
            if (row < rows) {
                long long col_2 = (long long)((run * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(h + (hoff + col_2))[0] = reinterpret_cast<int4*>(hw)[0];
            }
        }
    }
}

} // extern "C"
