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
kernel_cake_clamped_swiglu_det_train_ebff32b92dd1308d1c58(__nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ dh, __nv_bfloat16* __restrict__ d_gate, __nv_bfloat16* __restrict__ d_up, int rows, long long gate_stride, long long up_stride, long long dh_stride, long long d_gate_stride, long long d_up_stride, float limit)
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
        long long doff = (long long)src * dh_stride;
        long long goff_o = (long long)row * d_gate_stride;
        long long uoff_o = (long long)row * d_up_stride;
        unsigned int gw[4];
        unsigned int uw[4];
        unsigned int dw[4];
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
        {
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(dh + (doff + col_0) + 0);
            dw[0 + 0] = _uv4_3.x;
            dw[0 + 1] = _uv4_3.y;
            dw[0 + 2] = _uv4_3.z;
            dw[0 + 3] = _uv4_3.w;
        }
        unsigned int dgw[4];
        unsigned int duw[4];
        __nv_bfloat162 lowest_0 = __as_bf16x2(gw[0]);
        lowest_0 = __hmin2_nan(lowest_0, __as_bf16x2(gw[1]));
        lowest_0 = __hmin2_nan(lowest_0, __as_bf16x2(gw[2]));
        lowest_0 = __hmin2_nan(lowest_0, __as_bf16x2(gw[3]));
        float2 pair = cake_bf16x2_to_f32x2(lowest_0);
        float lowest;
        asm("min.NaN.f32 %0, %1, %2;" : "=f"(lowest) : "f"(pair.x), "f"(pair.y));
        float _min_5;
        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_5) : "f"(lowest), "f"(limit));
        int _warp_vote_4 = __all_sync(0xFFFFFFFF, _min_5 > -87.0f);
        if (_warp_vote_4 != 0) {
            float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
            float2 v_0 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
            float2 d_0 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
            float neg;
            asm("neg.f32 %0, %1;" : "=f"(neg) : "f"(limit));
            float _min_6;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_6) : "f"(x_0.x), "f"(limit));
            float _min_7;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_7) : "f"(x_0.y), "f"(limit));
            float2 gc = make_float2(_min_6, _min_7);
            float _min_9;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_9) : "f"(v_0.x), "f"(limit));
            float _max_8;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(_min_9), "f"(neg));
            float _min_11;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_11) : "f"(v_0.y), "f"(limit));
            float _max_10;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(_min_11), "f"(neg));
            float2 uc = make_float2(_max_8, _max_10);
            float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
            float2 t;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&gc), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
            float e0 = approx_exp2(t.x);
            float e1 = approx_exp2(t.y);
            float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
            float _rcp_13 = approx_rcp(-nd.x);
            float _rcp_14 = approx_rcp(-nd.y);
            float2 r = make_float2(_rcp_13, _rcp_14);
            float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
            float2 dsilu_0 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc, s), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s);
            float2 ucm_0 = make_float2(((x_0.x <= limit) ? uc.x : 0.0f), ((x_0.y <= limit) ? uc.y : 0.0f));
            float2 silm_0 = make_float2(((uc.x == v_0.x) ? mul_f32x2_noftz(gc, s).x : 0.0f), ((uc.y == v_0.y) ? mul_f32x2_noftz(gc, s).y : 0.0f));
            dgw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(d_0, ucm_0), dsilu_0)));
            duw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(d_0, silm_0)));
            float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
            float2 v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
            float2 d_1 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[1]));
            float neg_0;
            asm("neg.f32 %0, %1;" : "=f"(neg_0) : "f"(limit));
            float _min_15;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_15) : "f"(x_1.x), "f"(limit));
            float _min_16;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_16) : "f"(x_1.y), "f"(limit));
            float2 gc_0 = make_float2(_min_15, _min_16);
            float _min_18;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_18) : "f"(v_1.x), "f"(limit));
            float _max_17;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_17) : "f"(_min_18), "f"(neg_0));
            float _min_20;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_20) : "f"(v_1.y), "f"(limit));
            float _max_19;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_19) : "f"(_min_20), "f"(neg_0));
            float2 uc_0 = make_float2(_max_17, _max_19);
            float2 _f32x2_mul_b_21 = make_float2(-1.442695f, -1.442695f);
            float2 t_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&gc_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
            float e0_0 = approx_exp2(t_0.x);
            float e1_0 = approx_exp2(t_0.y);
            float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
            float _rcp_22 = approx_rcp(-nd_0.x);
            float _rcp_23 = approx_rcp(-nd_0.y);
            float2 r_0 = make_float2(_rcp_22, _rcp_23);
            float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
            float2 dsilu_1 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_0, s_0), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0);
            float2 ucm_1 = make_float2(((x_1.x <= limit) ? uc_0.x : 0.0f), ((x_1.y <= limit) ? uc_0.y : 0.0f));
            float2 silm_1 = make_float2(((uc_0.x == v_1.x) ? mul_f32x2_noftz(gc_0, s_0).x : 0.0f), ((uc_0.y == v_1.y) ? mul_f32x2_noftz(gc_0, s_0).y : 0.0f));
            dgw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(d_1, ucm_1), dsilu_1)));
            duw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(d_1, silm_1)));
            float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
            float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
            float2 d_2 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
            float neg_1;
            asm("neg.f32 %0, %1;" : "=f"(neg_1) : "f"(limit));
            float _min_24;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_24) : "f"(x_2.x), "f"(limit));
            float _min_25;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_25) : "f"(x_2.y), "f"(limit));
            float2 gc_1 = make_float2(_min_24, _min_25);
            float _min_27;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_27) : "f"(v_2.x), "f"(limit));
            float _max_26;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_26) : "f"(_min_27), "f"(neg_1));
            float _min_29;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_29) : "f"(v_2.y), "f"(limit));
            float _max_28;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_28) : "f"(_min_29), "f"(neg_1));
            float2 uc_1 = make_float2(_max_26, _max_28);
            float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
            float2 t_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&gc_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
            float e0_1 = approx_exp2(t_1.x);
            float e1_1 = approx_exp2(t_1.y);
            float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
            float _rcp_31 = approx_rcp(-nd_1.x);
            float _rcp_32 = approx_rcp(-nd_1.y);
            float2 r_1 = make_float2(_rcp_31, _rcp_32);
            float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
            float2 dsilu_2 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_1, s_1), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1);
            float2 ucm_2 = make_float2(((x_2.x <= limit) ? uc_1.x : 0.0f), ((x_2.y <= limit) ? uc_1.y : 0.0f));
            float2 silm_2 = make_float2(((uc_1.x == v_2.x) ? mul_f32x2_noftz(gc_1, s_1).x : 0.0f), ((uc_1.y == v_2.y) ? mul_f32x2_noftz(gc_1, s_1).y : 0.0f));
            dgw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(d_2, ucm_2), dsilu_2)));
            duw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(d_2, silm_2)));
            float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
            float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
            float2 d_3 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
            float neg_2;
            asm("neg.f32 %0, %1;" : "=f"(neg_2) : "f"(limit));
            float _min_33;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_33) : "f"(x_3.x), "f"(limit));
            float _min_34;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_34) : "f"(x_3.y), "f"(limit));
            float2 gc_2 = make_float2(_min_33, _min_34);
            float _min_36;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_36) : "f"(v_3.x), "f"(limit));
            float _max_35;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_35) : "f"(_min_36), "f"(neg_2));
            float _min_38;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_38) : "f"(v_3.y), "f"(limit));
            float _max_37;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(_min_38), "f"(neg_2));
            float2 uc_2 = make_float2(_max_35, _max_37);
            float2 _f32x2_mul_b_39 = make_float2(-1.442695f, -1.442695f);
            float2 t_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&gc_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
            float e0_2 = approx_exp2(t_2.x);
            float e1_2 = approx_exp2(t_2.y);
            float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
            float _rcp_40 = approx_rcp(-nd_2.x);
            float _rcp_41 = approx_rcp(-nd_2.y);
            float2 r_2 = make_float2(_rcp_40, _rcp_41);
            float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
            float2 dsilu_3 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_2, s_2), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2);
            float2 ucm_3 = make_float2(((x_3.x <= limit) ? uc_2.x : 0.0f), ((x_3.y <= limit) ? uc_2.y : 0.0f));
            float2 silm_3 = make_float2(((uc_2.x == v_3.x) ? mul_f32x2_noftz(gc_2, s_2).x : 0.0f), ((uc_2.y == v_3.y) ? mul_f32x2_noftz(gc_2, s_2).y : 0.0f));
            dgw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(d_3, ucm_3), dsilu_3)));
            duw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(d_3, silm_3)));
        } else {
            float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
            float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
            float2 d_4 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
            float neg_3;
            asm("neg.f32 %0, %1;" : "=f"(neg_3) : "f"(limit));
            float _min_42;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_42) : "f"(x_4.x), "f"(limit));
            float _min_43;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_43) : "f"(x_4.y), "f"(limit));
            float2 gc_3 = make_float2(_min_42, _min_43);
            float _min_45;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_45) : "f"(v_4.x), "f"(limit));
            float _max_44;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_44) : "f"(_min_45), "f"(neg_3));
            float _min_47;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_47) : "f"(v_4.y), "f"(limit));
            float _max_46;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_46) : "f"(_min_47), "f"(neg_3));
            float2 uc_3 = make_float2(_max_44, _max_46);
            float2 _f32x2_mul_b_48 = make_float2(-1.442695f, -1.442695f);
            float2 t_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&gc_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
            float _sub_49;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(t_3.x), "f"(126.0f));
            float u = ((t_3.x > 126.0f) ? _sub_49 : t_3.x);
            float z = approx_exp2(u);
            float _fma_50;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(z), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn = ((t_3.x > 126.0f) ? z : _fma_50);
            float r_3 = approx_rcp(dn);
            float _fma_51;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(-dn), "f"(r_3), "f"(1.0f));
            float q;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q) : "f"(r_3), "f"(_fma_51), "f"(r_3));
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(t_3.y), "f"(126.0f));
            float u_0 = ((t_3.y > 126.0f) ? _sub_52 : t_3.y);
            float z_0 = approx_exp2(u_0);
            float _fma_53;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(z_0), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_0 = ((t_3.y > 126.0f) ? z_0 : _fma_53);
            float r_3_1 = approx_rcp(dn_0);
            float _fma_54;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-dn_0), "f"(r_3_1), "f"(1.0f));
            float q_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_0) : "f"(r_3_1), "f"(_fma_54), "f"(r_3_1));
            float _fma_55;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(q), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_56;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(q_0), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_3 = make_float2(_fma_55, _fma_56);
            float2 dsilu_4 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_3, s_3), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3);
            float2 ucm_4 = make_float2(((x_4.x <= limit) ? uc_3.x : 0.0f), ((x_4.y <= limit) ? uc_3.y : 0.0f));
            float2 silm_4 = make_float2(((uc_3.x == v_4.x) ? mul_f32x2_noftz(gc_3, s_3).x : 0.0f), ((uc_3.y == v_4.y) ? mul_f32x2_noftz(gc_3, s_3).y : 0.0f));
            dgw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(d_4, ucm_4), dsilu_4)));
            duw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(d_4, silm_4)));
            float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
            float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
            float2 d_5 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[1]));
            float neg_3_1;
            asm("neg.f32 %0, %1;" : "=f"(neg_3_1) : "f"(limit));
            float _min_57;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_57) : "f"(x_5.x), "f"(limit));
            float _min_58;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_58) : "f"(x_5.y), "f"(limit));
            float2 gc_3_1 = make_float2(_min_57, _min_58);
            float _min_60;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_60) : "f"(v_5.x), "f"(limit));
            float _max_59;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_59) : "f"(_min_60), "f"(neg_3_1));
            float _min_62;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_62) : "f"(v_5.y), "f"(limit));
            float _max_61;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_61) : "f"(_min_62), "f"(neg_3_1));
            float2 uc_3_1 = make_float2(_max_59, _max_61);
            float2 _f32x2_mul_b_63 = make_float2(-1.442695f, -1.442695f);
            float2 t_3_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3_1) : "l"(*(const unsigned long long*)&gc_3_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
            float _sub_64;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(t_3_1.x), "f"(126.0f));
            float u_1 = ((t_3_1.x > 126.0f) ? _sub_64 : t_3_1.x);
            float z_1 = approx_exp2(u_1);
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(z_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_1 = ((t_3_1.x > 126.0f) ? z_1 : _fma_65);
            float r_4 = approx_rcp(dn_1);
            float _fma_66;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-dn_1), "f"(r_4), "f"(1.0f));
            float q_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_1) : "f"(r_4), "f"(_fma_66), "f"(r_4));
            float _sub_67;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(t_3_1.y), "f"(126.0f));
            float u_2 = ((t_3_1.y > 126.0f) ? _sub_67 : t_3_1.y);
            float z_2 = approx_exp2(u_2);
            float _fma_68;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(z_2), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_2 = ((t_3_1.y > 126.0f) ? z_2 : _fma_68);
            float r_5 = approx_rcp(dn_2);
            float _fma_69;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(-dn_2), "f"(r_5), "f"(1.0f));
            float q_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_2) : "f"(r_5), "f"(_fma_69), "f"(r_5));
            float _fma_70;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(q_1), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_71;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(q_2), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_3_1 = make_float2(_fma_70, _fma_71);
            float2 dsilu_5 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_3_1, s_3_1), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3_1), s_3_1);
            float2 ucm_5 = make_float2(((x_5.x <= limit) ? uc_3_1.x : 0.0f), ((x_5.y <= limit) ? uc_3_1.y : 0.0f));
            float2 silm_5 = make_float2(((uc_3_1.x == v_5.x) ? mul_f32x2_noftz(gc_3_1, s_3_1).x : 0.0f), ((uc_3_1.y == v_5.y) ? mul_f32x2_noftz(gc_3_1, s_3_1).y : 0.0f));
            dgw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(d_5, ucm_5), dsilu_5)));
            duw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(d_5, silm_5)));
            float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
            float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
            float2 d_6 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
            float neg_4;
            asm("neg.f32 %0, %1;" : "=f"(neg_4) : "f"(limit));
            float _min_72;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_72) : "f"(x_6.x), "f"(limit));
            float _min_73;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_73) : "f"(x_6.y), "f"(limit));
            float2 gc_4 = make_float2(_min_72, _min_73);
            float _min_75;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_75) : "f"(v_6.x), "f"(limit));
            float _max_74;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_74) : "f"(_min_75), "f"(neg_4));
            float _min_77;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_77) : "f"(v_6.y), "f"(limit));
            float _max_76;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_76) : "f"(_min_77), "f"(neg_4));
            float2 uc_4 = make_float2(_max_74, _max_76);
            float2 _f32x2_mul_b_78 = make_float2(-1.442695f, -1.442695f);
            float2 t_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&gc_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
            float _sub_79;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(t_4.x), "f"(126.0f));
            float u_3 = ((t_4.x > 126.0f) ? _sub_79 : t_4.x);
            float z_3 = approx_exp2(u_3);
            float _fma_80;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(z_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_3 = ((t_4.x > 126.0f) ? z_3 : _fma_80);
            float r_6 = approx_rcp(dn_3);
            float _fma_81;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(-dn_3), "f"(r_6), "f"(1.0f));
            float q_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_3) : "f"(r_6), "f"(_fma_81), "f"(r_6));
            float _sub_82;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(t_4.y), "f"(126.0f));
            float u_4 = ((t_4.y > 126.0f) ? _sub_82 : t_4.y);
            float z_4 = approx_exp2(u_4);
            float _fma_83;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(z_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_4 = ((t_4.y > 126.0f) ? z_4 : _fma_83);
            float r_7 = approx_rcp(dn_4);
            float _fma_84;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(-dn_4), "f"(r_7), "f"(1.0f));
            float q_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_4) : "f"(r_7), "f"(_fma_84), "f"(r_7));
            float _fma_85;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(q_3), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_86;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(q_4), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_4 = make_float2(_fma_85, _fma_86);
            float2 dsilu_6 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_4, s_4), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4);
            float2 ucm_6 = make_float2(((x_6.x <= limit) ? uc_4.x : 0.0f), ((x_6.y <= limit) ? uc_4.y : 0.0f));
            float2 silm_6 = make_float2(((uc_4.x == v_6.x) ? mul_f32x2_noftz(gc_4, s_4).x : 0.0f), ((uc_4.y == v_6.y) ? mul_f32x2_noftz(gc_4, s_4).y : 0.0f));
            dgw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(d_6, ucm_6), dsilu_6)));
            duw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(d_6, silm_6)));
            float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
            float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
            float2 d_7 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
            float neg_5;
            asm("neg.f32 %0, %1;" : "=f"(neg_5) : "f"(limit));
            float _min_87;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_87) : "f"(x_7.x), "f"(limit));
            float _min_88;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_88) : "f"(x_7.y), "f"(limit));
            float2 gc_5 = make_float2(_min_87, _min_88);
            float _min_90;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_90) : "f"(v_7.x), "f"(limit));
            float _max_89;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_89) : "f"(_min_90), "f"(neg_5));
            float _min_92;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_92) : "f"(v_7.y), "f"(limit));
            float _max_91;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_91) : "f"(_min_92), "f"(neg_5));
            float2 uc_5 = make_float2(_max_89, _max_91);
            float2 _f32x2_mul_b_93 = make_float2(-1.442695f, -1.442695f);
            float2 t_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&gc_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
            float _sub_94;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(t_5.x), "f"(126.0f));
            float u_5 = ((t_5.x > 126.0f) ? _sub_94 : t_5.x);
            float z_5 = approx_exp2(u_5);
            float _fma_95;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(z_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_5 = ((t_5.x > 126.0f) ? z_5 : _fma_95);
            float r_8 = approx_rcp(dn_5);
            float _fma_96;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(-dn_5), "f"(r_8), "f"(1.0f));
            float q_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_5) : "f"(r_8), "f"(_fma_96), "f"(r_8));
            float _sub_97;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(t_5.y), "f"(126.0f));
            float u_6 = ((t_5.y > 126.0f) ? _sub_97 : t_5.y);
            float z_6 = approx_exp2(u_6);
            float _fma_98;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(z_6), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_6 = ((t_5.y > 126.0f) ? z_6 : _fma_98);
            float r_9 = approx_rcp(dn_6);
            float _fma_99;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-dn_6), "f"(r_9), "f"(1.0f));
            float q_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_6) : "f"(r_9), "f"(_fma_99), "f"(r_9));
            float _fma_100;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(q_5), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_101;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(q_6), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_5 = make_float2(_fma_100, _fma_101);
            float2 dsilu_7 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_5, s_5), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5);
            float2 ucm_7 = make_float2(((x_7.x <= limit) ? uc_5.x : 0.0f), ((x_7.y <= limit) ? uc_5.y : 0.0f));
            float2 silm_7 = make_float2(((uc_5.x == v_7.x) ? mul_f32x2_noftz(gc_5, s_5).x : 0.0f), ((uc_5.y == v_7.y) ? mul_f32x2_noftz(gc_5, s_5).y : 0.0f));
            dgw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(d_7, ucm_7), dsilu_7)));
            duw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(d_7, silm_7)));
        }
        if (row < rows) {
            long long col_1 = (long long)((run * 32 + lane_1) * 8);
            reinterpret_cast<int4*>(d_gate + (goff_o + col_1))[0] = reinterpret_cast<int4*>(dgw)[0];
            reinterpret_cast<int4*>(d_up + (uoff_o + col_1))[0] = reinterpret_cast<int4*>(duw)[0];
        }
    }
}

} // extern "C"
