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

__global__ __launch_bounds__(288) void
kernel_cake_clamped_swiglu_det_train_c30f690752c972d064fa(__nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ dh, float* __restrict__ w, __nv_bfloat16* __restrict__ d_gate, __nv_bfloat16* __restrict__ d_up, float* __restrict__ d_weight, int rows, long long gate_stride, long long up_stride, long long dh_stride, long long d_gate_stride, long long d_up_stride, float limit, int weighted)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[16384];

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    unsigned int* _rec = reinterpret_cast<unsigned int*>(smem_static_raw + 0);
    const int rec_addr = (int)(unsigned long long)__cvta_generic_to_shared(_rec);

    // === Task calls (declaration order) ===
    int lane_1 = tid & 31;
    int run = tid >> 5;
    #pragma unroll 1
    for (int row = bid; row < rows; row += num_bids) {
        long long goff = (long long)row * gate_stride;
        long long uoff = (long long)row * up_stride;
        long long doff = (long long)row * dh_stride;
        long long goff_o = (long long)row * d_gate_stride;
        long long uoff_o = (long long)row * d_up_stride;
        float wv = w[row * weighted];
        float wr = ((weighted != 0) ? wv : 1.0f);
        float2 wr2 = make_float2(wr, wr);
        unsigned int gw[4];
        unsigned int uw[4];
        unsigned int dw[4];
        long long col_0 = (long long)((run * 32 + lane_1) * 8);
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(gate + (goff + col_0) + 0);
            gw[0 + 0] = _uv4_0.x;
            gw[0 + 1] = _uv4_0.y;
            gw[0 + 2] = _uv4_0.z;
            gw[0 + 3] = _uv4_0.w;
        }
        {
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(up + (uoff + col_0) + 0);
            uw[0 + 0] = _uv4_1.x;
            uw[0 + 1] = _uv4_1.y;
            uw[0 + 2] = _uv4_1.z;
            uw[0 + 3] = _uv4_1.w;
        }
        {
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(dh + (doff + col_0) + 0);
            dw[0 + 0] = _uv4_2.x;
            dw[0 + 1] = _uv4_2.y;
            dw[0 + 2] = _uv4_2.z;
            dw[0 + 3] = _uv4_2.w;
        }
        unsigned int dgw[4];
        unsigned int duw[4];
        float av[8];
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
            float2 d_0 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
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
            float2 dhw_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dhw_0) : "l"(*(const unsigned long long*)&d_0), "l"(*(const unsigned long long*)&wr2));
            float2 dsilu_0 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc, s), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s);
            float2 ucm_0 = make_float2(((x_0.x <= limit) ? uc.x : 0.0f), ((x_0.y <= limit) ? uc.y : 0.0f));
            float2 silm_0 = make_float2(((uc.x == v_0.x) ? mul_f32x2_noftz(gc, s).x : 0.0f), ((uc.y == v_0.y) ? mul_f32x2_noftz(gc, s).y : 0.0f));
            dgw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(dhw_0, ucm_0), dsilu_0)));
            duw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(dhw_0, silm_0)));
            float2 _f32x2_mul_a_14 = mul_f32x2_noftz(gc, s);
            float2 a_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&uc));
            av[0] = a_0.x;
            av[1] = a_0.y;
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
            float2 dhw_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dhw_1) : "l"(*(const unsigned long long*)&d_1), "l"(*(const unsigned long long*)&wr2));
            float2 dsilu_1 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_0, s_0), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0);
            float2 ucm_1 = make_float2(((x_1.x <= limit) ? uc_0.x : 0.0f), ((x_1.y <= limit) ? uc_0.y : 0.0f));
            float2 silm_1 = make_float2(((uc_0.x == v_1.x) ? mul_f32x2_noftz(gc_0, s_0).x : 0.0f), ((uc_0.y == v_1.y) ? mul_f32x2_noftz(gc_0, s_0).y : 0.0f));
            dgw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(dhw_1, ucm_1), dsilu_1)));
            duw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(dhw_1, silm_1)));
            float2 _f32x2_mul_a_24 = mul_f32x2_noftz(gc_0, s_0);
            float2 a_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&uc_0));
            av[2] = a_1.x;
            av[3] = a_1.y;
            float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
            float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
            float2 d_2 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
            float neg_1;
            asm("neg.f32 %0, %1;" : "=f"(neg_1) : "f"(limit));
            float _min_25;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_25) : "f"(x_2.x), "f"(limit));
            float _min_26;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_26) : "f"(x_2.y), "f"(limit));
            float2 gc_1 = make_float2(_min_25, _min_26);
            float _min_28;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_28) : "f"(v_2.x), "f"(limit));
            float _max_27;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_27) : "f"(_min_28), "f"(neg_1));
            float _min_30;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_30) : "f"(v_2.y), "f"(limit));
            float _max_29;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_29) : "f"(_min_30), "f"(neg_1));
            float2 uc_1 = make_float2(_max_27, _max_29);
            float2 _f32x2_mul_b_31 = make_float2(-1.442695f, -1.442695f);
            float2 t_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&gc_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
            float e0_1 = approx_exp2(t_1.x);
            float e1_1 = approx_exp2(t_1.y);
            float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
            float _rcp_32 = approx_rcp(-nd_1.x);
            float _rcp_33 = approx_rcp(-nd_1.y);
            float2 r_1 = make_float2(_rcp_32, _rcp_33);
            float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
            float2 dhw_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dhw_2) : "l"(*(const unsigned long long*)&d_2), "l"(*(const unsigned long long*)&wr2));
            float2 dsilu_2 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_1, s_1), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1);
            float2 ucm_2 = make_float2(((x_2.x <= limit) ? uc_1.x : 0.0f), ((x_2.y <= limit) ? uc_1.y : 0.0f));
            float2 silm_2 = make_float2(((uc_1.x == v_2.x) ? mul_f32x2_noftz(gc_1, s_1).x : 0.0f), ((uc_1.y == v_2.y) ? mul_f32x2_noftz(gc_1, s_1).y : 0.0f));
            dgw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(dhw_2, ucm_2), dsilu_2)));
            duw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(dhw_2, silm_2)));
            float2 _f32x2_mul_a_34 = mul_f32x2_noftz(gc_1, s_1);
            float2 a_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&uc_1));
            av[4] = a_2.x;
            av[5] = a_2.y;
            float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
            float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
            float2 d_3 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
            float neg_2;
            asm("neg.f32 %0, %1;" : "=f"(neg_2) : "f"(limit));
            float _min_35;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_35) : "f"(x_3.x), "f"(limit));
            float _min_36;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_36) : "f"(x_3.y), "f"(limit));
            float2 gc_2 = make_float2(_min_35, _min_36);
            float _min_38;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_38) : "f"(v_3.x), "f"(limit));
            float _max_37;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(_min_38), "f"(neg_2));
            float _min_40;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_40) : "f"(v_3.y), "f"(limit));
            float _max_39;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_39) : "f"(_min_40), "f"(neg_2));
            float2 uc_2 = make_float2(_max_37, _max_39);
            float2 _f32x2_mul_b_41 = make_float2(-1.442695f, -1.442695f);
            float2 t_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&gc_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
            float e0_2 = approx_exp2(t_2.x);
            float e1_2 = approx_exp2(t_2.y);
            float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
            float _rcp_42 = approx_rcp(-nd_2.x);
            float _rcp_43 = approx_rcp(-nd_2.y);
            float2 r_2 = make_float2(_rcp_42, _rcp_43);
            float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
            float2 dhw_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dhw_3) : "l"(*(const unsigned long long*)&d_3), "l"(*(const unsigned long long*)&wr2));
            float2 dsilu_3 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_2, s_2), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2);
            float2 ucm_3 = make_float2(((x_3.x <= limit) ? uc_2.x : 0.0f), ((x_3.y <= limit) ? uc_2.y : 0.0f));
            float2 silm_3 = make_float2(((uc_2.x == v_3.x) ? mul_f32x2_noftz(gc_2, s_2).x : 0.0f), ((uc_2.y == v_3.y) ? mul_f32x2_noftz(gc_2, s_2).y : 0.0f));
            dgw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(dhw_3, ucm_3), dsilu_3)));
            duw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(dhw_3, silm_3)));
            float2 _f32x2_mul_a_44 = mul_f32x2_noftz(gc_2, s_2);
            float2 a_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&uc_2));
            av[6] = a_3.x;
            av[7] = a_3.y;
        } else {
            float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
            float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
            float2 d_4 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
            float neg_3;
            asm("neg.f32 %0, %1;" : "=f"(neg_3) : "f"(limit));
            float _min_45;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_45) : "f"(x_4.x), "f"(limit));
            float _min_46;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_46) : "f"(x_4.y), "f"(limit));
            float2 gc_3 = make_float2(_min_45, _min_46);
            float _min_48;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_48) : "f"(v_4.x), "f"(limit));
            float _max_47;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_47) : "f"(_min_48), "f"(neg_3));
            float _min_50;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_50) : "f"(v_4.y), "f"(limit));
            float _max_49;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_49) : "f"(_min_50), "f"(neg_3));
            float2 uc_3 = make_float2(_max_47, _max_49);
            float2 _f32x2_mul_b_51 = make_float2(-1.442695f, -1.442695f);
            float2 t_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&gc_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(t_3.x), "f"(126.0f));
            float u = ((t_3.x > 126.0f) ? _sub_52 : t_3.x);
            float z = approx_exp2(u);
            float _fma_53;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(z), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn = ((t_3.x > 126.0f) ? z : _fma_53);
            float r_3 = approx_rcp(dn);
            float _fma_54;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-dn), "f"(r_3), "f"(1.0f));
            float q;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q) : "f"(r_3), "f"(_fma_54), "f"(r_3));
            float _sub_55;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(t_3.y), "f"(126.0f));
            float u_0 = ((t_3.y > 126.0f) ? _sub_55 : t_3.y);
            float z_0 = approx_exp2(u_0);
            float _fma_56;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(z_0), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_0 = ((t_3.y > 126.0f) ? z_0 : _fma_56);
            float r_3_1 = approx_rcp(dn_0);
            float _fma_57;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-dn_0), "f"(r_3_1), "f"(1.0f));
            float q_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_0) : "f"(r_3_1), "f"(_fma_57), "f"(r_3_1));
            float _fma_58;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(q), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_59;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(q_0), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_3 = make_float2(_fma_58, _fma_59);
            float2 dhw_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dhw_4) : "l"(*(const unsigned long long*)&d_4), "l"(*(const unsigned long long*)&wr2));
            float2 dsilu_4 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_3, s_3), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3);
            float2 ucm_4 = make_float2(((x_4.x <= limit) ? uc_3.x : 0.0f), ((x_4.y <= limit) ? uc_3.y : 0.0f));
            float2 silm_4 = make_float2(((uc_3.x == v_4.x) ? mul_f32x2_noftz(gc_3, s_3).x : 0.0f), ((uc_3.y == v_4.y) ? mul_f32x2_noftz(gc_3, s_3).y : 0.0f));
            dgw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(dhw_4, ucm_4), dsilu_4)));
            duw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(dhw_4, silm_4)));
            float2 _f32x2_mul_a_60 = mul_f32x2_noftz(gc_3, s_3);
            float2 a_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&uc_3));
            av[0] = a_4.x;
            av[1] = a_4.y;
            float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
            float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
            float2 d_5 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[1]));
            float neg_3_1;
            asm("neg.f32 %0, %1;" : "=f"(neg_3_1) : "f"(limit));
            float _min_61;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_61) : "f"(x_5.x), "f"(limit));
            float _min_62;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_62) : "f"(x_5.y), "f"(limit));
            float2 gc_3_1 = make_float2(_min_61, _min_62);
            float _min_64;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_64) : "f"(v_5.x), "f"(limit));
            float _max_63;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_63) : "f"(_min_64), "f"(neg_3_1));
            float _min_66;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_66) : "f"(v_5.y), "f"(limit));
            float _max_65;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_65) : "f"(_min_66), "f"(neg_3_1));
            float2 uc_3_1 = make_float2(_max_63, _max_65);
            float2 _f32x2_mul_b_67 = make_float2(-1.442695f, -1.442695f);
            float2 t_3_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3_1) : "l"(*(const unsigned long long*)&gc_3_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
            float _sub_68;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(t_3_1.x), "f"(126.0f));
            float u_1 = ((t_3_1.x > 126.0f) ? _sub_68 : t_3_1.x);
            float z_1 = approx_exp2(u_1);
            float _fma_69;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(z_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_1 = ((t_3_1.x > 126.0f) ? z_1 : _fma_69);
            float r_4 = approx_rcp(dn_1);
            float _fma_70;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(-dn_1), "f"(r_4), "f"(1.0f));
            float q_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_1) : "f"(r_4), "f"(_fma_70), "f"(r_4));
            float _sub_71;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(t_3_1.y), "f"(126.0f));
            float u_2 = ((t_3_1.y > 126.0f) ? _sub_71 : t_3_1.y);
            float z_2 = approx_exp2(u_2);
            float _fma_72;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(z_2), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_2 = ((t_3_1.y > 126.0f) ? z_2 : _fma_72);
            float r_5 = approx_rcp(dn_2);
            float _fma_73;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(-dn_2), "f"(r_5), "f"(1.0f));
            float q_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_2) : "f"(r_5), "f"(_fma_73), "f"(r_5));
            float _fma_74;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(q_1), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_75;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(q_2), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_3_1 = make_float2(_fma_74, _fma_75);
            float2 dhw_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dhw_5) : "l"(*(const unsigned long long*)&d_5), "l"(*(const unsigned long long*)&wr2));
            float2 dsilu_5 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_3_1, s_3_1), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3_1), s_3_1);
            float2 ucm_5 = make_float2(((x_5.x <= limit) ? uc_3_1.x : 0.0f), ((x_5.y <= limit) ? uc_3_1.y : 0.0f));
            float2 silm_5 = make_float2(((uc_3_1.x == v_5.x) ? mul_f32x2_noftz(gc_3_1, s_3_1).x : 0.0f), ((uc_3_1.y == v_5.y) ? mul_f32x2_noftz(gc_3_1, s_3_1).y : 0.0f));
            dgw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(dhw_5, ucm_5), dsilu_5)));
            duw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(dhw_5, silm_5)));
            float2 _f32x2_mul_a_76 = mul_f32x2_noftz(gc_3_1, s_3_1);
            float2 a_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&uc_3_1));
            av[2] = a_5.x;
            av[3] = a_5.y;
            float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
            float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
            float2 d_6 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
            float neg_4;
            asm("neg.f32 %0, %1;" : "=f"(neg_4) : "f"(limit));
            float _min_77;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_77) : "f"(x_6.x), "f"(limit));
            float _min_78;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_78) : "f"(x_6.y), "f"(limit));
            float2 gc_4 = make_float2(_min_77, _min_78);
            float _min_80;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_80) : "f"(v_6.x), "f"(limit));
            float _max_79;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_79) : "f"(_min_80), "f"(neg_4));
            float _min_82;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_82) : "f"(v_6.y), "f"(limit));
            float _max_81;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_81) : "f"(_min_82), "f"(neg_4));
            float2 uc_4 = make_float2(_max_79, _max_81);
            float2 _f32x2_mul_b_83 = make_float2(-1.442695f, -1.442695f);
            float2 t_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&gc_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
            float _sub_84;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(t_4.x), "f"(126.0f));
            float u_3 = ((t_4.x > 126.0f) ? _sub_84 : t_4.x);
            float z_3 = approx_exp2(u_3);
            float _fma_85;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(z_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_3 = ((t_4.x > 126.0f) ? z_3 : _fma_85);
            float r_6 = approx_rcp(dn_3);
            float _fma_86;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(-dn_3), "f"(r_6), "f"(1.0f));
            float q_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_3) : "f"(r_6), "f"(_fma_86), "f"(r_6));
            float _sub_87;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(t_4.y), "f"(126.0f));
            float u_4 = ((t_4.y > 126.0f) ? _sub_87 : t_4.y);
            float z_4 = approx_exp2(u_4);
            float _fma_88;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(z_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_4 = ((t_4.y > 126.0f) ? z_4 : _fma_88);
            float r_7 = approx_rcp(dn_4);
            float _fma_89;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(-dn_4), "f"(r_7), "f"(1.0f));
            float q_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_4) : "f"(r_7), "f"(_fma_89), "f"(r_7));
            float _fma_90;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(q_3), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_91;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(q_4), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_4 = make_float2(_fma_90, _fma_91);
            float2 dhw_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dhw_6) : "l"(*(const unsigned long long*)&d_6), "l"(*(const unsigned long long*)&wr2));
            float2 dsilu_6 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_4, s_4), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4);
            float2 ucm_6 = make_float2(((x_6.x <= limit) ? uc_4.x : 0.0f), ((x_6.y <= limit) ? uc_4.y : 0.0f));
            float2 silm_6 = make_float2(((uc_4.x == v_6.x) ? mul_f32x2_noftz(gc_4, s_4).x : 0.0f), ((uc_4.y == v_6.y) ? mul_f32x2_noftz(gc_4, s_4).y : 0.0f));
            dgw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(dhw_6, ucm_6), dsilu_6)));
            duw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(dhw_6, silm_6)));
            float2 _f32x2_mul_a_92 = mul_f32x2_noftz(gc_4, s_4);
            float2 a_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&uc_4));
            av[4] = a_6.x;
            av[5] = a_6.y;
            float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
            float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
            float2 d_7 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
            float neg_5;
            asm("neg.f32 %0, %1;" : "=f"(neg_5) : "f"(limit));
            float _min_93;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_93) : "f"(x_7.x), "f"(limit));
            float _min_94;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_94) : "f"(x_7.y), "f"(limit));
            float2 gc_5 = make_float2(_min_93, _min_94);
            float _min_96;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_96) : "f"(v_7.x), "f"(limit));
            float _max_95;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_95) : "f"(_min_96), "f"(neg_5));
            float _min_98;
            asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_98) : "f"(v_7.y), "f"(limit));
            float _max_97;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_97) : "f"(_min_98), "f"(neg_5));
            float2 uc_5 = make_float2(_max_95, _max_97);
            float2 _f32x2_mul_b_99 = make_float2(-1.442695f, -1.442695f);
            float2 t_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&gc_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_99));
            float _sub_100;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(t_5.x), "f"(126.0f));
            float u_5 = ((t_5.x > 126.0f) ? _sub_100 : t_5.x);
            float z_5 = approx_exp2(u_5);
            float _fma_101;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(z_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_5 = ((t_5.x > 126.0f) ? z_5 : _fma_101);
            float r_8 = approx_rcp(dn_5);
            float _fma_102;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(-dn_5), "f"(r_8), "f"(1.0f));
            float q_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_5) : "f"(r_8), "f"(_fma_102), "f"(r_8));
            float _sub_103;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_103) : "f"(t_5.y), "f"(126.0f));
            float u_6 = ((t_5.y > 126.0f) ? _sub_103 : t_5.y);
            float z_6 = approx_exp2(u_6);
            float _fma_104;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(z_6), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
            float dn_6 = ((t_5.y > 126.0f) ? z_6 : _fma_104);
            float r_9 = approx_rcp(dn_6);
            float _fma_105;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(-dn_6), "f"(r_9), "f"(1.0f));
            float q_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_6) : "f"(r_9), "f"(_fma_105), "f"(r_9));
            float _fma_106;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(q_5), "f"(1.1754944e-38f), "f"(0.0f));
            float _fma_107;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(q_6), "f"(1.1754944e-38f), "f"(0.0f));
            float2 s_5 = make_float2(_fma_106, _fma_107);
            float2 dhw_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dhw_7) : "l"(*(const unsigned long long*)&d_7), "l"(*(const unsigned long long*)&wr2));
            float2 dsilu_7 = fma_f32x2_rn_noftz(mul_f32x2_noftz(gc_5, s_5), sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5);
            float2 ucm_7 = make_float2(((x_7.x <= limit) ? uc_5.x : 0.0f), ((x_7.y <= limit) ? uc_5.y : 0.0f));
            float2 silm_7 = make_float2(((uc_5.x == v_7.x) ? mul_f32x2_noftz(gc_5, s_5).x : 0.0f), ((uc_5.y == v_7.y) ? mul_f32x2_noftz(gc_5, s_5).y : 0.0f));
            dgw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(mul_f32x2_noftz(dhw_7, ucm_7), dsilu_7)));
            duw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(dhw_7, silm_7)));
            float2 _f32x2_mul_a_108 = mul_f32x2_noftz(gc_5, s_5);
            float2 a_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&uc_5));
            av[6] = a_7.x;
            av[7] = a_7.y;
        }
        long long col_1 = (long long)((run * 32 + lane_1) * 8);
        reinterpret_cast<int4*>(d_gate + (goff_o + col_1))[0] = reinterpret_cast<int4*>(dgw)[0];
        reinterpret_cast<int4*>(d_up + (uoff_o + col_1))[0] = reinterpret_cast<int4*>(duw)[0];
        if (run > 0) {
            int slot_0 = (run - 1) * 4 * 32 + lane_1;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(rec_addr + (unsigned int)(16 * slot_0)), "r"(dw[0]), "r"(__as_u32(av[0])), "r"(__as_u32(av[1])), "r"(0) : "memory");
            int slot_1 = ((run - 1) * 4 + 1) * 32 + lane_1;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(rec_addr + (unsigned int)(16 * slot_1)), "r"(dw[1]), "r"(__as_u32(av[2])), "r"(__as_u32(av[3])), "r"(0) : "memory");
            int slot_2 = ((run - 1) * 4 + 2) * 32 + lane_1;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(rec_addr + (unsigned int)(16 * slot_2)), "r"(dw[2]), "r"(__as_u32(av[4])), "r"(__as_u32(av[5])), "r"(0) : "memory");
            int slot_3 = ((run - 1) * 4 + 3) * 32 + lane_1;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(rec_addr + (unsigned int)(16 * slot_3)), "r"(dw[3]), "r"(__as_u32(av[6])), "r"(__as_u32(av[7])), "r"(0) : "memory");
        }
        __syncthreads();
        if (run == 0) {
            float2 chains = make_float2(0.0f, 0.0f);
            float2 d_8 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
            chains = fma_f32x2_rn_noftz(d_8, make_float2(av[0], av[1]), chains);
            float2 d_9 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[1]));
            chains = fma_f32x2_rn_noftz(d_9, make_float2(av[2], av[3]), chains);
            float2 d_10 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
            chains = fma_f32x2_rn_noftz(d_10, make_float2(av[4], av[5]), chains);
            float2 d_11 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
            chains = fma_f32x2_rn_noftz(d_11, make_float2(av[6], av[7]), chains);
            int slot_4 = lane_1;
            unsigned int r4_0[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_0[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_4)));
            float2 d_12 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_0[0]));
            float2 a_8 = make_float2(__uint_as_float(r4_0[1]), __uint_as_float(r4_0[2]));
            chains = fma_f32x2_rn_noftz(d_12, a_8, chains);
            int slot_5 = 32 + lane_1;
            unsigned int r4_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_1[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_5)));
            float2 d_13 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_1[0]));
            float2 a_9 = make_float2(__uint_as_float(r4_1[1]), __uint_as_float(r4_1[2]));
            chains = fma_f32x2_rn_noftz(d_13, a_9, chains);
            int slot_6 = 64 + lane_1;
            unsigned int r4_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_2[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_6)));
            float2 d_14 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_2[0]));
            float2 a_10 = make_float2(__uint_as_float(r4_2[1]), __uint_as_float(r4_2[2]));
            chains = fma_f32x2_rn_noftz(d_14, a_10, chains);
            int slot_7 = 96 + lane_1;
            unsigned int r4_3[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_3[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_7)));
            float2 d_15 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_3[0]));
            float2 a_11 = make_float2(__uint_as_float(r4_3[1]), __uint_as_float(r4_3[2]));
            chains = fma_f32x2_rn_noftz(d_15, a_11, chains);
            int slot_8 = 128 + lane_1;
            unsigned int r4_4[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_4[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_8)));
            float2 d_16 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_4[0]));
            float2 a_12 = make_float2(__uint_as_float(r4_4[1]), __uint_as_float(r4_4[2]));
            chains = fma_f32x2_rn_noftz(d_16, a_12, chains);
            int slot_9 = 160 + lane_1;
            unsigned int r4_5[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_5[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_9)));
            float2 d_17 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_5[0]));
            float2 a_13 = make_float2(__uint_as_float(r4_5[1]), __uint_as_float(r4_5[2]));
            chains = fma_f32x2_rn_noftz(d_17, a_13, chains);
            int slot_10 = 192 + lane_1;
            unsigned int r4_6[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_6[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_10)));
            float2 d_18 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_6[0]));
            float2 a_14 = make_float2(__uint_as_float(r4_6[1]), __uint_as_float(r4_6[2]));
            chains = fma_f32x2_rn_noftz(d_18, a_14, chains);
            int slot_11 = 224 + lane_1;
            unsigned int r4_7[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_7[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_11)));
            float2 d_19 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_7[0]));
            float2 a_15 = make_float2(__uint_as_float(r4_7[1]), __uint_as_float(r4_7[2]));
            chains = fma_f32x2_rn_noftz(d_19, a_15, chains);
            int slot_12 = 256 + lane_1;
            unsigned int r4_8[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_8[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_12)));
            float2 d_20 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_8[0]));
            float2 a_16 = make_float2(__uint_as_float(r4_8[1]), __uint_as_float(r4_8[2]));
            chains = fma_f32x2_rn_noftz(d_20, a_16, chains);
            int slot_13 = 288 + lane_1;
            unsigned int r4_9[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_9[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_13)));
            float2 d_21 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_9[0]));
            float2 a_17 = make_float2(__uint_as_float(r4_9[1]), __uint_as_float(r4_9[2]));
            chains = fma_f32x2_rn_noftz(d_21, a_17, chains);
            int slot_14 = 320 + lane_1;
            unsigned int r4_10[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_10[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_14)));
            float2 d_22 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_10[0]));
            float2 a_18 = make_float2(__uint_as_float(r4_10[1]), __uint_as_float(r4_10[2]));
            chains = fma_f32x2_rn_noftz(d_22, a_18, chains);
            int slot_15 = 352 + lane_1;
            unsigned int r4_11[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_11[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_15)));
            float2 d_23 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_11[0]));
            float2 a_19 = make_float2(__uint_as_float(r4_11[1]), __uint_as_float(r4_11[2]));
            chains = fma_f32x2_rn_noftz(d_23, a_19, chains);
            int slot_16 = 384 + lane_1;
            unsigned int r4_12[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_12[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_16)));
            float2 d_24 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_12[0]));
            float2 a_20 = make_float2(__uint_as_float(r4_12[1]), __uint_as_float(r4_12[2]));
            chains = fma_f32x2_rn_noftz(d_24, a_20, chains);
            int slot_17 = 416 + lane_1;
            unsigned int r4_13[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_13[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_17)));
            float2 d_25 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_13[0]));
            float2 a_21 = make_float2(__uint_as_float(r4_13[1]), __uint_as_float(r4_13[2]));
            chains = fma_f32x2_rn_noftz(d_25, a_21, chains);
            int slot_18 = 448 + lane_1;
            unsigned int r4_14[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_14[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_18)));
            float2 d_26 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_14[0]));
            float2 a_22 = make_float2(__uint_as_float(r4_14[1]), __uint_as_float(r4_14[2]));
            chains = fma_f32x2_rn_noftz(d_26, a_22, chains);
            int slot_19 = 480 + lane_1;
            unsigned int r4_15[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_15[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_19)));
            float2 d_27 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_15[0]));
            float2 a_23 = make_float2(__uint_as_float(r4_15[1]), __uint_as_float(r4_15[2]));
            chains = fma_f32x2_rn_noftz(d_27, a_23, chains);
            int slot_20 = 512 + lane_1;
            unsigned int r4_16[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_16[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_20)));
            float2 d_28 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_16[0]));
            float2 a_24 = make_float2(__uint_as_float(r4_16[1]), __uint_as_float(r4_16[2]));
            chains = fma_f32x2_rn_noftz(d_28, a_24, chains);
            int slot_21 = 544 + lane_1;
            unsigned int r4_17[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_17[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_21)));
            float2 d_29 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_17[0]));
            float2 a_25 = make_float2(__uint_as_float(r4_17[1]), __uint_as_float(r4_17[2]));
            chains = fma_f32x2_rn_noftz(d_29, a_25, chains);
            int slot_22 = 576 + lane_1;
            unsigned int r4_18[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_18[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_22)));
            float2 d_30 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_18[0]));
            float2 a_26 = make_float2(__uint_as_float(r4_18[1]), __uint_as_float(r4_18[2]));
            chains = fma_f32x2_rn_noftz(d_30, a_26, chains);
            int slot_23 = 608 + lane_1;
            unsigned int r4_19[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_19[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_23)));
            float2 d_31 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_19[0]));
            float2 a_27 = make_float2(__uint_as_float(r4_19[1]), __uint_as_float(r4_19[2]));
            chains = fma_f32x2_rn_noftz(d_31, a_27, chains);
            int slot_24 = 640 + lane_1;
            unsigned int r4_20[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_20[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_24)));
            float2 d_32 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_20[0]));
            float2 a_28 = make_float2(__uint_as_float(r4_20[1]), __uint_as_float(r4_20[2]));
            chains = fma_f32x2_rn_noftz(d_32, a_28, chains);
            int slot_25 = 672 + lane_1;
            unsigned int r4_21[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_21[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_25)));
            float2 d_33 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_21[0]));
            float2 a_29 = make_float2(__uint_as_float(r4_21[1]), __uint_as_float(r4_21[2]));
            chains = fma_f32x2_rn_noftz(d_33, a_29, chains);
            int slot_26 = 704 + lane_1;
            unsigned int r4_22[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_22[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_26)));
            float2 d_34 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_22[0]));
            float2 a_30 = make_float2(__uint_as_float(r4_22[1]), __uint_as_float(r4_22[2]));
            chains = fma_f32x2_rn_noftz(d_34, a_30, chains);
            int slot_27 = 736 + lane_1;
            unsigned int r4_23[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_23[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_27)));
            float2 d_35 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_23[0]));
            float2 a_31 = make_float2(__uint_as_float(r4_23[1]), __uint_as_float(r4_23[2]));
            chains = fma_f32x2_rn_noftz(d_35, a_31, chains);
            int slot_28 = 768 + lane_1;
            unsigned int r4_24[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_24[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_28)));
            float2 d_36 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_24[0]));
            float2 a_32 = make_float2(__uint_as_float(r4_24[1]), __uint_as_float(r4_24[2]));
            chains = fma_f32x2_rn_noftz(d_36, a_32, chains);
            int slot_29 = 800 + lane_1;
            unsigned int r4_25[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_25[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_29)));
            float2 d_37 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_25[0]));
            float2 a_33 = make_float2(__uint_as_float(r4_25[1]), __uint_as_float(r4_25[2]));
            chains = fma_f32x2_rn_noftz(d_37, a_33, chains);
            int slot_30 = 832 + lane_1;
            unsigned int r4_26[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_26[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_30)));
            float2 d_38 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_26[0]));
            float2 a_34 = make_float2(__uint_as_float(r4_26[1]), __uint_as_float(r4_26[2]));
            chains = fma_f32x2_rn_noftz(d_38, a_34, chains);
            int slot_31 = 864 + lane_1;
            unsigned int r4_27[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_27[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_31)));
            float2 d_39 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_27[0]));
            float2 a_35 = make_float2(__uint_as_float(r4_27[1]), __uint_as_float(r4_27[2]));
            chains = fma_f32x2_rn_noftz(d_39, a_35, chains);
            int slot_32 = 896 + lane_1;
            unsigned int r4_28[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_28[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_32)));
            float2 d_40 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_28[0]));
            float2 a_36 = make_float2(__uint_as_float(r4_28[1]), __uint_as_float(r4_28[2]));
            chains = fma_f32x2_rn_noftz(d_40, a_36, chains);
            int slot_33 = 928 + lane_1;
            unsigned int r4_29[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_29[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_33)));
            float2 d_41 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_29[0]));
            float2 a_37 = make_float2(__uint_as_float(r4_29[1]), __uint_as_float(r4_29[2]));
            chains = fma_f32x2_rn_noftz(d_41, a_37, chains);
            int slot_34 = 960 + lane_1;
            unsigned int r4_30[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_30[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_34)));
            float2 d_42 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_30[0]));
            float2 a_38 = make_float2(__uint_as_float(r4_30[1]), __uint_as_float(r4_30[2]));
            chains = fma_f32x2_rn_noftz(d_42, a_38, chains);
            int slot_35 = 992 + lane_1;
            unsigned int r4_31[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r4_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_31[(0) + 3]))
                : "r"(rec_addr + (unsigned int)(16 * slot_35)));
            float2 d_43 = cake_bf16x2_to_f32x2(__as_bf16x2(r4_31[0]));
            float2 a_39 = make_float2(__uint_as_float(r4_31[1]), __uint_as_float(r4_31[2]));
            chains = fma_f32x2_rn_noftz(d_43, a_39, chains);
            float total;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(total) : "f"(chains.x), "f"(chains.y));
            float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, total, 16);
            total = total + _shfl_xor_sync_109;
            float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
            total = total + _shfl_xor_sync_110;
            float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
            total = total + _shfl_xor_sync_111;
            float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_sync_112;
            float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
            total = total + _shfl_xor_sync_113;
            if (lane_1 == 0) {
                *(reinterpret_cast<float*>(d_weight + row) + (0)) = total;
            }
        }
        __syncthreads();
    }
}

} // extern "C"
