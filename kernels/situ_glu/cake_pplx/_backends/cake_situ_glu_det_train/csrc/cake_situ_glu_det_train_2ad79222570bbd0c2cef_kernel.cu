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


__device__ __forceinline__ float approx_exp2_noftz(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
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
kernel_cake_situ_glu_det_train_2ad79222570bbd0c2cef(__nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ dh, __nv_bfloat16* __restrict__ d_gate, __nv_bfloat16* __restrict__ d_up, int rows, int width, long long gate_stride, long long up_stride, long long dh_stride, long long d_gate_stride, long long d_up_stride, float beta, float lb)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int part = warp_1 & 3;
    int sub = warp_1 >> 2;
    int nseg = width >> 8;
    float k1 = -2.88539f / beta;
    float k2 = -2.88539f / lb;
    float bl = beta * lb;
    float cg = k1 * 0.6931472f;
    float cu = k2 * 0.6931472f;
    float nb = -beta;
    float nl = -lb;
    float sdb = beta * 0.99999905f;
    float sdl = lb * 0.99999905f;
    #pragma unroll 1
    for (int first = bid; first < rows; first += num_bids) {
        int row = first + sub;
        int limit = ((row < rows) ? nseg : 0);
        int _min_0 = ((row) < (rows - 1) ? (row) : (rows - 1));
        long long at = (long long)_min_0;
        long long goff = at * gate_stride;
        long long uoff = at * up_stride;
        long long doff = at * dh_stride;
        long long dgoff = at * d_gate_stride;
        long long duoff = at * d_up_stride;
        #pragma unroll 1
        for (int s0 = part; s0 < nseg; s0 += 4) {
            unsigned int gw[4];
            unsigned int uw[4];
            unsigned int dw[4];
            int seg_0 = ((s0) < (nseg - 1) ? (s0) : (nseg - 1));
            long long col_0 = (long long)(seg_0 * 256 + lane_1 * 8);
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
            if (beta == 4.0f) {
                float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
                float2 v_0 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
                float2 d_0 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
                float _abs_4;
                asm("abs.f32 %0, %1;" : "=f"(_abs_4) : "f"(x_0.x));
                float _abs_5;
                asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(x_0.y));
                float2 ax = make_float2(_abs_4, _abs_5);
                float _abs_6;
                asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(v_0.x));
                float _abs_7;
                asm("abs.f32 %0, %1;" : "=f"(_abs_7) : "f"(v_0.y));
                float2 av = make_float2(_abs_6, _abs_7);
                float2 _f32x2_mul_b_8 = make_float2(k1, k1);
                float2 te;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                float2 _f32x2_mul_b_9 = make_float2(k2, k2);
                float2 tw;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw) : "l"(*(const unsigned long long*)&av), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
                float _exp2_10 = approx_exp2(te.x);
                float _exp2_11 = approx_exp2(te.y);
                float2 e = make_float2(_exp2_10, _exp2_11);
                float _exp2_12 = approx_exp2(tw.x);
                float _exp2_13 = approx_exp2(tw.y);
                float2 w = make_float2(_exp2_12, _exp2_13);
                float2 p;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p) : "l"(*(const unsigned long long*)&e), "l"(*(const unsigned long long*)&e));
                float2 d1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                float2 d2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p);
                float2 dw_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                float2 ng = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                float2 nu = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                float _abs_14;
                asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(x_0.x));
                float _abs_15;
                asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(x_0.y));
                float2 ax_0 = make_float2(_abs_14, _abs_15);
                float _abs_16;
                asm("abs.f32 %0, %1;" : "=f"(_abs_16) : "f"(v_0.x));
                float _abs_17;
                asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(v_0.y));
                float2 av_0 = make_float2(_abs_16, _abs_17);
                float2 d12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12) : "l"(*(const unsigned long long*)&d1), "l"(*(const unsigned long long*)&d2));
                float both = approx_rcp(mul_f32x2_noftz(d12, dw_0).x * mul_f32x2_noftz(d12, dw_0).y);
                float2 _f32x2_mul_a_18 = make_float2(mul_f32x2_noftz(d12, dw_0).y, mul_f32x2_noftz(d12, dw_0).x);
                float2 _f32x2_mul_b_19 = make_float2(both, both);
                float2 r_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
                float2 r = r_0;
                float2 _f32x2_mul_b_20 = mul_f32x2_noftz(r, r);
                float2 m1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1) : "l"(*(const unsigned long long*)&d_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
                float2 _f32x2_mul_b_21 = fma_f32x2_rn_noftz(ax_0, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
                float2 far = fma_f32x2_rn_noftz(e, make_float2(nb, nb), make_float2(sdb, sdb));
                unsigned int lo = __as_u32(make_float2(max_noftz(near.x, far.x), max_noftz(near.y, far.y)).x) | __as_u32(x_0.x) & 2147483648u;
                unsigned int hi = __as_u32(make_float2(max_noftz(near.x, far.x), max_noftz(near.y, far.y)).y) | __as_u32(x_0.y) & 2147483648u;
                float2 bng = make_float2(__uint_as_float(lo), __uint_as_float(hi));
                float2 _f32x2_mul_b_22 = fma_f32x2_rn_noftz(av_0, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_0) : "l"(*(const unsigned long long*)&av_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
                float2 far_0 = fma_f32x2_rn_noftz(w, make_float2(nl, nl), make_float2(sdl, sdl));
                unsigned int lo_0 = __as_u32(make_float2(max_noftz(near_0.x, far_0.x), max_noftz(near_0.y, far_0.y)).x) | __as_u32(v_0.x) & 2147483648u;
                unsigned int hi_0 = __as_u32(make_float2(max_noftz(near_0.x, far_0.x), max_noftz(near_0.y, far_0.y)).y) | __as_u32(v_0.y) & 2147483648u;
                float2 lnu = make_float2(__uint_as_float(lo_0), __uint_as_float(hi_0));
                float2 q = make_float2(((x_0.x < 0.0f) ? 1.0f : p.x), ((x_0.y < 0.0f) ? 1.0f : p.y));
                float2 post = make_float2(((x_0.x < 0.0f) ? p.x : 1.0f), ((x_0.y < 0.0f) ? p.y : 1.0f));
                float2 inner = fma_f32x2_rn_noftz(bng, mul_f32x2_noftz(d1, q), mul_f32x2_noftz(mul_f32x2_noftz(e, make_float2(4.0f, 4.0f)), d2));
                float2 _f32x2_mul_a_23 = mul_f32x2_noftz(mul_f32x2_noftz(m1, lnu), mul_f32x2_noftz(dw_0, inner));
                float2 dg;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&post));
                float2 _f32x2_mul_a_24 = mul_f32x2_noftz(mul_f32x2_noftz(m1, bng), mul_f32x2_noftz(mul_f32x2_noftz(w, make_float2(4.0f, 4.0f)), d12));
                float2 du;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&post));
                gw[0] = __as_u32(__float22bfloat162_rn(dg));
                uw[0] = __as_u32(__float22bfloat162_rn(du));
                float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
                float2 v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
                float2 d_1 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[1]));
                float _abs_25;
                asm("abs.f32 %0, %1;" : "=f"(_abs_25) : "f"(x_1.x));
                float _abs_26;
                asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(x_1.y));
                float2 ax_1 = make_float2(_abs_25, _abs_26);
                float _abs_27;
                asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(v_1.x));
                float _abs_28;
                asm("abs.f32 %0, %1;" : "=f"(_abs_28) : "f"(v_1.y));
                float2 av_1 = make_float2(_abs_27, _abs_28);
                float2 _f32x2_mul_b_29 = make_float2(k1, k1);
                float2 te_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_0) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
                float2 _f32x2_mul_b_30 = make_float2(k2, k2);
                float2 tw_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_0) : "l"(*(const unsigned long long*)&av_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
                float _exp2_31 = approx_exp2(te_0.x);
                float _exp2_32 = approx_exp2(te_0.y);
                float2 e_0 = make_float2(_exp2_31, _exp2_32);
                float _exp2_33 = approx_exp2(tw_0.x);
                float _exp2_34 = approx_exp2(tw_0.y);
                float2 w_0 = make_float2(_exp2_33, _exp2_34);
                float2 p_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_0) : "l"(*(const unsigned long long*)&e_0), "l"(*(const unsigned long long*)&e_0));
                float2 d1_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                float2 d2_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_0);
                float2 dw_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                float2 ng_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                float2 nu_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                float _abs_35;
                asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(x_1.x));
                float _abs_36;
                asm("abs.f32 %0, %1;" : "=f"(_abs_36) : "f"(x_1.y));
                float2 ax_2 = make_float2(_abs_35, _abs_36);
                float _abs_37;
                asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(v_1.x));
                float _abs_38;
                asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(v_1.y));
                float2 av_2 = make_float2(_abs_37, _abs_38);
                float2 d12_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_0) : "l"(*(const unsigned long long*)&d1_0), "l"(*(const unsigned long long*)&d2_0));
                float both_0 = approx_rcp(mul_f32x2_noftz(d12_0, dw_1).x * mul_f32x2_noftz(d12_0, dw_1).y);
                float2 _f32x2_mul_a_39 = make_float2(mul_f32x2_noftz(d12_0, dw_1).y, mul_f32x2_noftz(d12_0, dw_1).x);
                float2 _f32x2_mul_b_40 = make_float2(both_0, both_0);
                float2 r_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
                float2 r_1 = r_2;
                float2 _f32x2_mul_b_41 = mul_f32x2_noftz(r_1, r_1);
                float2 m1_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_0) : "l"(*(const unsigned long long*)&d_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
                float2 _f32x2_mul_b_42 = fma_f32x2_rn_noftz(ax_2, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_1) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
                float2 far_1 = fma_f32x2_rn_noftz(e_0, make_float2(nb, nb), make_float2(sdb, sdb));
                unsigned int lo_1 = __as_u32(make_float2(max_noftz(near_1.x, far_1.x), max_noftz(near_1.y, far_1.y)).x) | __as_u32(x_1.x) & 2147483648u;
                unsigned int hi_1 = __as_u32(make_float2(max_noftz(near_1.x, far_1.x), max_noftz(near_1.y, far_1.y)).y) | __as_u32(x_1.y) & 2147483648u;
                float2 bng_0 = make_float2(__uint_as_float(lo_1), __uint_as_float(hi_1));
                float2 _f32x2_mul_b_43 = fma_f32x2_rn_noftz(av_2, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_2) : "l"(*(const unsigned long long*)&av_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                float2 far_2 = fma_f32x2_rn_noftz(w_0, make_float2(nl, nl), make_float2(sdl, sdl));
                unsigned int lo_2 = __as_u32(make_float2(max_noftz(near_2.x, far_2.x), max_noftz(near_2.y, far_2.y)).x) | __as_u32(v_1.x) & 2147483648u;
                unsigned int hi_2 = __as_u32(make_float2(max_noftz(near_2.x, far_2.x), max_noftz(near_2.y, far_2.y)).y) | __as_u32(v_1.y) & 2147483648u;
                float2 lnu_0 = make_float2(__uint_as_float(lo_2), __uint_as_float(hi_2));
                float2 q_0 = make_float2(((x_1.x < 0.0f) ? 1.0f : p_0.x), ((x_1.y < 0.0f) ? 1.0f : p_0.y));
                float2 post_0 = make_float2(((x_1.x < 0.0f) ? p_0.x : 1.0f), ((x_1.y < 0.0f) ? p_0.y : 1.0f));
                float2 inner_0 = fma_f32x2_rn_noftz(bng_0, mul_f32x2_noftz(d1_0, q_0), mul_f32x2_noftz(mul_f32x2_noftz(e_0, make_float2(4.0f, 4.0f)), d2_0));
                float2 _f32x2_mul_a_44 = mul_f32x2_noftz(mul_f32x2_noftz(m1_0, lnu_0), mul_f32x2_noftz(dw_1, inner_0));
                float2 dg_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&post_0));
                float2 _f32x2_mul_a_45 = mul_f32x2_noftz(mul_f32x2_noftz(m1_0, bng_0), mul_f32x2_noftz(mul_f32x2_noftz(w_0, make_float2(4.0f, 4.0f)), d12_0));
                float2 du_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&post_0));
                gw[1] = __as_u32(__float22bfloat162_rn(dg_0));
                uw[1] = __as_u32(__float22bfloat162_rn(du_0));
                float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
                float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
                float2 d_2 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
                float _abs_46;
                asm("abs.f32 %0, %1;" : "=f"(_abs_46) : "f"(x_2.x));
                float _abs_47;
                asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(x_2.y));
                float2 ax_3 = make_float2(_abs_46, _abs_47);
                float _abs_48;
                asm("abs.f32 %0, %1;" : "=f"(_abs_48) : "f"(v_2.x));
                float _abs_49;
                asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(v_2.y));
                float2 av_3 = make_float2(_abs_48, _abs_49);
                float2 _f32x2_mul_b_50 = make_float2(k1, k1);
                float2 te_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_1) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
                float2 _f32x2_mul_b_51 = make_float2(k2, k2);
                float2 tw_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_1) : "l"(*(const unsigned long long*)&av_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
                float _exp2_52 = approx_exp2(te_1.x);
                float _exp2_53 = approx_exp2(te_1.y);
                float2 e_1 = make_float2(_exp2_52, _exp2_53);
                float _exp2_54 = approx_exp2(tw_1.x);
                float _exp2_55 = approx_exp2(tw_1.y);
                float2 w_1 = make_float2(_exp2_54, _exp2_55);
                float2 p_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_1) : "l"(*(const unsigned long long*)&e_1), "l"(*(const unsigned long long*)&e_1));
                float2 d1_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                float2 d2_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_1);
                float2 dw_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                float2 ng_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                float2 nu_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                float _abs_56;
                asm("abs.f32 %0, %1;" : "=f"(_abs_56) : "f"(x_2.x));
                float _abs_57;
                asm("abs.f32 %0, %1;" : "=f"(_abs_57) : "f"(x_2.y));
                float2 ax_4 = make_float2(_abs_56, _abs_57);
                float _abs_58;
                asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(v_2.x));
                float _abs_59;
                asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(v_2.y));
                float2 av_4 = make_float2(_abs_58, _abs_59);
                float2 d12_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_1) : "l"(*(const unsigned long long*)&d1_1), "l"(*(const unsigned long long*)&d2_1));
                float both_1 = approx_rcp(mul_f32x2_noftz(d12_1, dw_2).x * mul_f32x2_noftz(d12_1, dw_2).y);
                float2 _f32x2_mul_a_60 = make_float2(mul_f32x2_noftz(d12_1, dw_2).y, mul_f32x2_noftz(d12_1, dw_2).x);
                float2 _f32x2_mul_b_61 = make_float2(both_1, both_1);
                float2 r_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
                float2 r_3 = r_4;
                float2 _f32x2_mul_b_62 = mul_f32x2_noftz(r_3, r_3);
                float2 m1_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_1) : "l"(*(const unsigned long long*)&d_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
                float2 _f32x2_mul_b_63 = fma_f32x2_rn_noftz(ax_4, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_3) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
                float2 far_3 = fma_f32x2_rn_noftz(e_1, make_float2(nb, nb), make_float2(sdb, sdb));
                unsigned int lo_3 = __as_u32(make_float2(max_noftz(near_3.x, far_3.x), max_noftz(near_3.y, far_3.y)).x) | __as_u32(x_2.x) & 2147483648u;
                unsigned int hi_3 = __as_u32(make_float2(max_noftz(near_3.x, far_3.x), max_noftz(near_3.y, far_3.y)).y) | __as_u32(x_2.y) & 2147483648u;
                float2 bng_1 = make_float2(__uint_as_float(lo_3), __uint_as_float(hi_3));
                float2 _f32x2_mul_b_64 = fma_f32x2_rn_noftz(av_4, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_4) : "l"(*(const unsigned long long*)&av_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
                float2 far_4 = fma_f32x2_rn_noftz(w_1, make_float2(nl, nl), make_float2(sdl, sdl));
                unsigned int lo_4 = __as_u32(make_float2(max_noftz(near_4.x, far_4.x), max_noftz(near_4.y, far_4.y)).x) | __as_u32(v_2.x) & 2147483648u;
                unsigned int hi_4 = __as_u32(make_float2(max_noftz(near_4.x, far_4.x), max_noftz(near_4.y, far_4.y)).y) | __as_u32(v_2.y) & 2147483648u;
                float2 lnu_1 = make_float2(__uint_as_float(lo_4), __uint_as_float(hi_4));
                float2 q_1 = make_float2(((x_2.x < 0.0f) ? 1.0f : p_1.x), ((x_2.y < 0.0f) ? 1.0f : p_1.y));
                float2 post_1 = make_float2(((x_2.x < 0.0f) ? p_1.x : 1.0f), ((x_2.y < 0.0f) ? p_1.y : 1.0f));
                float2 inner_1 = fma_f32x2_rn_noftz(bng_1, mul_f32x2_noftz(d1_1, q_1), mul_f32x2_noftz(mul_f32x2_noftz(e_1, make_float2(4.0f, 4.0f)), d2_1));
                float2 _f32x2_mul_a_65 = mul_f32x2_noftz(mul_f32x2_noftz(m1_1, lnu_1), mul_f32x2_noftz(dw_2, inner_1));
                float2 dg_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&post_1));
                float2 _f32x2_mul_a_66 = mul_f32x2_noftz(mul_f32x2_noftz(m1_1, bng_1), mul_f32x2_noftz(mul_f32x2_noftz(w_1, make_float2(4.0f, 4.0f)), d12_1));
                float2 du_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&post_1));
                gw[2] = __as_u32(__float22bfloat162_rn(dg_1));
                uw[2] = __as_u32(__float22bfloat162_rn(du_1));
                float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
                float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
                float2 d_3 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
                float _abs_67;
                asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(x_3.x));
                float _abs_68;
                asm("abs.f32 %0, %1;" : "=f"(_abs_68) : "f"(x_3.y));
                float2 ax_5 = make_float2(_abs_67, _abs_68);
                float _abs_69;
                asm("abs.f32 %0, %1;" : "=f"(_abs_69) : "f"(v_3.x));
                float _abs_70;
                asm("abs.f32 %0, %1;" : "=f"(_abs_70) : "f"(v_3.y));
                float2 av_5 = make_float2(_abs_69, _abs_70);
                float2 _f32x2_mul_b_71 = make_float2(k1, k1);
                float2 te_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_2) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                float2 _f32x2_mul_b_72 = make_float2(k2, k2);
                float2 tw_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_2) : "l"(*(const unsigned long long*)&av_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_72));
                float _exp2_73 = approx_exp2(te_2.x);
                float _exp2_74 = approx_exp2(te_2.y);
                float2 e_2 = make_float2(_exp2_73, _exp2_74);
                float _exp2_75 = approx_exp2(tw_2.x);
                float _exp2_76 = approx_exp2(tw_2.y);
                float2 w_2 = make_float2(_exp2_75, _exp2_76);
                float2 p_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_2) : "l"(*(const unsigned long long*)&e_2), "l"(*(const unsigned long long*)&e_2));
                float2 d1_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                float2 d2_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_2);
                float2 dw_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                float2 ng_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                float2 nu_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                float _abs_77;
                asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(x_3.x));
                float _abs_78;
                asm("abs.f32 %0, %1;" : "=f"(_abs_78) : "f"(x_3.y));
                float2 ax_6 = make_float2(_abs_77, _abs_78);
                float _abs_79;
                asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(v_3.x));
                float _abs_80;
                asm("abs.f32 %0, %1;" : "=f"(_abs_80) : "f"(v_3.y));
                float2 av_6 = make_float2(_abs_79, _abs_80);
                float2 d12_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_2) : "l"(*(const unsigned long long*)&d1_2), "l"(*(const unsigned long long*)&d2_2));
                float both_2 = approx_rcp(mul_f32x2_noftz(d12_2, dw_3).x * mul_f32x2_noftz(d12_2, dw_3).y);
                float2 _f32x2_mul_a_81 = make_float2(mul_f32x2_noftz(d12_2, dw_3).y, mul_f32x2_noftz(d12_2, dw_3).x);
                float2 _f32x2_mul_b_82 = make_float2(both_2, both_2);
                float2 r_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
                float2 r_5 = r_6;
                float2 _f32x2_mul_b_83 = mul_f32x2_noftz(r_5, r_5);
                float2 m1_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_2) : "l"(*(const unsigned long long*)&d_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
                float2 _f32x2_mul_b_84 = fma_f32x2_rn_noftz(ax_6, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_5) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_84));
                float2 far_5 = fma_f32x2_rn_noftz(e_2, make_float2(nb, nb), make_float2(sdb, sdb));
                unsigned int lo_5 = __as_u32(make_float2(max_noftz(near_5.x, far_5.x), max_noftz(near_5.y, far_5.y)).x) | __as_u32(x_3.x) & 2147483648u;
                unsigned int hi_5 = __as_u32(make_float2(max_noftz(near_5.x, far_5.x), max_noftz(near_5.y, far_5.y)).y) | __as_u32(x_3.y) & 2147483648u;
                float2 bng_2 = make_float2(__uint_as_float(lo_5), __uint_as_float(hi_5));
                float2 _f32x2_mul_b_85 = fma_f32x2_rn_noftz(av_6, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_6) : "l"(*(const unsigned long long*)&av_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
                float2 far_6 = fma_f32x2_rn_noftz(w_2, make_float2(nl, nl), make_float2(sdl, sdl));
                unsigned int lo_6 = __as_u32(make_float2(max_noftz(near_6.x, far_6.x), max_noftz(near_6.y, far_6.y)).x) | __as_u32(v_3.x) & 2147483648u;
                unsigned int hi_6 = __as_u32(make_float2(max_noftz(near_6.x, far_6.x), max_noftz(near_6.y, far_6.y)).y) | __as_u32(v_3.y) & 2147483648u;
                float2 lnu_2 = make_float2(__uint_as_float(lo_6), __uint_as_float(hi_6));
                float2 q_2 = make_float2(((x_3.x < 0.0f) ? 1.0f : p_2.x), ((x_3.y < 0.0f) ? 1.0f : p_2.y));
                float2 post_2 = make_float2(((x_3.x < 0.0f) ? p_2.x : 1.0f), ((x_3.y < 0.0f) ? p_2.y : 1.0f));
                float2 inner_2 = fma_f32x2_rn_noftz(bng_2, mul_f32x2_noftz(d1_2, q_2), mul_f32x2_noftz(mul_f32x2_noftz(e_2, make_float2(4.0f, 4.0f)), d2_2));
                float2 _f32x2_mul_a_86 = mul_f32x2_noftz(mul_f32x2_noftz(m1_2, lnu_2), mul_f32x2_noftz(dw_3, inner_2));
                float2 dg_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&post_2));
                float2 _f32x2_mul_a_87 = mul_f32x2_noftz(mul_f32x2_noftz(m1_2, bng_2), mul_f32x2_noftz(mul_f32x2_noftz(w_2, make_float2(4.0f, 4.0f)), d12_2));
                float2 du_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&post_2));
                gw[3] = __as_u32(__float22bfloat162_rn(dg_2));
                uw[3] = __as_u32(__float22bfloat162_rn(du_2));
            } else {
                float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
                float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
                float2 d_4 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
                float _abs_88;
                asm("abs.f32 %0, %1;" : "=f"(_abs_88) : "f"(x_4.x));
                float _abs_89;
                asm("abs.f32 %0, %1;" : "=f"(_abs_89) : "f"(x_4.y));
                float2 ax_7 = make_float2(_abs_88, _abs_89);
                float _abs_90;
                asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(v_4.x));
                float _abs_91;
                asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(v_4.y));
                float2 av_7 = make_float2(_abs_90, _abs_91);
                float2 _f32x2_mul_b_92 = make_float2(k1, k1);
                float2 te_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
                float2 _f32x2_mul_b_93 = make_float2(k2, k2);
                float2 tw_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3) : "l"(*(const unsigned long long*)&av_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
                float _exp2_94 = approx_exp2(te_3.x);
                float _exp2_95 = approx_exp2(te_3.y);
                float2 e_3 = make_float2(_exp2_94, _exp2_95);
                float _exp2_96 = approx_exp2(tw_3.x);
                float _exp2_97 = approx_exp2(tw_3.y);
                float2 w_3 = make_float2(_exp2_96, _exp2_97);
                float2 _f32x2_mul_b_98 = make_float2(-1.442695f, -1.442695f);
                float2 tp;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
                float _exp2_99 = approx_exp2_noftz(tp.x);
                float _exp2_100 = approx_exp2_noftz(tp.y);
                float2 p_3 = make_float2(_exp2_99, _exp2_100);
                float2 d1_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                float2 d2_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3);
                float2 dw_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                float2 ng_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                float2 nu_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                float _abs_101;
                asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(x_4.x));
                float _abs_102;
                asm("abs.f32 %0, %1;" : "=f"(_abs_102) : "f"(x_4.y));
                float2 ax_7_1 = make_float2(_abs_101, _abs_102);
                float _abs_103;
                asm("abs.f32 %0, %1;" : "=f"(_abs_103) : "f"(v_4.x));
                float _abs_104;
                asm("abs.f32 %0, %1;" : "=f"(_abs_104) : "f"(v_4.y));
                float2 av_7_1 = make_float2(_abs_103, _abs_104);
                float2 d12_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_3) : "l"(*(const unsigned long long*)&d1_3), "l"(*(const unsigned long long*)&d2_3));
                float both_3 = approx_rcp(mul_f32x2_noftz(d12_3, dw_4).x * mul_f32x2_noftz(d12_3, dw_4).y);
                float2 _f32x2_mul_a_105 = make_float2(mul_f32x2_noftz(d12_3, dw_4).y, mul_f32x2_noftz(d12_3, dw_4).x);
                float2 _f32x2_mul_b_106 = make_float2(both_3, both_3);
                float2 r_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
                float2 r_8 = r_7;
                float2 _f32x2_mul_b_107 = mul_f32x2_noftz(r_8, r_8);
                float2 m1_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_3) : "l"(*(const unsigned long long*)&d_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_107));
                float2 _f32x2_mul_b_108 = fma_f32x2_rn_noftz(ax_7_1, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_7) : "l"(*(const unsigned long long*)&ax_7_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_108));
                float2 far_7 = fma_f32x2_rn_noftz(e_3, make_float2(nb, nb), make_float2(sdb, sdb));
                unsigned int lo_7 = __as_u32(make_float2(max_noftz(near_7.x, far_7.x), max_noftz(near_7.y, far_7.y)).x) | __as_u32(x_4.x) & 2147483648u;
                unsigned int hi_7 = __as_u32(make_float2(max_noftz(near_7.x, far_7.x), max_noftz(near_7.y, far_7.y)).y) | __as_u32(x_4.y) & 2147483648u;
                float2 bng_3 = make_float2(__uint_as_float(lo_7), __uint_as_float(hi_7));
                float2 _f32x2_mul_b_109 = fma_f32x2_rn_noftz(av_7_1, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_7_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_7_1) : "l"(*(const unsigned long long*)&av_7_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_109));
                float2 far_7_1 = fma_f32x2_rn_noftz(w_3, make_float2(nl, nl), make_float2(sdl, sdl));
                unsigned int lo_7_1 = __as_u32(make_float2(max_noftz(near_7_1.x, far_7_1.x), max_noftz(near_7_1.y, far_7_1.y)).x) | __as_u32(v_4.x) & 2147483648u;
                unsigned int hi_7_1 = __as_u32(make_float2(max_noftz(near_7_1.x, far_7_1.x), max_noftz(near_7_1.y, far_7_1.y)).y) | __as_u32(v_4.y) & 2147483648u;
                float2 lnu_3 = make_float2(__uint_as_float(lo_7_1), __uint_as_float(hi_7_1));
                float2 q_3 = make_float2(((x_4.x < 0.0f) ? 1.0f : p_3.x), ((x_4.y < 0.0f) ? 1.0f : p_3.y));
                float2 post_3 = make_float2(((x_4.x < 0.0f) ? p_3.x : 1.0f), ((x_4.y < 0.0f) ? p_3.y : 1.0f));
                float2 inner_3 = fma_f32x2_rn_noftz(bng_3, mul_f32x2_noftz(d1_3, q_3), mul_f32x2_noftz(mul_f32x2_noftz(e_3, make_float2(4.0f, 4.0f)), d2_3));
                float2 _f32x2_mul_a_110 = mul_f32x2_noftz(mul_f32x2_noftz(m1_3, lnu_3), mul_f32x2_noftz(dw_4, inner_3));
                float2 dg_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_110), "l"(*(const unsigned long long*)&post_3));
                float2 _f32x2_mul_a_111 = mul_f32x2_noftz(mul_f32x2_noftz(m1_3, bng_3), mul_f32x2_noftz(mul_f32x2_noftz(w_3, make_float2(4.0f, 4.0f)), d12_3));
                float2 du_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&post_3));
                gw[0] = __as_u32(__float22bfloat162_rn(dg_3));
                uw[0] = __as_u32(__float22bfloat162_rn(du_3));
                float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
                float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
                float2 d_5 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[1]));
                float _abs_112;
                asm("abs.f32 %0, %1;" : "=f"(_abs_112) : "f"(x_5.x));
                float _abs_113;
                asm("abs.f32 %0, %1;" : "=f"(_abs_113) : "f"(x_5.y));
                float2 ax_8 = make_float2(_abs_112, _abs_113);
                float _abs_114;
                asm("abs.f32 %0, %1;" : "=f"(_abs_114) : "f"(v_5.x));
                float _abs_115;
                asm("abs.f32 %0, %1;" : "=f"(_abs_115) : "f"(v_5.y));
                float2 av_8 = make_float2(_abs_114, _abs_115);
                float2 _f32x2_mul_b_116 = make_float2(k1, k1);
                float2 te_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3_1) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
                float2 _f32x2_mul_b_117 = make_float2(k2, k2);
                float2 tw_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3_1) : "l"(*(const unsigned long long*)&av_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_117));
                float _exp2_118 = approx_exp2(te_3_1.x);
                float _exp2_119 = approx_exp2(te_3_1.y);
                float2 e_3_1 = make_float2(_exp2_118, _exp2_119);
                float _exp2_120 = approx_exp2(tw_3_1.x);
                float _exp2_121 = approx_exp2(tw_3_1.y);
                float2 w_3_1 = make_float2(_exp2_120, _exp2_121);
                float2 _f32x2_mul_b_122 = make_float2(-1.442695f, -1.442695f);
                float2 tp_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_0) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
                float _exp2_123 = approx_exp2_noftz(tp_0.x);
                float _exp2_124 = approx_exp2_noftz(tp_0.y);
                float2 p_3_1 = make_float2(_exp2_123, _exp2_124);
                float2 d1_3_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3_1);
                float2 d2_3_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3_1);
                float2 dw_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3_1);
                float2 ng_3_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3_1);
                float2 nu_3_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3_1);
                float _abs_125;
                asm("abs.f32 %0, %1;" : "=f"(_abs_125) : "f"(x_5.x));
                float _abs_126;
                asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(x_5.y));
                float2 ax_9 = make_float2(_abs_125, _abs_126);
                float _abs_127;
                asm("abs.f32 %0, %1;" : "=f"(_abs_127) : "f"(v_5.x));
                float _abs_128;
                asm("abs.f32 %0, %1;" : "=f"(_abs_128) : "f"(v_5.y));
                float2 av_9 = make_float2(_abs_127, _abs_128);
                float2 d12_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_3_1) : "l"(*(const unsigned long long*)&d1_3_1), "l"(*(const unsigned long long*)&d2_3_1));
                float both_3_1 = approx_rcp(mul_f32x2_noftz(d12_3_1, dw_5).x * mul_f32x2_noftz(d12_3_1, dw_5).y);
                float2 _f32x2_mul_a_129 = make_float2(mul_f32x2_noftz(d12_3_1, dw_5).y, mul_f32x2_noftz(d12_3_1, dw_5).x);
                float2 _f32x2_mul_b_130 = make_float2(both_3_1, both_3_1);
                float2 r_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&_f32x2_mul_b_130));
                float2 r_8_1 = r_9;
                float2 _f32x2_mul_b_131 = mul_f32x2_noftz(r_8_1, r_8_1);
                float2 m1_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_3_1) : "l"(*(const unsigned long long*)&d_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
                float2 _f32x2_mul_b_132 = fma_f32x2_rn_noftz(ax_9, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_8) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_132));
                float2 far_8 = fma_f32x2_rn_noftz(e_3_1, make_float2(nb, nb), make_float2(sdb, sdb));
                unsigned int lo_8 = __as_u32(make_float2(max_noftz(near_8.x, far_8.x), max_noftz(near_8.y, far_8.y)).x) | __as_u32(x_5.x) & 2147483648u;
                unsigned int hi_8 = __as_u32(make_float2(max_noftz(near_8.x, far_8.x), max_noftz(near_8.y, far_8.y)).y) | __as_u32(x_5.y) & 2147483648u;
                float2 bng_3_1 = make_float2(__uint_as_float(lo_8), __uint_as_float(hi_8));
                float2 _f32x2_mul_b_133 = fma_f32x2_rn_noftz(av_9, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_9) : "l"(*(const unsigned long long*)&av_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
                float2 far_9 = fma_f32x2_rn_noftz(w_3_1, make_float2(nl, nl), make_float2(sdl, sdl));
                unsigned int lo_9 = __as_u32(make_float2(max_noftz(near_9.x, far_9.x), max_noftz(near_9.y, far_9.y)).x) | __as_u32(v_5.x) & 2147483648u;
                unsigned int hi_9 = __as_u32(make_float2(max_noftz(near_9.x, far_9.x), max_noftz(near_9.y, far_9.y)).y) | __as_u32(v_5.y) & 2147483648u;
                float2 lnu_3_1 = make_float2(__uint_as_float(lo_9), __uint_as_float(hi_9));
                float2 q_3_1 = make_float2(((x_5.x < 0.0f) ? 1.0f : p_3_1.x), ((x_5.y < 0.0f) ? 1.0f : p_3_1.y));
                float2 post_3_1 = make_float2(((x_5.x < 0.0f) ? p_3_1.x : 1.0f), ((x_5.y < 0.0f) ? p_3_1.y : 1.0f));
                float2 inner_3_1 = fma_f32x2_rn_noftz(bng_3_1, mul_f32x2_noftz(d1_3_1, q_3_1), mul_f32x2_noftz(mul_f32x2_noftz(e_3_1, make_float2(4.0f, 4.0f)), d2_3_1));
                float2 _f32x2_mul_a_134 = mul_f32x2_noftz(mul_f32x2_noftz(m1_3_1, lnu_3_1), mul_f32x2_noftz(dw_5, inner_3_1));
                float2 dg_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&post_3_1));
                float2 _f32x2_mul_a_135 = mul_f32x2_noftz(mul_f32x2_noftz(m1_3_1, bng_3_1), mul_f32x2_noftz(mul_f32x2_noftz(w_3_1, make_float2(4.0f, 4.0f)), d12_3_1));
                float2 du_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&post_3_1));
                gw[1] = __as_u32(__float22bfloat162_rn(dg_3_1));
                uw[1] = __as_u32(__float22bfloat162_rn(du_3_1));
                float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
                float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
                float2 d_6 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
                float _abs_136;
                asm("abs.f32 %0, %1;" : "=f"(_abs_136) : "f"(x_6.x));
                float _abs_137;
                asm("abs.f32 %0, %1;" : "=f"(_abs_137) : "f"(x_6.y));
                float2 ax_10 = make_float2(_abs_136, _abs_137);
                float _abs_138;
                asm("abs.f32 %0, %1;" : "=f"(_abs_138) : "f"(v_6.x));
                float _abs_139;
                asm("abs.f32 %0, %1;" : "=f"(_abs_139) : "f"(v_6.y));
                float2 av_10 = make_float2(_abs_138, _abs_139);
                float2 _f32x2_mul_b_140 = make_float2(k1, k1);
                float2 te_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_4) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_140));
                float2 _f32x2_mul_b_141 = make_float2(k2, k2);
                float2 tw_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_4) : "l"(*(const unsigned long long*)&av_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
                float _exp2_142 = approx_exp2(te_4.x);
                float _exp2_143 = approx_exp2(te_4.y);
                float2 e_4 = make_float2(_exp2_142, _exp2_143);
                float _exp2_144 = approx_exp2(tw_4.x);
                float _exp2_145 = approx_exp2(tw_4.y);
                float2 w_4 = make_float2(_exp2_144, _exp2_145);
                float2 _f32x2_mul_b_146 = make_float2(-1.442695f, -1.442695f);
                float2 tp_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_1) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_146));
                float _exp2_147 = approx_exp2_noftz(tp_1.x);
                float _exp2_148 = approx_exp2_noftz(tp_1.y);
                float2 p_4 = make_float2(_exp2_147, _exp2_148);
                float2 d1_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                float2 d2_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_4);
                float2 dw_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                float2 ng_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                float2 nu_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                float _abs_149;
                asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(x_6.x));
                float _abs_150;
                asm("abs.f32 %0, %1;" : "=f"(_abs_150) : "f"(x_6.y));
                float2 ax_11 = make_float2(_abs_149, _abs_150);
                float _abs_151;
                asm("abs.f32 %0, %1;" : "=f"(_abs_151) : "f"(v_6.x));
                float _abs_152;
                asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(v_6.y));
                float2 av_11 = make_float2(_abs_151, _abs_152);
                float2 d12_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_4) : "l"(*(const unsigned long long*)&d1_4), "l"(*(const unsigned long long*)&d2_4));
                float both_4 = approx_rcp(mul_f32x2_noftz(d12_4, dw_6).x * mul_f32x2_noftz(d12_4, dw_6).y);
                float2 _f32x2_mul_a_153 = make_float2(mul_f32x2_noftz(d12_4, dw_6).y, mul_f32x2_noftz(d12_4, dw_6).x);
                float2 _f32x2_mul_b_154 = make_float2(both_4, both_4);
                float2 r_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&_f32x2_mul_b_154));
                float2 r_10 = r_11;
                float2 _f32x2_mul_b_155 = mul_f32x2_noftz(r_10, r_10);
                float2 m1_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_4) : "l"(*(const unsigned long long*)&d_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
                float2 _f32x2_mul_b_156 = fma_f32x2_rn_noftz(ax_11, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_10) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_156));
                float2 far_10 = fma_f32x2_rn_noftz(e_4, make_float2(nb, nb), make_float2(sdb, sdb));
                unsigned int lo_10 = __as_u32(make_float2(max_noftz(near_10.x, far_10.x), max_noftz(near_10.y, far_10.y)).x) | __as_u32(x_6.x) & 2147483648u;
                unsigned int hi_10 = __as_u32(make_float2(max_noftz(near_10.x, far_10.x), max_noftz(near_10.y, far_10.y)).y) | __as_u32(x_6.y) & 2147483648u;
                float2 bng_4 = make_float2(__uint_as_float(lo_10), __uint_as_float(hi_10));
                float2 _f32x2_mul_b_157 = fma_f32x2_rn_noftz(av_11, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_11) : "l"(*(const unsigned long long*)&av_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
                float2 far_11 = fma_f32x2_rn_noftz(w_4, make_float2(nl, nl), make_float2(sdl, sdl));
                unsigned int lo_11 = __as_u32(make_float2(max_noftz(near_11.x, far_11.x), max_noftz(near_11.y, far_11.y)).x) | __as_u32(v_6.x) & 2147483648u;
                unsigned int hi_11 = __as_u32(make_float2(max_noftz(near_11.x, far_11.x), max_noftz(near_11.y, far_11.y)).y) | __as_u32(v_6.y) & 2147483648u;
                float2 lnu_4 = make_float2(__uint_as_float(lo_11), __uint_as_float(hi_11));
                float2 q_4 = make_float2(((x_6.x < 0.0f) ? 1.0f : p_4.x), ((x_6.y < 0.0f) ? 1.0f : p_4.y));
                float2 post_4 = make_float2(((x_6.x < 0.0f) ? p_4.x : 1.0f), ((x_6.y < 0.0f) ? p_4.y : 1.0f));
                float2 inner_4 = fma_f32x2_rn_noftz(bng_4, mul_f32x2_noftz(d1_4, q_4), mul_f32x2_noftz(mul_f32x2_noftz(e_4, make_float2(4.0f, 4.0f)), d2_4));
                float2 _f32x2_mul_a_158 = mul_f32x2_noftz(mul_f32x2_noftz(m1_4, lnu_4), mul_f32x2_noftz(dw_6, inner_4));
                float2 dg_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&post_4));
                float2 _f32x2_mul_a_159 = mul_f32x2_noftz(mul_f32x2_noftz(m1_4, bng_4), mul_f32x2_noftz(mul_f32x2_noftz(w_4, make_float2(4.0f, 4.0f)), d12_4));
                float2 du_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&post_4));
                gw[2] = __as_u32(__float22bfloat162_rn(dg_4));
                uw[2] = __as_u32(__float22bfloat162_rn(du_4));
                float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
                float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
                float2 d_7 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
                float _abs_160;
                asm("abs.f32 %0, %1;" : "=f"(_abs_160) : "f"(x_7.x));
                float _abs_161;
                asm("abs.f32 %0, %1;" : "=f"(_abs_161) : "f"(x_7.y));
                float2 ax_12 = make_float2(_abs_160, _abs_161);
                float _abs_162;
                asm("abs.f32 %0, %1;" : "=f"(_abs_162) : "f"(v_7.x));
                float _abs_163;
                asm("abs.f32 %0, %1;" : "=f"(_abs_163) : "f"(v_7.y));
                float2 av_12 = make_float2(_abs_162, _abs_163);
                float2 _f32x2_mul_b_164 = make_float2(k1, k1);
                float2 te_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_5) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_164));
                float2 _f32x2_mul_b_165 = make_float2(k2, k2);
                float2 tw_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_5) : "l"(*(const unsigned long long*)&av_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
                float _exp2_166 = approx_exp2(te_5.x);
                float _exp2_167 = approx_exp2(te_5.y);
                float2 e_5 = make_float2(_exp2_166, _exp2_167);
                float _exp2_168 = approx_exp2(tw_5.x);
                float _exp2_169 = approx_exp2(tw_5.y);
                float2 w_5 = make_float2(_exp2_168, _exp2_169);
                float2 _f32x2_mul_b_170 = make_float2(-1.442695f, -1.442695f);
                float2 tp_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_2) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_170));
                float _exp2_171 = approx_exp2_noftz(tp_2.x);
                float _exp2_172 = approx_exp2_noftz(tp_2.y);
                float2 p_5 = make_float2(_exp2_171, _exp2_172);
                float2 d1_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                float2 d2_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_5);
                float2 dw_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                float2 ng_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                float2 nu_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                float _abs_173;
                asm("abs.f32 %0, %1;" : "=f"(_abs_173) : "f"(x_7.x));
                float _abs_174;
                asm("abs.f32 %0, %1;" : "=f"(_abs_174) : "f"(x_7.y));
                float2 ax_13 = make_float2(_abs_173, _abs_174);
                float _abs_175;
                asm("abs.f32 %0, %1;" : "=f"(_abs_175) : "f"(v_7.x));
                float _abs_176;
                asm("abs.f32 %0, %1;" : "=f"(_abs_176) : "f"(v_7.y));
                float2 av_13 = make_float2(_abs_175, _abs_176);
                float2 d12_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_5) : "l"(*(const unsigned long long*)&d1_5), "l"(*(const unsigned long long*)&d2_5));
                float both_5 = approx_rcp(mul_f32x2_noftz(d12_5, dw_7).x * mul_f32x2_noftz(d12_5, dw_7).y);
                float2 _f32x2_mul_a_177 = make_float2(mul_f32x2_noftz(d12_5, dw_7).y, mul_f32x2_noftz(d12_5, dw_7).x);
                float2 _f32x2_mul_b_178 = make_float2(both_5, both_5);
                float2 r_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&_f32x2_mul_b_178));
                float2 r_12 = r_13;
                float2 _f32x2_mul_b_179 = mul_f32x2_noftz(r_12, r_12);
                float2 m1_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_5) : "l"(*(const unsigned long long*)&d_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_179));
                float2 _f32x2_mul_b_180 = fma_f32x2_rn_noftz(ax_13, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_12) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_180));
                float2 far_12 = fma_f32x2_rn_noftz(e_5, make_float2(nb, nb), make_float2(sdb, sdb));
                unsigned int lo_12 = __as_u32(make_float2(max_noftz(near_12.x, far_12.x), max_noftz(near_12.y, far_12.y)).x) | __as_u32(x_7.x) & 2147483648u;
                unsigned int hi_12 = __as_u32(make_float2(max_noftz(near_12.x, far_12.x), max_noftz(near_12.y, far_12.y)).y) | __as_u32(x_7.y) & 2147483648u;
                float2 bng_5 = make_float2(__uint_as_float(lo_12), __uint_as_float(hi_12));
                float2 _f32x2_mul_b_181 = fma_f32x2_rn_noftz(av_13, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_13) : "l"(*(const unsigned long long*)&av_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_181));
                float2 far_13 = fma_f32x2_rn_noftz(w_5, make_float2(nl, nl), make_float2(sdl, sdl));
                unsigned int lo_13 = __as_u32(make_float2(max_noftz(near_13.x, far_13.x), max_noftz(near_13.y, far_13.y)).x) | __as_u32(v_7.x) & 2147483648u;
                unsigned int hi_13 = __as_u32(make_float2(max_noftz(near_13.x, far_13.x), max_noftz(near_13.y, far_13.y)).y) | __as_u32(v_7.y) & 2147483648u;
                float2 lnu_5 = make_float2(__uint_as_float(lo_13), __uint_as_float(hi_13));
                float2 q_5 = make_float2(((x_7.x < 0.0f) ? 1.0f : p_5.x), ((x_7.y < 0.0f) ? 1.0f : p_5.y));
                float2 post_5 = make_float2(((x_7.x < 0.0f) ? p_5.x : 1.0f), ((x_7.y < 0.0f) ? p_5.y : 1.0f));
                float2 inner_5 = fma_f32x2_rn_noftz(bng_5, mul_f32x2_noftz(d1_5, q_5), mul_f32x2_noftz(mul_f32x2_noftz(e_5, make_float2(4.0f, 4.0f)), d2_5));
                float2 _f32x2_mul_a_182 = mul_f32x2_noftz(mul_f32x2_noftz(m1_5, lnu_5), mul_f32x2_noftz(dw_7, inner_5));
                float2 dg_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&post_5));
                float2 _f32x2_mul_a_183 = mul_f32x2_noftz(mul_f32x2_noftz(m1_5, bng_5), mul_f32x2_noftz(mul_f32x2_noftz(w_5, make_float2(4.0f, 4.0f)), d12_5));
                float2 du_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_183), "l"(*(const unsigned long long*)&post_5));
                gw[3] = __as_u32(__float22bfloat162_rn(dg_5));
                uw[3] = __as_u32(__float22bfloat162_rn(du_5));
            }
            if (s0 < limit) {
                long long col_1 = (long long)(s0 * 256 + lane_1 * 8);
                reinterpret_cast<int4*>(d_gate + (dgoff + col_1))[0] = reinterpret_cast<int4*>(gw)[0];
                reinterpret_cast<int4*>(d_up + (duoff + col_1))[0] = reinterpret_cast<int4*>(uw)[0];
            }
        }
    }
}

} // extern "C"
