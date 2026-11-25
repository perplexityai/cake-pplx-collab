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

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
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
kernel_cake_route_weight_scale_det_train_fdcbf1098d5f29c13f73(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ dy, float* __restrict__ w, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, int rows, long long x_stride, long long dy_stride, long long dx_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[1024];

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    unsigned int* _rec = reinterpret_cast<unsigned int*>(smem_static_raw + 0);
    const int rec_addr = (int)(unsigned long long)__cvta_generic_to_shared(_rec);

    // === Task calls (declaration order) ===
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    #pragma unroll 1
    for (int row = bid; row < rows; row += num_bids) {
        long long xoff = (long long)row * x_stride;
        long long doff = (long long)row * dy_stride;
        long long ooff = (long long)row * dx_stride;
        float wr = w[row];
        float2 wr2 = make_float2(wr, wr);
        if (warp_1 < 4) {
            unsigned int xw[16];
            unsigned int yw[16];
            long long col_0 = (long long)((warp_1 * 4 * 32 + lane_1) * 8);
            {
                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(x + (xoff + col_0) + 0);
                xw[0 + 0] = _uv4_0.x;
                xw[0 + 1] = _uv4_0.y;
                xw[0 + 2] = _uv4_0.z;
                xw[0 + 3] = _uv4_0.w;
            }
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(dy + (doff + col_0) + 0);
                yw[0 + 0] = _uv4_1.x;
                yw[0 + 1] = _uv4_1.y;
                yw[0 + 2] = _uv4_1.z;
                yw[0 + 3] = _uv4_1.w;
            }
            long long col_1 = (long long)(((warp_1 * 4 + 1) * 32 + lane_1) * 8);
            {
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(x + (xoff + col_1) + 0);
                xw[4 + 0] = _uv4_2.x;
                xw[4 + 1] = _uv4_2.y;
                xw[4 + 2] = _uv4_2.z;
                xw[4 + 3] = _uv4_2.w;
            }
            {
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(dy + (doff + col_1) + 0);
                yw[4 + 0] = _uv4_3.x;
                yw[4 + 1] = _uv4_3.y;
                yw[4 + 2] = _uv4_3.z;
                yw[4 + 3] = _uv4_3.w;
            }
            long long col_2 = (long long)(((warp_1 * 4 + 2) * 32 + lane_1) * 8);
            {
                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(x + (xoff + col_2) + 0);
                xw[8 + 0] = _uv4_4.x;
                xw[8 + 1] = _uv4_4.y;
                xw[8 + 2] = _uv4_4.z;
                xw[8 + 3] = _uv4_4.w;
            }
            {
                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(dy + (doff + col_2) + 0);
                yw[8 + 0] = _uv4_5.x;
                yw[8 + 1] = _uv4_5.y;
                yw[8 + 2] = _uv4_5.z;
                yw[8 + 3] = _uv4_5.w;
            }
            long long col_3 = (long long)(((warp_1 * 4 + 3) * 32 + lane_1) * 8);
            {
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(x + (xoff + col_3) + 0);
                xw[12 + 0] = _uv4_6.x;
                xw[12 + 1] = _uv4_6.y;
                xw[12 + 2] = _uv4_6.z;
                xw[12 + 3] = _uv4_6.w;
            }
            {
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(dy + (doff + col_3) + 0);
                yw[12 + 0] = _uv4_7.x;
                yw[12 + 1] = _uv4_7.y;
                yw[12 + 2] = _uv4_7.z;
                yw[12 + 3] = _uv4_7.w;
            }
            unsigned int gw[16];
            float2 d_0 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[0]));
            float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[0]));
            gw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_0)));
            float2 d_1 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[1]));
            float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[1]));
            gw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_1)));
            float2 d_2 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[2]));
            float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[2]));
            gw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_2)));
            float2 d_3 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[3]));
            float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[3]));
            gw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_3)));
            float2 lo_0 = fma_f32x2_rn_noftz(d_1, xv_1, mul_f32x2_noftz(d_0, xv_0));
            float2 hi_0 = fma_f32x2_rn_noftz(d_3, xv_3, mul_f32x2_noftz(d_2, xv_2));
            float2 piece_0 = add_f32x2_noftz(lo_0, hi_0);
            float2 d_4 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[4]));
            float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4]));
            gw[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_4)));
            float2 d_5 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[5]));
            float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[5]));
            gw[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_5)));
            float2 d_6 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[6]));
            float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[6]));
            gw[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_6)));
            float2 d_7 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[7]));
            float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[7]));
            gw[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_7)));
            float2 lo_1 = fma_f32x2_rn_noftz(d_5, xv_5, mul_f32x2_noftz(d_4, xv_4));
            float2 hi_1 = fma_f32x2_rn_noftz(d_7, xv_7, mul_f32x2_noftz(d_6, xv_6));
            float2 piece_1 = add_f32x2_noftz(lo_1, hi_1);
            float2 joined_0 = add_f32x2_noftz(piece_0, piece_1);
            float2 d_8 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[8]));
            float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[8]));
            gw[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_8)));
            float2 d_9 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[9]));
            float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[9]));
            gw[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_9)));
            float2 d_10 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[10]));
            float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[10]));
            gw[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_10)));
            float2 d_11 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[11]));
            float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[11]));
            gw[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_11)));
            float2 lo_2 = fma_f32x2_rn_noftz(d_9, xv_9, mul_f32x2_noftz(d_8, xv_8));
            float2 hi_2 = fma_f32x2_rn_noftz(d_11, xv_11, mul_f32x2_noftz(d_10, xv_10));
            float2 piece_2 = add_f32x2_noftz(lo_2, hi_2);
            float2 d_12 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[12]));
            float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[12]));
            gw[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_12)));
            float2 d_13 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[13]));
            float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[13]));
            gw[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_13)));
            float2 d_14 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[14]));
            float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[14]));
            gw[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_14)));
            float2 d_15 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[15]));
            float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[15]));
            gw[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_15)));
            float2 lo_3 = fma_f32x2_rn_noftz(d_13, xv_13, mul_f32x2_noftz(d_12, xv_12));
            float2 hi_3 = fma_f32x2_rn_noftz(d_15, xv_15, mul_f32x2_noftz(d_14, xv_14));
            float2 piece_3 = add_f32x2_noftz(lo_3, hi_3);
            float2 joined_1 = add_f32x2_noftz(piece_2, piece_3);
            float2 joined_2 = add_f32x2_noftz(joined_0, joined_1);
            long long col_4 = (long long)((warp_1 * 4 * 32 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_4))[0] = reinterpret_cast<int4*>(gw)[0];
            long long col_5 = (long long)(((warp_1 * 4 + 1) * 32 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_5))[0] = reinterpret_cast<int4*>(gw + (4))[0];
            long long col_6 = (long long)(((warp_1 * 4 + 2) * 32 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_6))[0] = reinterpret_cast<int4*>(gw + (8))[0];
            long long col_7 = (long long)(((warp_1 * 4 + 3) * 32 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_7))[0] = reinterpret_cast<int4*>(gw + (12))[0];
            int slot = warp_1 * 32 + lane_1;
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(rec_addr + (unsigned int)(8 * slot)), "r"(__as_u32(joined_2.x)), "r"(__as_u32(joined_2.y)) : "memory");
        }
        __syncthreads();
        if (warp_1 == 0) {
            unsigned int r2_0[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r2_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&r2_0[(0) + 1]))
                : "r"(rec_addr + (unsigned int)(8 * lane_1)));
            float2 part_0 = make_float2(__uint_as_float(r2_0[0]), __uint_as_float(r2_0[1]));
            unsigned int r2_1[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&r2_1[(0) + 1]))
                : "r"(rec_addr + (unsigned int)(8 * (32 + lane_1))));
            float2 part_1 = make_float2(__uint_as_float(r2_1[0]), __uint_as_float(r2_1[1]));
            float2 above_0 = add_f32x2_noftz(part_0, part_1);
            unsigned int r2_2[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r2_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&r2_2[(0) + 1]))
                : "r"(rec_addr + (unsigned int)(8 * (64 + lane_1))));
            float2 part_2 = make_float2(__uint_as_float(r2_2[0]), __uint_as_float(r2_2[1]));
            unsigned int r2_3[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&r2_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&r2_3[(0) + 1]))
                : "r"(rec_addr + (unsigned int)(8 * (96 + lane_1))));
            float2 part_3 = make_float2(__uint_as_float(r2_3[0]), __uint_as_float(r2_3[1]));
            float2 above_1 = add_f32x2_noftz(part_2, part_3);
            float2 above_2 = add_f32x2_noftz(above_0, above_1);
            float total;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(total) : "f"(above_2.x), "f"(above_2.y));
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 16);
            float _add_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(total), "f"(_shfl_xor_sync_9));
            total = _add_8;
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
            float _add_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(total), "f"(_shfl_xor_sync_11));
            total = _add_10;
            float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(total), "f"(_shfl_xor_sync_13));
            total = _add_12;
            float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            float _add_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(total), "f"(_shfl_xor_sync_15));
            total = _add_14;
            float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
            float _add_16;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(total), "f"(_shfl_xor_sync_17));
            total = _add_16;
            if (lane_1 == 0) {
                *(reinterpret_cast<float*>(dw + row) + (0)) = total;
            }
        }
        __syncthreads();
    }
}

} // extern "C"
