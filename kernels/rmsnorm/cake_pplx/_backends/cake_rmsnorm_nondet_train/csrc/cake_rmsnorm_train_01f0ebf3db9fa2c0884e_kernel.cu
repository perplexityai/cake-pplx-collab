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

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_rmsnorm_train_01f0ebf3db9fa2c0884e(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (declaration order) ===
    int warp_0 = warp;
    int group = tid / 128;
    int gtid = tid - group * 128;
    float h_f = 2048.0f;
    int it = 0;
    #pragma unroll 1
    for (int row0 = bid * 1; row0 < bid * 1 + 1; row0++) {
        int row = row0 + group;
        long long row64 = (long long)row;
        long long xbase = row64 * x_stride_p0;
        long long obase = row64 * 2048ll;
        float x_cache[16];
        float w_cache[16];
        float sum_sq = 0.0f;
        if (row < T) {
            #pragma unroll
            for (int v = 0; v < 2; v++) {
                float _vec_load_0[8];
                {
                    unsigned _ldv8_0_0;
                    unsigned _ldv8_0_1;
                    unsigned _ldv8_0_2;
                    unsigned _ldv8_0_3;
                    unsigned _ldv8_0_4;
                    unsigned _ldv8_0_5;
                    unsigned _ldv8_0_6;
                    unsigned _ldv8_0_7;
                    asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + (gtid * 8 + v * (128 * 8)) + (0))) : "memory");
                    _vec_load_0[0 + 0] = __uint_as_float(_ldv8_0_0);
                    _vec_load_0[0 + 1] = __uint_as_float(_ldv8_0_1);
                    _vec_load_0[0 + 2] = __uint_as_float(_ldv8_0_2);
                    _vec_load_0[0 + 3] = __uint_as_float(_ldv8_0_3);
                    _vec_load_0[0 + 4] = __uint_as_float(_ldv8_0_4);
                    _vec_load_0[0 + 5] = __uint_as_float(_ldv8_0_5);
                    _vec_load_0[0 + 6] = __uint_as_float(_ldv8_0_6);
                    _vec_load_0[0 + 7] = __uint_as_float(_ldv8_0_7);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float wv = _vec_load_0[j];
                    w_cache[v * 8 + j] = wv;
                }
            }
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                float _vec_load_1[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + (xbase + (long long)(gtid * 8 + v_1 * (128 * 8))) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    float value = _vec_load_1[j_1];
                    x_cache[v_1 * 8 + j_1] = value;
                }
            }
            #pragma unroll
            for (int v_2 = 0; v_2 < 2; v_2++) {
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    sum_sq = __fmaf_rn(x_cache[v_2 * 8 + j_2], x_cache[v_2 * 8 + j_2], sum_sq);
                }
            }
        }
        float _warp_reduce_2 = sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
        sum_sq = _warp_reduce_2;
        int par = (it & 1) * 4;
        if (lane == 0) {
            partials[par + warp_0] = sum_sq;
        }
        __syncthreads();
        float total = partials[par + group * 4];
        #pragma unroll
        for (int k = 1; k < 4; k++) {
            total += partials[par + group * 4 + k];
        }
        it += 1;
        float mean = total * 0.00048828125f;
        float rs_a = mean + eps;
        unsigned int rs_ab = __as_u32(rs_a);
        float rs_y;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs_y) : "f"(rs_a));
        float rs_t;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_t) : "f"(rs_a), "f"(rs_y));
        float _neg_3;
        asm("neg.f32 %0, %1;" : "=f"(_neg_3) : "f"(rs_t));
        float rs_tl;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_tl) : "f"(rs_a), "f"(rs_y), "f"(_neg_3));
        float _neg_4;
        asm("neg.f32 %0, %1;" : "=f"(_neg_4) : "f"(rs_t));
        float rs_e;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_e) : "f"(_neg_4), "f"(rs_y), "f"(1.0f));
        float _neg_6;
        asm("neg.f32 %0, %1;" : "=f"(_neg_6) : "f"(rs_tl));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(_neg_6), "f"(rs_y), "f"(rs_e));
        rs_e = _fma_5;
        float rs_h;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_h) : "f"(rs_y), "f"(0.5f));
        float _sub_7;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_lo;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_lo) : "f"(_sub_7), "f"(rs_h), "f"(rs_y));
        float _add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_hi;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_hi) : "f"(_add_8), "f"(rs_h), "f"(rs_y));
        float rs_r = rs_lo;
        unsigned int rs_slow = ((rs_ab - 8388608 >= 2130706432) ? 1 : 0);
        if (__as_u32(rs_lo) != __as_u32(rs_hi)) {
            rs_slow = 1;
        }
        if (rs_slow != 0) {
            float _sqrt_10;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_10) : "f"(rs_a));
            float _rcp_9 = approx_rcp(_sqrt_10);
            rs_y = _rcp_9;
            rs_t = mul_f32_rn_noftz(rs_a, rs_y);
            float _neg_12;
            asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(rs_t));
            float _fma_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(rs_a), "f"(rs_y), "f"(_neg_12));
            rs_tl = _fma_11;
            float _neg_14;
            asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(rs_t));
            float _fma_13;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(_neg_14), "f"(rs_y), "f"(1.0f));
            rs_e = _fma_13;
            float _neg_16;
            asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(rs_tl));
            float _fma_15;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(_neg_16), "f"(rs_y), "f"(rs_e));
            rs_e = _fma_15;
            float _fma_17;
            asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(rs_e), "f"(mul_f32_rn_noftz(rs_y, 0.5f)), "f"(rs_y));
            unsigned int rs_fb = __as_u32(_fma_17);
            unsigned long long rs_m = (unsigned long long)((rs_fb & 8388607) << 1 | 16777217);
            unsigned int rs_an = ((rs_ab < 8388608) ? __as_u32((float)rs_ab) : rs_ab);
            unsigned long long rs_p = rs_m * rs_m * (unsigned long long)(rs_an & 8388607 | 8388608);
            rs_r = ((rs_ab - 1 < 2139095039) ? __uint_as_float(rs_fb + (unsigned int)(rs_p >> 63ull)) : rs_y);
        }
        float rstd = rs_r;
        if (row < T) {
            if (gtid == 0) {
                r[row] = rstd;
            }
            #pragma unroll
            for (int v_3 = 0; v_3 < 2; v_3++) {
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    x_cache[v_3 * 8 + j_3] = x_cache[v_3 * 8 + j_3] * rstd * w_cache[v_3 * 8 + j_3];
                }
                unsigned int packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + v_3 * 8)[_lp*2 + 0], (x_cache + v_3 * 8)[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + v_3 * 8)[_lp*2 + 2], (x_cache + v_3 * 8)[_lp*2+1 + 2]));
                    packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + v_3 * 8)[_lp*2 + 4], (x_cache + v_3 * 8)[_lp*2+1 + 4]));
                    packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + v_3 * 8)[_lp*2 + 6], (x_cache + v_3 * 8)[_lp*2+1 + 6]));
                    packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(y + (obase + (long long)(gtid * 8 + v_3 * (128 * 8))))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
        }
    }
}

} // extern "C"
