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

#define LAUNCH_MIN_BLOCKS 2

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

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_train_2a2c48dec0406daa9e81(__nv_bfloat16* __restrict__ x, void* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw_acc, unsigned int* __restrict__ work, void* __restrict__ dw, void* __restrict__ db, void* __restrict__ b, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles, int flags)
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
    float* red = reinterpret_cast<float*>(smem_raw + 0);
    const int red_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    long long C64 = (long long)C;
    int ct = bid % n_ch_tiles;
    int rank = bid / n_ch_tiles;
    int g_ct = (((num_bids - ct + n_ch_tiles - 1) / n_ch_tiles) > (1) ? ((num_bids - ct + n_ch_tiles - 1) / n_ch_tiles) : (1));
    int sub = tid / ct_vec;
    int v = tid - sub * ct_vec;
    int c0 = (ct * ct_vec + v) * 8;
    long long c64 = (long long)c0;
    int lane_on = 0;
    if (sub < n_sub && c0 < C) {
        lane_on = 1;
    }
    float wr[34];
    wr[32] = __uint_as_float((unsigned int)S >> 31 | 2147483648u);
    wr[33] = wr[32];
    float br[8];
    float accb[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        br[j] = 0.0f;
        accb[j] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        wr[i] = 0.0f;
    }
    if (lane_on != 0) {
        if ((flags & 1) != 0) {
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                {
                    uint2 _vld_0;
                    _vld_0 = *reinterpret_cast<const uint2*>(reinterpret_cast<__nv_bfloat16*>(w) + (c64 + (long long)j_1) * 4ll);
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&wr[j_1 * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                        (&wr[j_1 * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int j_2 = 0; j_2 < 8; j_2++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(w) + (c64 + (long long)j_2) * 4ll);
                    wr[j_2 * 4 + 0] = _v4.x;
                    wr[j_2 * 4 + 1] = _v4.y;
                    wr[j_2 * 4 + 2] = _v4.z;
                    wr[j_2 * 4 + 3] = _v4.w;
                }
            }
        }
        if ((flags & 4) != 0) {
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(reinterpret_cast<__nv_bfloat16*>(b) + c64);
                uint4 _vld_2[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&br[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&br[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int q = 0; q < 2; q++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(b) + c64 + (long long)(q * 4));
                    br[q * 4 + 0] = _v4.x;
                    br[q * 4 + 1] = _v4.y;
                    br[q * 4 + 2] = _v4.z;
                    br[q * 4 + 3] = _v4.w;
                }
            }
        }
    }
    float acc[32];
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; i_1++) {
        acc[i_1] = 0.0f;
    }
    int block = n_sub * 4;
    int n_blocks = (S + block - 1) / block;
    int n_chunks = (num_docs + 32) / 32;
    unsigned int xr[40] = {0};
    unsigned int dr[28] = {0};
    float win[32];
    float pend[32];
    float dyf[8];
    float dp[8];
    #pragma unroll
    for (int i_2 = 0; i_2 < 32; i_2++) {
        win[i_2] = 0.0f;
        pend[i_2] = 0.0f;
    }
    #pragma unroll 1
    for (int tb = rank; tb < n_blocks; tb += g_ct) {
        int tb0 = tb * block;
        int t0 = tb0 + sub * 4;
        #pragma unroll
        for (int r = 0; r < 10; r++) {
            int row_r = t0 - 3 + r;
            int read = 0;
            if (lane_on != 0 && row_r >= 0 && row_r < S) {
                read = 1;
            }
            int _max_5 = ((row_r) > (0) ? (row_r) : (0));
            int _min_4 = ((_max_5) < (S - 1) ? (_max_5) : (S - 1));
            long long at = (long long)_min_4 * C64 + c64;
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.s32 p, %5, 0;\n\t@p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n\t}" : "+r"(xr[r * 4 + 0]), "+r"(xr[r * 4 + 1]), "+r"(xr[r * 4 + 2]), "+r"(xr[r * 4 + 3]) : "l"(reinterpret_cast<const unsigned int*>(x + at) + 0), "r"((int)(read != 0)) : "memory");
        }
        #pragma unroll
        for (int r_1 = 0; r_1 < 7; r_1++) {
            int row_d = t0 + r_1;
            int readd = 0;
            if (lane_on != 0 && row_d < S) {
                readd = 1;
            }
            int _min_6 = ((row_d) < (S - 1) ? (row_d) : (S - 1));
            long long atd = (long long)_min_6 * C64 + c64;
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.s32 p, %5, 0;\n\t@p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n\t}" : "+r"(dr[r_1 * 4 + 0]), "+r"(dr[r_1 * 4 + 1]), "+r"(dr[r_1 * 4 + 2]), "+r"(dr[r_1 * 4 + 3]) : "l"(reinterpret_cast<const unsigned int*>(dy + atd) + 0), "r"((int)(readd != 0)) : "memory");
        }
        unsigned int starts = 0;
        int lo_d = 0;
        if (num_docs > 31) {
            int tq = ((tb0 - 3) > (0) ? (tb0 - 3) : (0));
            int hi_d = num_docs - 1;
            #pragma unroll 1
            for (int step_d = 0; step_d < search_steps; step_d++) {
                int mid_d = lo_d + hi_d + 1 >> 1;
                if (tq >= cu[mid_d]) {
                    lo_d = mid_d;
                } else {
                    hi_d = mid_d - 1;
                }
            }
        }
        #pragma unroll 1
        for (int chunk = 0; chunk < n_chunks; chunk++) {
            int entry = S + block + 10 + 8;
            if (lo_d + lane <= num_docs) {
                entry = cu[lo_d + lane];
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 32; i_3++) {
                int _shfl_sync_7 = __shfl_sync(0xFFFFFFFF, entry, i_3);
                int pos = _shfl_sync_7 - t0 + 3;
                if (pos >= 1 && pos <= 9) {
                    starts = starts | 1 << (unsigned int)pos;
                }
            }
            int last_d = __shfl_sync(0xFFFFFFFF, entry, 31);
            if (last_d > tb0 + block + 3 || lo_d + 32 > num_docs) {
                break;
            }
            lo_d = lo_d + 32;
        }
        int dstart = t0 - 3;
        #pragma unroll
        for (int r_2 = 0; r_2 < 10; r_2++) {
            int tcur = t0 - 3 + r_2;
            if ((starts & 1 << (unsigned int)r_2) != 0) {
                #pragma unroll
                for (int m = 1; m < 4; m++) {
                    int tm = tcur - m;
                    if (r_2 - m >= 3 && r_2 - m - 3 < 4 && tm >= dstart && tm < S && lane_on != 0) {
                        unsigned int _pk_c[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + ((r_2 + 1) % 4 + 4 - m) % 4 * 8)[_lp*2 + 0], (pend + ((r_2 + 1) % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 0]));
                            _pk_c[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + ((r_2 + 1) % 4 + 4 - m) % 4 * 8)[_lp*2 + 2], (pend + ((r_2 + 1) % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 2]));
                            _pk_c[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + ((r_2 + 1) % 4 + 4 - m) % 4 * 8)[_lp*2 + 4], (pend + ((r_2 + 1) % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 4]));
                            _pk_c[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + ((r_2 + 1) % 4 + 4 - m) % 4 * 8)[_lp*2 + 6], (pend + ((r_2 + 1) % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 6]));
                            _pk_c[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + ((long long)tm * C64 + c64))[0] = reinterpret_cast<int4*>(_pk_c)[0];
                    }
                }
                dstart = tcur;
                #pragma unroll
                for (int q_1 = 1; q_1 < 4; q_1++) {
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        win[(r_2 + 1 + q_1) % 4 * 8 + e] = 0.0f;
                    }
                }
            }
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                win[(r_2 + 1) % 4 * 8 + 2 * kk] = __uint_as_float(xr[r_2 * 4 + kk] << 16);
                win[(r_2 + 1) % 4 * 8 + 2 * kk + 1] = __uint_as_float(xr[r_2 * 4 + kk] & 4294901760u);
            }
            if (r_2 >= 3) {
                #pragma unroll
                for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                    dyf[2 * kk_1] = __uint_as_float(dr[(r_2 - 3) * 4 + kk_1] << 16);
                    dyf[2 * kk_1 + 1] = __uint_as_float(dr[(r_2 - 3) * 4 + kk_1] & 4294901760u);
                }
                if (tcur < S) {
                    if (r_2 - 3 < 4) {
                        #pragma unroll
                        for (int pv_h = 0; pv_h < 4; pv_h++) {
                            float2 pv_pre2 = add_f32x2_noftz(add_f32x2_noftz(add_f32x2_noftz(add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h], wr[8 * pv_h + 4]), make_float2(win[((r_2 + 1) % 4 + 1) % 4 * 8 + 2 * pv_h], win[((r_2 + 1) % 4 + 1) % 4 * 8 + 2 * pv_h + 1]), make_float2(wr[32], wr[33])), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 1], wr[8 * pv_h + 4 + 1]), make_float2(win[((r_2 + 1) % 4 + 2) % 4 * 8 + 2 * pv_h], win[((r_2 + 1) % 4 + 2) % 4 * 8 + 2 * pv_h + 1]), make_float2(wr[32], wr[33]))), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 2], wr[8 * pv_h + 4 + 2]), make_float2(win[((r_2 + 1) % 4 + 3) % 4 * 8 + 2 * pv_h], win[((r_2 + 1) % 4 + 3) % 4 * 8 + 2 * pv_h + 1]), make_float2(wr[32], wr[33]))), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 3], wr[8 * pv_h + 4 + 3]), make_float2(win[(r_2 + 1) % 4 * 8 + 2 * pv_h], win[(r_2 + 1) % 4 * 8 + 2 * pv_h + 1]), make_float2(wr[32], wr[33]))), make_float2(br[2 * pv_h], br[2 * pv_h + 1]));
                            float2 pv_p2 = pv_pre2;
                            float2 _f32x2_mul_b_8 = make_float2(-1.442695f, -1.442695f);
                            float2 pv_e2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2) : "l"(*(const unsigned long long*)&pv_p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                            float _exp2_9 = approx_exp2(pv_e2.x);
                            float _exp2_10 = approx_exp2(pv_e2.y);
                            float2 pv_den2 = add_f32x2_noftz(make_float2(_exp2_9, _exp2_10), make_float2(1.0f, 1.0f));
                            float _rcp_11 = approx_rcp(pv_den2.x);
                            float _rcp_12 = approx_rcp(pv_den2.y);
                            float2 pv_sg2 = make_float2(_rcp_11, _rcp_12);
                            float2 _f32x2_mul_b_13 = fma_f32x2_rn_noftz(pv_p2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2), make_float2(1.0f, 1.0f));
                            float2 pv_g2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2) : "l"(*(const unsigned long long*)&pv_sg2), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
                            float2 pv_d2 = fma_f32x2_rn_noftz(make_float2(dyf[2 * pv_h], dyf[2 * pv_h + 1]), pv_g2, make_float2(wr[32], wr[33]));
                            dp[2 * pv_h] = pv_d2.x;
                            dp[2 * pv_h + 1] = pv_d2.y;
                            #pragma unroll
                            for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                                float2 pv_a2 = fma_f32x2_rn_noftz(pv_d2, make_float2(win[((r_2 + 1) % 4 + 1 + kk_2) % 4 * 8 + 2 * pv_h], win[((r_2 + 1) % 4 + 1 + kk_2) % 4 * 8 + 2 * pv_h + 1]), make_float2(acc[8 * pv_h + kk_2], acc[8 * pv_h + 4 + kk_2]));
                                acc[8 * pv_h + kk_2] = pv_a2.x;
                                acc[8 * pv_h + 4 + kk_2] = pv_a2.y;
                            }
                            float2 pv_b2 = add_f32x2_noftz(make_float2(accb[2 * pv_h], accb[2 * pv_h + 1]), pv_d2);
                            accb[2 * pv_h] = pv_b2.x;
                            accb[2 * pv_h + 1] = pv_b2.y;
                            #pragma unroll
                            for (int pv_k = 0; pv_k < 3; pv_k++) {
                                float2 pv_q2 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + pv_k], wr[8 * pv_h + 4 + pv_k]), pv_d2, make_float2(pend[((r_2 + 1) % 4 + 1 + pv_k) % 4 * 8 + 2 * pv_h], pend[((r_2 + 1) % 4 + 1 + pv_k) % 4 * 8 + 2 * pv_h + 1]));
                                pend[((r_2 + 1) % 4 + 1 + pv_k) % 4 * 8 + 2 * pv_h] = pv_q2.x;
                                pend[((r_2 + 1) % 4 + 1 + pv_k) % 4 * 8 + 2 * pv_h + 1] = pv_q2.y;
                            }
                            float2 _f32x2_mul_a_14 = make_float2(wr[8 * pv_h + 3], wr[8 * pv_h + 4 + 3]);
                            float2 pv_r2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&pv_d2));
                            pend[(r_2 + 1) % 4 * 8 + 2 * pv_h] = pv_r2.x;
                            pend[(r_2 + 1) % 4 * 8 + 2 * pv_h + 1] = pv_r2.y;
                        }
                    } else {
                        #pragma unroll
                        for (int pv_h_1 = 0; pv_h_1 < 4; pv_h_1++) {
                            float2 pv_pre2_1 = add_f32x2_noftz(add_f32x2_noftz(add_f32x2_noftz(add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1], wr[8 * pv_h_1 + 4]), make_float2(win[((r_2 + 1) % 4 + 1) % 4 * 8 + 2 * pv_h_1], win[((r_2 + 1) % 4 + 1) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(wr[32], wr[33])), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 1], wr[8 * pv_h_1 + 4 + 1]), make_float2(win[((r_2 + 1) % 4 + 2) % 4 * 8 + 2 * pv_h_1], win[((r_2 + 1) % 4 + 2) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(wr[32], wr[33]))), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 2], wr[8 * pv_h_1 + 4 + 2]), make_float2(win[((r_2 + 1) % 4 + 3) % 4 * 8 + 2 * pv_h_1], win[((r_2 + 1) % 4 + 3) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(wr[32], wr[33]))), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 3], wr[8 * pv_h_1 + 4 + 3]), make_float2(win[(r_2 + 1) % 4 * 8 + 2 * pv_h_1], win[(r_2 + 1) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(wr[32], wr[33]))), make_float2(br[2 * pv_h_1], br[2 * pv_h_1 + 1]));
                            float2 pv_p2_1 = pv_pre2_1;
                            float2 _f32x2_mul_b_15 = make_float2(-1.442695f, -1.442695f);
                            float2 pv_e2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_1) : "l"(*(const unsigned long long*)&pv_p2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
                            float _exp2_16 = approx_exp2(pv_e2_1.x);
                            float _exp2_17 = approx_exp2(pv_e2_1.y);
                            float2 pv_den2_1 = add_f32x2_noftz(make_float2(_exp2_16, _exp2_17), make_float2(1.0f, 1.0f));
                            float _rcp_18 = approx_rcp(pv_den2_1.x);
                            float _rcp_19 = approx_rcp(pv_den2_1.y);
                            float2 pv_sg2_1 = make_float2(_rcp_18, _rcp_19);
                            float2 _f32x2_mul_b_20 = fma_f32x2_rn_noftz(pv_p2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_1), make_float2(1.0f, 1.0f));
                            float2 pv_g2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_1) : "l"(*(const unsigned long long*)&pv_sg2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
                            float2 pv_d2_1 = fma_f32x2_rn_noftz(make_float2(dyf[2 * pv_h_1], dyf[2 * pv_h_1 + 1]), pv_g2_1, make_float2(wr[32], wr[33]));
                            dp[2 * pv_h_1] = pv_d2_1.x;
                            dp[2 * pv_h_1 + 1] = pv_d2_1.y;
                            #pragma unroll
                            for (int pv_k_1 = 0; pv_k_1 < 3; pv_k_1++) {
                                float2 pv_q2_1 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + pv_k_1], wr[8 * pv_h_1 + 4 + pv_k_1]), pv_d2_1, make_float2(pend[((r_2 + 1) % 4 + 1 + pv_k_1) % 4 * 8 + 2 * pv_h_1], pend[((r_2 + 1) % 4 + 1 + pv_k_1) % 4 * 8 + 2 * pv_h_1 + 1]));
                                pend[((r_2 + 1) % 4 + 1 + pv_k_1) % 4 * 8 + 2 * pv_h_1] = pv_q2_1.x;
                                pend[((r_2 + 1) % 4 + 1 + pv_k_1) % 4 * 8 + 2 * pv_h_1 + 1] = pv_q2_1.y;
                            }
                            float2 _f32x2_mul_a_21 = make_float2(wr[8 * pv_h_1 + 3], wr[8 * pv_h_1 + 4 + 3]);
                            float2 pv_r2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&pv_d2_1));
                            pend[(r_2 + 1) % 4 * 8 + 2 * pv_h_1] = pv_r2_1.x;
                            pend[(r_2 + 1) % 4 * 8 + 2 * pv_h_1 + 1] = pv_r2_1.y;
                        }
                    }
                }
                if (r_2 >= 6) {
                    int t3 = tcur - 3;
                    if (t3 >= dstart && t3 < S && lane_on != 0) {
                        unsigned int _pk_n[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + ((r_2 + 1) % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend + ((r_2 + 1) % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                            _pk_n[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + ((r_2 + 1) % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend + ((r_2 + 1) % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                            _pk_n[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + ((r_2 + 1) % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend + ((r_2 + 1) % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                            _pk_n[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + ((r_2 + 1) % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend + ((r_2 + 1) % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                            _pk_n[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + ((long long)t3 * C64 + c64))[0] = reinterpret_cast<int4*>(_pk_n)[0];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; i_4++) {
        red[i_4 * 128 + (tid ^ i_4 & 28)] = acc[i_4];
    }
    #pragma unroll
    for (int j_3 = 0; j_3 < 8; j_3++) {
        red[(32 + j_3) * 128 + (tid ^ 4 * j_3)] = accb[j_3];
    }
    __syncthreads();
    int cb0 = ct * ct_vec * 8;
    int nt = ct_vec * 8;
    int n_items = nt + nt / 4;
    #pragma unroll 1
    for (int q_2 = tid; q_2 < n_items; q_2 += 128) {
        if (q_2 < nt) {
            if (cb0 + q_2 < C) {
                int vq = q_2 / 8;
                int jq = q_2 - vq * 8;
                float t4[4];
                float u4[4];
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    t4[k] = red[(jq * 4 + k) * 128 + (vq ^ jq * 4)];
                }
                #pragma unroll 1
                for (int s = 1; s < n_sub; s++) {
                    int rq = s * ct_vec + vq;
                    #pragma unroll
                    for (int k_1 = 0; k_1 < 4; k_1++) {
                        u4[k_1] = red[(jq * 4 + k_1) * 128 + (rq ^ jq * 4)];
                    }
                    #pragma unroll
                    for (int k_2 = 0; k_2 < 4; k_2++) {
                        t4[k_2] = t4[k_2] + u4[k_2];
                    }
                }
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dw_acc[(cb0 + q_2) * 4])), "f"(t4[0]), "f"(t4[1]), "f"(t4[2]), "f"(t4[3]) : "memory");
            }
        }
        if (q_2 >= nt) {
            int cq = (q_2 - nt) * 4;
            if (cb0 + cq < C) {
                int vb = cq / 8;
                int jb = cq - vb * 8;
                float b4[4];
                float v4[4];
                #pragma unroll
                for (int e_1 = 0; e_1 < 4; e_1++) {
                    b4[e_1] = red[(32 + jb + e_1) * 128 + (vb ^ (jb + e_1) * 4)];
                }
                #pragma unroll 1
                for (int s_1 = 1; s_1 < n_sub; s_1++) {
                    int rb = s_1 * ct_vec + vb;
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        v4[e_2] = red[(32 + jb + e_2) * 128 + (rb ^ (jb + e_2) * 4)];
                    }
                    #pragma unroll
                    for (int e_3 = 0; e_3 < 4; e_3++) {
                        b4[e_3] = b4[e_3] + v4[e_3];
                    }
                }
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dw_acc[C * 4 + cb0 + cq])), "f"(b4[0]), "f"(b4[1]), "f"(b4[2]), "f"(b4[3]) : "memory");
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        uint32_t done;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(done) : "l"(&work[ct]), "r"(static_cast<uint32_t>((unsigned int)(g_ct - 1))) : "memory");
        if ((int)done == g_ct - 1) {
            red[0] = 1.0f;
        } else {
            red[0] = 0.0f;
        }
    }
    __syncthreads();
    float last = red[0];
    if (last != 0.0f) {
        int ch0 = ct * ct_vec * 8;
        int lim = ((C) < (ch0 + ct_vec * 8) ? (C) : (ch0 + ct_vec * 8));
        float z4[4];
        #pragma unroll
        for (int e_4 = 0; e_4 < 4; e_4++) {
            z4[e_4] = 0.0f;
        }
        float a[16];
        float tb_1[4];
        #pragma unroll 1
        for (int c0_0 = ch0 + tid; c0_0 < ch0 + ct_vec * 8; c0_0 += 512) {
            #pragma unroll
            for (int j_4 = 0; j_4 < 4; j_4++) {
                int cl = c0_0 + 128 * j_4;
                if (cl < lim) {
                    long long cl64 = (long long)cl;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dw_acc + cl64 * 4ll);
                        a[4 * j_4 + 0] = _v4.x;
                        a[4 * j_4 + 1] = _v4.y;
                        a[4 * j_4 + 2] = _v4.z;
                        a[4 * j_4 + 3] = _v4.w;
                    }
                    {
                        tb_1[j_4] = *reinterpret_cast<const float*>(dw_acc + ((long long)C * 4ll + cl64));
                    }
                }
            }
            #pragma unroll
            for (int j_5 = 0; j_5 < 4; j_5++) {
                int c = c0_0 + 128 * j_5;
                if (c < lim) {
                    long long c64_0 = (long long)c;
                    long long bi = (long long)C * 4ll + c64_0;
                    {
                        float4 _v4 = make_float4(z4[0 + 0], z4[0 + 1], z4[0 + 2], z4[0 + 3]);
                        *reinterpret_cast<float4*>(dw_acc + c64_0 * 4ll) = _v4;
                    }
                    *(reinterpret_cast<float*>(dw_acc) + (bi)) = z4[0];
                    if ((flags & 2) != 0) {
                        #pragma unroll
                        for (int e_5 = 0; e_5 < 4; e_5++) {
                            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(dw)) + (c64_0 * 4ll + (long long)e_5)) = (__nv_bfloat16)a[4 * j_5 + e_5];
                        }
                    } else {
                        {
                            float4 _v4 = make_float4(a[4 * j_5 + 0], a[4 * j_5 + 1], a[4 * j_5 + 2], a[4 * j_5 + 3]);
                            *reinterpret_cast<float4*>(reinterpret_cast<float*>(dw) + c64_0 * 4ll) = _v4;
                        }
                    }
                    if ((flags & 8) != 0) {
                        *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(db)) + (c64_0)) = (__nv_bfloat16)tb_1[j_5];
                    } else {
                        *(reinterpret_cast<float*>(reinterpret_cast<float*>(db)) + (c64_0)) = tb_1[j_5];
                    }
                }
            }
        }
    }
}

} // extern "C"
