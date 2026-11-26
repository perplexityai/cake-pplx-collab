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

#define LAUNCH_MIN_BLOCKS 8

__device__ __forceinline__ uint32_t cvt_bf16x2_e2m1x2(uint32_t src) {
    uint16_t _fp4_u16 = (uint16_t)(src & 0xFFu);
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
        : "=r"(dst) : "h"(_fp4_u16));
#else
    uint32_t _fp4_f16x2;
    asm("{ .reg .b8 _fp4b, _fp4z0, _fp4z1, _fp4z2;      \n\t"
        "  mov.b32 {_fp4b, _fp4z0, _fp4z1, _fp4z2}, %1;   \n\t"
        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;              }"
        : "=r"(_fp4_f16x2) : "r"((uint32_t)_fp4_u16));
    dst = ((_fp4_f16x2 & 0x7FFF7FFFu) >> 3) | (_fp4_f16x2 & 0x80008000u);
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst) : "r"(dst), "r"(0x77807780u));
#endif
    return dst;
}

__device__ __forceinline__ uint32_t cvt_bf16x2_e2m1x2_scaled(uint32_t src, uint16_t scale) {
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2.scaled::n2::ue8m0 %0, _fp4b, %2; }"
        : "=r"(dst) : "h"((uint16_t)(src & 0xFFu)), "h"(scale));
#else
    uint32_t _scale;
    asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_scale) : "h"(scale));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst) : "r"(cvt_bf16x2_e2m1x2(src)), "r"(_scale));
#endif
    return dst;
}

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_act_qdq_3fcef995964af95f49e7(const __nv_bfloat16* x, __nv_bfloat16* y, unsigned int* __restrict__ counters0, unsigned int* __restrict__ counters1, int n_vecs, int first_tile, int first_die, int chunk_tile_shift, int batch_shift)
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
    int* sword = reinterpret_cast<int*>(smem_raw + 0);
    const int sword_addr = smem + 0;

    // === Task calls (declaration order) ===
    int tiles = n_vecs + 1023 >> 10;
    int last = n_vecs - 1;
    int last_chunk = first_tile + tiles - 1 >> chunk_tile_shift;
    int items_shift = chunk_tile_shift - batch_shift;
    int items0 = last_chunk - first_die + 2 >> 1 << items_shift;
    int items1 = last_chunk - (first_die ^ 1) + 2 >> 1 << items_shift;
    int trips = items0 + items1 + 1;
    int visit = 0;
    #pragma unroll 1
    for (int _trip = 0; _trip < trips; _trip++) {
        if (tid == 0) {
            uint32_t _read_special_reg_0;
            asm("mov.u32 %0, %%locality_domain_id;" : "=r"(_read_special_reg_0));
            int home = (int)_read_special_reg_0;
            int mine = -1;
            if (visit == 0) {
                unsigned int taken = 0;
                if (home == 0) {
                    uint32_t _atomic_fetch_inc_1;
                    asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_inc_1) : "l"(counters0), "r"(static_cast<uint32_t>((unsigned int)(items0 + num_bids - 1))) : "memory");
                    taken = _atomic_fetch_inc_1;
                } else {
                    uint32_t _atomic_fetch_inc_2;
                    asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_inc_2) : "l"(counters1), "r"(static_cast<uint32_t>((unsigned int)(items1 + num_bids - 1))) : "memory");
                    taken = _atomic_fetch_inc_2;
                }
                mine = (int)taken;
            }
            int v = visit;
            int found = -1;
            if (v == 0) {
                if (mine < ((home == 1) ? items1 : items0)) {
                    found = mine * 2 + home;
                } else {
                    v = 1;
                }
            }
            if (v == 1) {
                int other = home ^ 1;
                unsigned int taken_1 = 0;
                if (other == 0) {
                    uint32_t _atomic_fetch_inc_3;
                    asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_inc_3) : "l"(counters0), "r"(static_cast<uint32_t>((unsigned int)(items0 + num_bids - 1))) : "memory");
                    taken_1 = _atomic_fetch_inc_3;
                } else {
                    uint32_t _atomic_fetch_inc_4;
                    asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_inc_4) : "l"(counters1), "r"(static_cast<uint32_t>((unsigned int)(items1 + num_bids - 1))) : "memory");
                    taken_1 = _atomic_fetch_inc_4;
                }
                int theirs = (int)taken_1;
                if (theirs < ((other == 1) ? items1 : items0)) {
                    found = theirs * 2 + other;
                } else {
                    v = 2;
                }
            }
            visit = v;
            sword[0] = found;
        }
        __syncthreads();
        int word;
        asm volatile("redux.sync.min.s32 %0, %1, 0xFFFFFFFF;" : "=r"(word) : "r"(sword[0]));
        __syncthreads();
        if (word < 0) {
            break;
        }
        int item = word >> 1;
        int chunk = (word & 1 ^ first_die) + (item >> items_shift << 1);
        int slot = (chunk << chunk_tile_shift) + ((item & (1 << items_shift) - 1) << batch_shift);
        int lo = ((slot < first_tile) ? 0 : slot - first_tile);
        int end = slot - first_tile + (1 << batch_shift);
        int hi = ((end < tiles) ? end : tiles);
        #pragma unroll 1
        for (int c = lo; c < hi; c++) {
            unsigned int w[32];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int i = c * 1024 + r * 128 + tid;
                int ci = ((i < n_vecs) ? i : last);
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + (long long)ci * 8ll + 0);
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&w[r * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            #pragma unroll
            for (int r_1 = 0; r_1 < 8; r_1++) {
                int i_1 = c * 1024 + r_1 * 128 + tid;
                unsigned int pair_abs[4];
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    uint32_t abs_word;
                    asm("abs.bf16x2 %0, %1;" : "=r"(abs_word) : "r"(w[r_1 * 4 + p]));
                    pair_abs[p] = abs_word;
                }
                uint32_t max01;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(max01) : "r"(pair_abs[0]), "r"(pair_abs[1]));
                uint32_t max23;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(max23) : "r"(pair_abs[2]), "r"(pair_abs[3]));
                uint32_t max_word;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(max_word) : "r"(max01), "r"(max23));
                float amax = fmaxf(__uint_as_float(max_word << 16), __uint_as_float(max_word & 4294901760u));
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
                amax = fmaxf(amax, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, amax, 2);
                amax = fmaxf(amax, _shfl_xor_sync_7);
                int amax_i = 0;
                amax_i = reinterpret_cast<int*>(&amax)[0];
                int biased = (amax_i >> 23) - 2 + (((amax_i & 8388607) > 4194304) ? 1 : 0);
                int scale_byte = ((biased > 1) ? biased : 1);
                float scale = 0.0f;
                int _bit_cast_src_8 = scale_byte << 23;
                scale = reinterpret_cast<float*>(&_bit_cast_src_8)[0];
                int inv_bits = 254 - scale_byte << 23;
                unsigned int inv_pair = (unsigned int)(inv_bits | inv_bits >> 16);
                uint16_t scale_pair = (uint16_t)(scale_byte << 8 | scale_byte);
                unsigned int out[4];
                #pragma unroll
                for (int p_1 = 0; p_1 < 4; p_1++) {
                    uint32_t quot;
                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quot) : "r"(w[r_1 * 4 + p_1]), "r"(inv_pair));
                    uint16_t codes = 0;
                    uint16_t _cvt_satfinite_e2m1x2_bf16x2_9;
                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _q; cvt.rn.satfinite.e2m1x2.bf16x2 _q, %1; mov.b16 %0, {_q, 0}; }" : "=h"(_cvt_satfinite_e2m1x2_bf16x2_9) : "r"(quot));
                    #else
                    {
                    const uint32_t _bf16x2 = (uint32_t)(quot);
                    const float _lo = __uint_as_float(_bf16x2 << 16);
                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                    asm("{ .reg .b8 _q; cvt.rn.satfinite.e2m1x2.f32 _q, %1, %2; mov.b16 %0, {_q, 0}; }" : "=h"(_cvt_satfinite_e2m1x2_bf16x2_9) : "f"(_hi), "f"(_lo));
                    }
                    #endif
                    codes = _cvt_satfinite_e2m1x2_bf16x2_9;
                    unsigned int y_pair = 0;
                    uint32_t _cvt_bf16x2_e2m1x2_10 = cvt_bf16x2_e2m1x2_scaled(codes, scale_pair);
                    y_pair = _cvt_bf16x2_e2m1x2_10;
                    out[p_1] = y_pair;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(y + (long long)i_1 * 8ll), "r"(out[(0) + 0]), "r"(out[(0) + 1]), "r"(out[(0) + 2]), "r"(out[(0) + 3]), "r"((unsigned int)(i_1 < n_vecs)) : "memory");
            }
        }
    }
}

} // extern "C"
