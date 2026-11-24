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

#define LAUNCH_MIN_BLOCKS 16

#include <math_constants.h>

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(64, LAUNCH_MIN_BLOCKS) void
kernel_cake_act_qdq_48c0dbfaea5f3b9c07f7(const __nv_bfloat16* x, __nv_bfloat16* y, float* __restrict__ scales, unsigned int* __restrict__ counters0, unsigned int* __restrict__ counters1, int n_vecs, int first_tile, int first_die, int chunk_tile_shift, int batch_shift)
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
    int tiles = n_vecs + 255 >> 8;
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
            unsigned int w[16];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                int i = c * 256 + r * 64 + tid;
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
            for (int r_1 = 0; r_1 < 4; r_1++) {
                int i_1 = c * 256 + r_1 * 64 + tid;
                float v_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&v_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((w + r_1 * 4)[_pair]) << 16);
                    (&v_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((w + r_1 * 4)[_pair]) & 0xffff0000u);
                }
                float am = fmaxf(v_1[0], -v_1[0]);
                #pragma unroll
                for (int k = 1; k < 8; k++) {
                    float _max3_6;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_6) : "f"(am), "f"(v_1[k]), "f"(-v_1[k]));
                    am = _max3_6;
                }
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, am, 1);
                am = fmaxf(am, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, am, 2);
                am = fmaxf(am, _shfl_xor_sync_8);
                float s = 0.0f;
                float inv = 0.0f;
                int sbits = 0;
                int abits = 0;
                abits = reinterpret_cast<int*>(&am)[0];
                int rounded = (abits >> 23) - 8 + (((abits & 8388607) > 6291456) ? 1 : 0);
                int sbyte = ((rounded > 105) ? rounded : 105);
                sbits = sbyte << 23;
                int ibits = 254 - sbyte << 23;
                s = reinterpret_cast<float*>(&sbits)[0];
                inv = reinterpret_cast<float*>(&ibits)[0];
                unsigned int s_word = (unsigned int)(sbits | sbits >> 16);
                float2 inv2 = make_float2(inv, inv);
                float2 s2 = make_float2(s, s);
                float2 neg_inv2 = make_float2(-inv, -inv);
                float2 u[4];
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    float2 x2 = make_float2(v_1[2 * p], v_1[2 * p + 1]);
                    u[p] = mul_f32x2_noftz(x2, inv2);
                }
                unsigned int halves[4];
                #pragma unroll
                for (int p_1 = 0; p_1 < 4; p_1++) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_10;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_10) : "f"(u[p_1].y), "f"(u[p_1].x));
                    uint32_t _cvt_f16x2_e4m3x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_9) : "h"((uint16_t)(_cvt_satfinite_e4m3x2_f32_10)));
                    halves[p_1] = _cvt_f16x2_e4m3x2_9;
                }
                float yq[8];
                #pragma unroll
                for (int p_2 = 0; p_2 < 4; p_2++) {
                    float _cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(halves[p_2])));
                    yq[2 * p_2] = mul_f32_rn_noftz(_cvt_f32_f16_11, s);
                    float _cvt_f32_f16_12;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(halves[p_2] >> 16)));
                    yq[2 * p_2 + 1] = mul_f32_rn_noftz(_cvt_f32_f16_12, s);
                }
                unsigned int packed[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yq[_lp*2 + 0], yq[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(y + (long long)i_1 * 8ll), "r"(packed[(0) + 0]), "r"(packed[(0) + 1]), "r"(packed[(0) + 2]), "r"(packed[(0) + 3]), "r"((unsigned int)(i_1 < n_vecs)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(scales + (long long)(i_1 >> 2))), "f"(s), "r"((unsigned int)(i_1 < n_vecs && tid % 4 == 0)) : "memory");
            }
        }
    }
}

} // extern "C"
