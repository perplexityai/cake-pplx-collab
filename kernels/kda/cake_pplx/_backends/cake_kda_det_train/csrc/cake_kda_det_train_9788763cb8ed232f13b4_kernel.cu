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

__global__ __launch_bounds__(1024) void
kernel_cake_kda_det_train_9788763cb8ed232f13b4(float* __restrict__ dA_part, float* __restrict__ db_part, int* __restrict__ seq_chunk_start, int num_seqs, float* __restrict__ part2, int seg_cap, unsigned int* __restrict__ ticket, float* __restrict__ dA_log, float* __restrict__ ddt_bias, int num_heads)
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
    float* part = reinterpret_cast<float*>(smem_raw + 0);
    const int part_addr = smem + 0;
    float* counter = reinterpret_cast<float*>(smem_raw + 16384);
    const int counter_addr = smem + 16384;
    float* wsum = reinterpret_cast<float*>(smem_raw + 24576);
    const int wsum_addr = smem + 24576;
    unsigned int* flag = reinterpret_cast<unsigned int*>(smem_raw + 24592);
    const int flag_addr = smem + 24592;

    // === Task calls (declaration order) ===
    int head = blockIdx.x;
    int p = blockIdx.y;
    int g = gridDim.y;
    int j = tid >> 7;
    int d = tid & 127;
    int nc = seq_chunk_start[num_seqs];
    int nseg = (((nc + 127 >> 7)) < (seg_cap) ? ((nc + 127 >> 7)) : (seg_cap));
    unsigned int slot_addr = part_addr + (unsigned int)((j * 128 + d) * 4 * 4);
    float sums[4];
    float top[2];
    unsigned int last = 0;
    if (g == 1) {
        if (j == 0 && nseg > 1) {
            top[0] = 0.0f;
            top[1] = 0.0f;
            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(counter_addr + (unsigned int)(d * 8)), "f"(top[0]), "f"(top[1]) : "memory");
            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(counter_addr + (unsigned int)((128 + d) * 8)), "f"(top[0]), "f"(top[1]) : "memory");
            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(counter_addr + (unsigned int)((256 + d) * 8)), "f"(top[0]), "f"(top[1]) : "memory");
            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(counter_addr + (unsigned int)((384 + d) * 8)), "f"(top[0]), "f"(top[1]) : "memory");
            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(counter_addr + (unsigned int)((512 + d) * 8)), "f"(top[0]), "f"(top[1]) : "memory");
            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(counter_addr + (unsigned int)((640 + d) * 8)), "f"(top[0]), "f"(top[1]) : "memory");
            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(counter_addr + (unsigned int)((768 + d) * 8)), "f"(top[0]), "f"(top[1]) : "memory");
            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(counter_addr + (unsigned int)((896 + d) * 8)), "f"(top[0]), "f"(top[1]) : "memory");
        }
        for (int s1 = 0; s1 < nseg; s1++) {
            if (s1 * 128 + 128 <= nc) {
                float xa[32];
                float xb[32];
                long long row = (((long long)(s1 * 128 + j) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long step = (long long)num_heads * 2048ll;
                {
                    float2 _v2_0 = *reinterpret_cast<const float2*>(dA_part + row);
                    xa[0] = _v2_0.x;
                    xa[0 + 1] = _v2_0.y;
                }
                {
                    float2 _v2_1 = *reinterpret_cast<const float2*>(db_part + row);
                    xb[0] = _v2_1.x;
                    xb[0 + 1] = _v2_1.y;
                }
                {
                    float2 _v2_2 = *reinterpret_cast<const float2*>(dA_part + row + step);
                    xa[2] = _v2_2.x;
                    xa[2 + 1] = _v2_2.y;
                }
                {
                    float2 _v2_3 = *reinterpret_cast<const float2*>(db_part + row + step);
                    xb[2] = _v2_3.x;
                    xb[2 + 1] = _v2_3.y;
                }
                {
                    float2 _v2_4 = *reinterpret_cast<const float2*>(dA_part + row + step * 2ll);
                    xa[4] = _v2_4.x;
                    xa[4 + 1] = _v2_4.y;
                }
                {
                    float2 _v2_5 = *reinterpret_cast<const float2*>(db_part + row + step * 2ll);
                    xb[4] = _v2_5.x;
                    xb[4 + 1] = _v2_5.y;
                }
                {
                    float2 _v2_6 = *reinterpret_cast<const float2*>(dA_part + row + step * 3ll);
                    xa[6] = _v2_6.x;
                    xa[6 + 1] = _v2_6.y;
                }
                {
                    float2 _v2_7 = *reinterpret_cast<const float2*>(db_part + row + step * 3ll);
                    xb[6] = _v2_7.x;
                    xb[6 + 1] = _v2_7.y;
                }
                {
                    float2 _v2_8 = *reinterpret_cast<const float2*>(dA_part + row + step * 4ll);
                    xa[8] = _v2_8.x;
                    xa[8 + 1] = _v2_8.y;
                }
                {
                    float2 _v2_9 = *reinterpret_cast<const float2*>(db_part + row + step * 4ll);
                    xb[8] = _v2_9.x;
                    xb[8 + 1] = _v2_9.y;
                }
                {
                    float2 _v2_10 = *reinterpret_cast<const float2*>(dA_part + row + step * 5ll);
                    xa[10] = _v2_10.x;
                    xa[10 + 1] = _v2_10.y;
                }
                {
                    float2 _v2_11 = *reinterpret_cast<const float2*>(db_part + row + step * 5ll);
                    xb[10] = _v2_11.x;
                    xb[10 + 1] = _v2_11.y;
                }
                {
                    float2 _v2_12 = *reinterpret_cast<const float2*>(dA_part + row + step * 6ll);
                    xa[12] = _v2_12.x;
                    xa[12 + 1] = _v2_12.y;
                }
                {
                    float2 _v2_13 = *reinterpret_cast<const float2*>(db_part + row + step * 6ll);
                    xb[12] = _v2_13.x;
                    xb[12 + 1] = _v2_13.y;
                }
                {
                    float2 _v2_14 = *reinterpret_cast<const float2*>(dA_part + row + step * 7ll);
                    xa[14] = _v2_14.x;
                    xa[14 + 1] = _v2_14.y;
                }
                {
                    float2 _v2_15 = *reinterpret_cast<const float2*>(db_part + row + step * 7ll);
                    xb[14] = _v2_15.x;
                    xb[14 + 1] = _v2_15.y;
                }
                {
                    float2 _v2_16 = *reinterpret_cast<const float2*>(dA_part + row + step * 8ll);
                    xa[16] = _v2_16.x;
                    xa[16 + 1] = _v2_16.y;
                }
                {
                    float2 _v2_17 = *reinterpret_cast<const float2*>(db_part + row + step * 8ll);
                    xb[16] = _v2_17.x;
                    xb[16 + 1] = _v2_17.y;
                }
                {
                    float2 _v2_18 = *reinterpret_cast<const float2*>(dA_part + row + step * 9ll);
                    xa[18] = _v2_18.x;
                    xa[18 + 1] = _v2_18.y;
                }
                {
                    float2 _v2_19 = *reinterpret_cast<const float2*>(db_part + row + step * 9ll);
                    xb[18] = _v2_19.x;
                    xb[18 + 1] = _v2_19.y;
                }
                {
                    float2 _v2_20 = *reinterpret_cast<const float2*>(dA_part + row + step * 10ll);
                    xa[20] = _v2_20.x;
                    xa[20 + 1] = _v2_20.y;
                }
                {
                    float2 _v2_21 = *reinterpret_cast<const float2*>(db_part + row + step * 10ll);
                    xb[20] = _v2_21.x;
                    xb[20 + 1] = _v2_21.y;
                }
                {
                    float2 _v2_22 = *reinterpret_cast<const float2*>(dA_part + row + step * 11ll);
                    xa[22] = _v2_22.x;
                    xa[22 + 1] = _v2_22.y;
                }
                {
                    float2 _v2_23 = *reinterpret_cast<const float2*>(db_part + row + step * 11ll);
                    xb[22] = _v2_23.x;
                    xb[22 + 1] = _v2_23.y;
                }
                {
                    float2 _v2_24 = *reinterpret_cast<const float2*>(dA_part + row + step * 12ll);
                    xa[24] = _v2_24.x;
                    xa[24 + 1] = _v2_24.y;
                }
                {
                    float2 _v2_25 = *reinterpret_cast<const float2*>(db_part + row + step * 12ll);
                    xb[24] = _v2_25.x;
                    xb[24 + 1] = _v2_25.y;
                }
                {
                    float2 _v2_26 = *reinterpret_cast<const float2*>(dA_part + row + step * 13ll);
                    xa[26] = _v2_26.x;
                    xa[26 + 1] = _v2_26.y;
                }
                {
                    float2 _v2_27 = *reinterpret_cast<const float2*>(db_part + row + step * 13ll);
                    xb[26] = _v2_27.x;
                    xb[26 + 1] = _v2_27.y;
                }
                {
                    float2 _v2_28 = *reinterpret_cast<const float2*>(dA_part + row + step * 14ll);
                    xa[28] = _v2_28.x;
                    xa[28 + 1] = _v2_28.y;
                }
                {
                    float2 _v2_29 = *reinterpret_cast<const float2*>(db_part + row + step * 14ll);
                    xb[28] = _v2_29.x;
                    xb[28 + 1] = _v2_29.y;
                }
                {
                    float2 _v2_30 = *reinterpret_cast<const float2*>(dA_part + row + step * 15ll);
                    xa[30] = _v2_30.x;
                    xa[30 + 1] = _v2_30.y;
                }
                {
                    float2 _v2_31 = *reinterpret_cast<const float2*>(db_part + row + step * 15ll);
                    xb[30] = _v2_31.x;
                    xb[30 + 1] = _v2_31.y;
                }
                float t_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_0) : "f"(xa[0]), "f"(xa[16]));
                xa[0] = t_0;
                float t_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_1) : "f"(xa[1]), "f"(xa[17]));
                xa[1] = t_1;
                float t_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_2) : "f"(xa[2]), "f"(xa[18]));
                xa[2] = t_2;
                float t_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_3) : "f"(xa[3]), "f"(xa[19]));
                xa[3] = t_3;
                float t_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_4) : "f"(xa[4]), "f"(xa[20]));
                xa[4] = t_4;
                float t_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_5) : "f"(xa[5]), "f"(xa[21]));
                xa[5] = t_5;
                float t_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_6) : "f"(xa[6]), "f"(xa[22]));
                xa[6] = t_6;
                float t_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_7) : "f"(xa[7]), "f"(xa[23]));
                xa[7] = t_7;
                float t_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_8) : "f"(xa[8]), "f"(xa[24]));
                xa[8] = t_8;
                float t_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_9) : "f"(xa[9]), "f"(xa[25]));
                xa[9] = t_9;
                float t_10;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_10) : "f"(xa[10]), "f"(xa[26]));
                xa[10] = t_10;
                float t_11;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_11) : "f"(xa[11]), "f"(xa[27]));
                xa[11] = t_11;
                float t_12;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_12) : "f"(xa[12]), "f"(xa[28]));
                xa[12] = t_12;
                float t_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_13) : "f"(xa[13]), "f"(xa[29]));
                xa[13] = t_13;
                float t_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_14) : "f"(xa[14]), "f"(xa[30]));
                xa[14] = t_14;
                float t_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_15) : "f"(xa[15]), "f"(xa[31]));
                xa[15] = t_15;
                float t_16;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_16) : "f"(xa[0]), "f"(xa[8]));
                xa[0] = t_16;
                float t_17;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_17) : "f"(xa[1]), "f"(xa[9]));
                xa[1] = t_17;
                float t_18;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_18) : "f"(xa[2]), "f"(xa[10]));
                xa[2] = t_18;
                float t_19;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_19) : "f"(xa[3]), "f"(xa[11]));
                xa[3] = t_19;
                float t_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_20) : "f"(xa[4]), "f"(xa[12]));
                xa[4] = t_20;
                float t_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_21) : "f"(xa[5]), "f"(xa[13]));
                xa[5] = t_21;
                float t_22;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_22) : "f"(xa[6]), "f"(xa[14]));
                xa[6] = t_22;
                float t_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_23) : "f"(xa[7]), "f"(xa[15]));
                xa[7] = t_23;
                float t_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_24) : "f"(xa[0]), "f"(xa[4]));
                xa[0] = t_24;
                float t_25;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_25) : "f"(xa[1]), "f"(xa[5]));
                xa[1] = t_25;
                float t_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_26) : "f"(xa[2]), "f"(xa[6]));
                xa[2] = t_26;
                float t_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_27) : "f"(xa[3]), "f"(xa[7]));
                xa[3] = t_27;
                float t_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_28) : "f"(xa[0]), "f"(xa[2]));
                xa[0] = t_28;
                float t_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_29) : "f"(xa[1]), "f"(xa[3]));
                xa[1] = t_29;
                float t_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_30) : "f"(xb[0]), "f"(xb[16]));
                xb[0] = t_30;
                float t_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_31) : "f"(xb[1]), "f"(xb[17]));
                xb[1] = t_31;
                float t_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_32) : "f"(xb[2]), "f"(xb[18]));
                xb[2] = t_32;
                float t_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_33) : "f"(xb[3]), "f"(xb[19]));
                xb[3] = t_33;
                float t_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_34) : "f"(xb[4]), "f"(xb[20]));
                xb[4] = t_34;
                float t_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_35) : "f"(xb[5]), "f"(xb[21]));
                xb[5] = t_35;
                float t_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_36) : "f"(xb[6]), "f"(xb[22]));
                xb[6] = t_36;
                float t_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_37) : "f"(xb[7]), "f"(xb[23]));
                xb[7] = t_37;
                float t_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_38) : "f"(xb[8]), "f"(xb[24]));
                xb[8] = t_38;
                float t_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_39) : "f"(xb[9]), "f"(xb[25]));
                xb[9] = t_39;
                float t_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_40) : "f"(xb[10]), "f"(xb[26]));
                xb[10] = t_40;
                float t_41;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_41) : "f"(xb[11]), "f"(xb[27]));
                xb[11] = t_41;
                float t_42;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_42) : "f"(xb[12]), "f"(xb[28]));
                xb[12] = t_42;
                float t_43;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_43) : "f"(xb[13]), "f"(xb[29]));
                xb[13] = t_43;
                float t_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_44) : "f"(xb[14]), "f"(xb[30]));
                xb[14] = t_44;
                float t_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_45) : "f"(xb[15]), "f"(xb[31]));
                xb[15] = t_45;
                float t_46;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_46) : "f"(xb[0]), "f"(xb[8]));
                xb[0] = t_46;
                float t_47;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_47) : "f"(xb[1]), "f"(xb[9]));
                xb[1] = t_47;
                float t_48;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_48) : "f"(xb[2]), "f"(xb[10]));
                xb[2] = t_48;
                float t_49;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_49) : "f"(xb[3]), "f"(xb[11]));
                xb[3] = t_49;
                float t_50;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_50) : "f"(xb[4]), "f"(xb[12]));
                xb[4] = t_50;
                float t_51;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_51) : "f"(xb[5]), "f"(xb[13]));
                xb[5] = t_51;
                float t_52;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_52) : "f"(xb[6]), "f"(xb[14]));
                xb[6] = t_52;
                float t_53;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_53) : "f"(xb[7]), "f"(xb[15]));
                xb[7] = t_53;
                float t_54;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_54) : "f"(xb[0]), "f"(xb[4]));
                xb[0] = t_54;
                float t_55;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_55) : "f"(xb[1]), "f"(xb[5]));
                xb[1] = t_55;
                float t_56;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_56) : "f"(xb[2]), "f"(xb[6]));
                xb[2] = t_56;
                float t_57;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_57) : "f"(xb[3]), "f"(xb[7]));
                xb[3] = t_57;
                float t_58;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_58) : "f"(xb[0]), "f"(xb[2]));
                xb[0] = t_58;
                float t_59;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_59) : "f"(xb[1]), "f"(xb[3]));
                xb[1] = t_59;
                sums[0] = xa[0];
                sums[1] = xa[1];
                sums[2] = xb[0];
                sums[3] = xb[1];
            } else if (s1 * 128 + j + 64 >= nc) {
                float xa_1[16];
                float xb_1[16];
                long long row_1 = (((long long)(s1 * 128 + j) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long step_1 = (long long)num_heads * 2048ll;
                long long end = (((long long)(nc - 1) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long at_0 = ((s1 * 128 + j < nc) ? row_1 : end);
                {
                    float2 _v2_32 = *reinterpret_cast<const float2*>(dA_part + at_0);
                    xa_1[0] = _v2_32.x;
                    xa_1[0 + 1] = _v2_32.y;
                }
                {
                    float2 _v2_33 = *reinterpret_cast<const float2*>(db_part + at_0);
                    xb_1[0] = _v2_33.x;
                    xb_1[0 + 1] = _v2_33.y;
                }
                long long at_1 = ((s1 * 128 + j + 8 < nc) ? row_1 + step_1 : end);
                {
                    float2 _v2_34 = *reinterpret_cast<const float2*>(dA_part + at_1);
                    xa_1[2] = _v2_34.x;
                    xa_1[2 + 1] = _v2_34.y;
                }
                {
                    float2 _v2_35 = *reinterpret_cast<const float2*>(db_part + at_1);
                    xb_1[2] = _v2_35.x;
                    xb_1[2 + 1] = _v2_35.y;
                }
                long long at_2 = ((s1 * 128 + j + 16 < nc) ? row_1 + step_1 * 2ll : end);
                {
                    float2 _v2_36 = *reinterpret_cast<const float2*>(dA_part + at_2);
                    xa_1[4] = _v2_36.x;
                    xa_1[4 + 1] = _v2_36.y;
                }
                {
                    float2 _v2_37 = *reinterpret_cast<const float2*>(db_part + at_2);
                    xb_1[4] = _v2_37.x;
                    xb_1[4 + 1] = _v2_37.y;
                }
                long long at_3 = ((s1 * 128 + j + 24 < nc) ? row_1 + step_1 * 3ll : end);
                {
                    float2 _v2_38 = *reinterpret_cast<const float2*>(dA_part + at_3);
                    xa_1[6] = _v2_38.x;
                    xa_1[6 + 1] = _v2_38.y;
                }
                {
                    float2 _v2_39 = *reinterpret_cast<const float2*>(db_part + at_3);
                    xb_1[6] = _v2_39.x;
                    xb_1[6 + 1] = _v2_39.y;
                }
                long long at_4 = ((s1 * 128 + j + 32 < nc) ? row_1 + step_1 * 4ll : end);
                {
                    float2 _v2_40 = *reinterpret_cast<const float2*>(dA_part + at_4);
                    xa_1[8] = _v2_40.x;
                    xa_1[8 + 1] = _v2_40.y;
                }
                {
                    float2 _v2_41 = *reinterpret_cast<const float2*>(db_part + at_4);
                    xb_1[8] = _v2_41.x;
                    xb_1[8 + 1] = _v2_41.y;
                }
                long long at_5 = ((s1 * 128 + j + 40 < nc) ? row_1 + step_1 * 5ll : end);
                {
                    float2 _v2_42 = *reinterpret_cast<const float2*>(dA_part + at_5);
                    xa_1[10] = _v2_42.x;
                    xa_1[10 + 1] = _v2_42.y;
                }
                {
                    float2 _v2_43 = *reinterpret_cast<const float2*>(db_part + at_5);
                    xb_1[10] = _v2_43.x;
                    xb_1[10 + 1] = _v2_43.y;
                }
                long long at_6 = ((s1 * 128 + j + 48 < nc) ? row_1 + step_1 * 6ll : end);
                {
                    float2 _v2_44 = *reinterpret_cast<const float2*>(dA_part + at_6);
                    xa_1[12] = _v2_44.x;
                    xa_1[12 + 1] = _v2_44.y;
                }
                {
                    float2 _v2_45 = *reinterpret_cast<const float2*>(db_part + at_6);
                    xb_1[12] = _v2_45.x;
                    xb_1[12 + 1] = _v2_45.y;
                }
                long long at_7 = ((s1 * 128 + j + 56 < nc) ? row_1 + step_1 * 7ll : end);
                {
                    float2 _v2_46 = *reinterpret_cast<const float2*>(dA_part + at_7);
                    xa_1[14] = _v2_46.x;
                    xa_1[14 + 1] = _v2_46.y;
                }
                {
                    float2 _v2_47 = *reinterpret_cast<const float2*>(db_part + at_7);
                    xb_1[14] = _v2_47.x;
                    xb_1[14 + 1] = _v2_47.y;
                }
                xa_1[0] = ((s1 * 128 + j < nc) ? xa_1[0] : 0.0f);
                xb_1[0] = ((s1 * 128 + j < nc) ? xb_1[0] : 0.0f);
                xa_1[1] = ((s1 * 128 + j < nc) ? xa_1[1] : 0.0f);
                xb_1[1] = ((s1 * 128 + j < nc) ? xb_1[1] : 0.0f);
                xa_1[2] = ((s1 * 128 + j + 8 < nc) ? xa_1[2] : 0.0f);
                xb_1[2] = ((s1 * 128 + j + 8 < nc) ? xb_1[2] : 0.0f);
                xa_1[3] = ((s1 * 128 + j + 8 < nc) ? xa_1[3] : 0.0f);
                xb_1[3] = ((s1 * 128 + j + 8 < nc) ? xb_1[3] : 0.0f);
                xa_1[4] = ((s1 * 128 + j + 16 < nc) ? xa_1[4] : 0.0f);
                xb_1[4] = ((s1 * 128 + j + 16 < nc) ? xb_1[4] : 0.0f);
                xa_1[5] = ((s1 * 128 + j + 16 < nc) ? xa_1[5] : 0.0f);
                xb_1[5] = ((s1 * 128 + j + 16 < nc) ? xb_1[5] : 0.0f);
                xa_1[6] = ((s1 * 128 + j + 24 < nc) ? xa_1[6] : 0.0f);
                xb_1[6] = ((s1 * 128 + j + 24 < nc) ? xb_1[6] : 0.0f);
                xa_1[7] = ((s1 * 128 + j + 24 < nc) ? xa_1[7] : 0.0f);
                xb_1[7] = ((s1 * 128 + j + 24 < nc) ? xb_1[7] : 0.0f);
                xa_1[8] = ((s1 * 128 + j + 32 < nc) ? xa_1[8] : 0.0f);
                xb_1[8] = ((s1 * 128 + j + 32 < nc) ? xb_1[8] : 0.0f);
                xa_1[9] = ((s1 * 128 + j + 32 < nc) ? xa_1[9] : 0.0f);
                xb_1[9] = ((s1 * 128 + j + 32 < nc) ? xb_1[9] : 0.0f);
                xa_1[10] = ((s1 * 128 + j + 40 < nc) ? xa_1[10] : 0.0f);
                xb_1[10] = ((s1 * 128 + j + 40 < nc) ? xb_1[10] : 0.0f);
                xa_1[11] = ((s1 * 128 + j + 40 < nc) ? xa_1[11] : 0.0f);
                xb_1[11] = ((s1 * 128 + j + 40 < nc) ? xb_1[11] : 0.0f);
                xa_1[12] = ((s1 * 128 + j + 48 < nc) ? xa_1[12] : 0.0f);
                xb_1[12] = ((s1 * 128 + j + 48 < nc) ? xb_1[12] : 0.0f);
                xa_1[13] = ((s1 * 128 + j + 48 < nc) ? xa_1[13] : 0.0f);
                xb_1[13] = ((s1 * 128 + j + 48 < nc) ? xb_1[13] : 0.0f);
                xa_1[14] = ((s1 * 128 + j + 56 < nc) ? xa_1[14] : 0.0f);
                xb_1[14] = ((s1 * 128 + j + 56 < nc) ? xb_1[14] : 0.0f);
                xa_1[15] = ((s1 * 128 + j + 56 < nc) ? xa_1[15] : 0.0f);
                xb_1[15] = ((s1 * 128 + j + 56 < nc) ? xb_1[15] : 0.0f);
                float t0_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_0) : "f"(xa_1[0]), "f"(0.0f));
                xa_1[0] = t0_0;
                float t0_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_1) : "f"(xa_1[1]), "f"(0.0f));
                xa_1[1] = t0_1;
                float t0_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_2) : "f"(xa_1[2]), "f"(0.0f));
                xa_1[2] = t0_2;
                float t0_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_3) : "f"(xa_1[3]), "f"(0.0f));
                xa_1[3] = t0_3;
                float t0_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_4) : "f"(xa_1[4]), "f"(0.0f));
                xa_1[4] = t0_4;
                float t0_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_5) : "f"(xa_1[5]), "f"(0.0f));
                xa_1[5] = t0_5;
                float t0_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_6) : "f"(xa_1[6]), "f"(0.0f));
                xa_1[6] = t0_6;
                float t0_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_7) : "f"(xa_1[7]), "f"(0.0f));
                xa_1[7] = t0_7;
                float t0_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_8) : "f"(xa_1[8]), "f"(0.0f));
                xa_1[8] = t0_8;
                float t0_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_9) : "f"(xa_1[9]), "f"(0.0f));
                xa_1[9] = t0_9;
                float t0_10;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_10) : "f"(xa_1[10]), "f"(0.0f));
                xa_1[10] = t0_10;
                float t0_11;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_11) : "f"(xa_1[11]), "f"(0.0f));
                xa_1[11] = t0_11;
                float t0_12;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_12) : "f"(xa_1[12]), "f"(0.0f));
                xa_1[12] = t0_12;
                float t0_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_13) : "f"(xa_1[13]), "f"(0.0f));
                xa_1[13] = t0_13;
                float t0_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_14) : "f"(xa_1[14]), "f"(0.0f));
                xa_1[14] = t0_14;
                float t0_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_15) : "f"(xa_1[15]), "f"(0.0f));
                xa_1[15] = t0_15;
                float t_60;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_60) : "f"(xa_1[0]), "f"(xa_1[8]));
                xa_1[0] = t_60;
                float t_61;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_61) : "f"(xa_1[1]), "f"(xa_1[9]));
                xa_1[1] = t_61;
                float t_62;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_62) : "f"(xa_1[2]), "f"(xa_1[10]));
                xa_1[2] = t_62;
                float t_63;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_63) : "f"(xa_1[3]), "f"(xa_1[11]));
                xa_1[3] = t_63;
                float t_64;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_64) : "f"(xa_1[4]), "f"(xa_1[12]));
                xa_1[4] = t_64;
                float t_65;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_65) : "f"(xa_1[5]), "f"(xa_1[13]));
                xa_1[5] = t_65;
                float t_66;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_66) : "f"(xa_1[6]), "f"(xa_1[14]));
                xa_1[6] = t_66;
                float t_67;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_67) : "f"(xa_1[7]), "f"(xa_1[15]));
                xa_1[7] = t_67;
                float t_68;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_68) : "f"(xa_1[0]), "f"(xa_1[4]));
                xa_1[0] = t_68;
                float t_69;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_69) : "f"(xa_1[1]), "f"(xa_1[5]));
                xa_1[1] = t_69;
                float t_70;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_70) : "f"(xa_1[2]), "f"(xa_1[6]));
                xa_1[2] = t_70;
                float t_71;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_71) : "f"(xa_1[3]), "f"(xa_1[7]));
                xa_1[3] = t_71;
                float t_72;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_72) : "f"(xa_1[0]), "f"(xa_1[2]));
                xa_1[0] = t_72;
                float t_73;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_73) : "f"(xa_1[1]), "f"(xa_1[3]));
                xa_1[1] = t_73;
                float t0_16;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_16) : "f"(xb_1[0]), "f"(0.0f));
                xb_1[0] = t0_16;
                float t0_17;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_17) : "f"(xb_1[1]), "f"(0.0f));
                xb_1[1] = t0_17;
                float t0_18;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_18) : "f"(xb_1[2]), "f"(0.0f));
                xb_1[2] = t0_18;
                float t0_19;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_19) : "f"(xb_1[3]), "f"(0.0f));
                xb_1[3] = t0_19;
                float t0_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_20) : "f"(xb_1[4]), "f"(0.0f));
                xb_1[4] = t0_20;
                float t0_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_21) : "f"(xb_1[5]), "f"(0.0f));
                xb_1[5] = t0_21;
                float t0_22;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_22) : "f"(xb_1[6]), "f"(0.0f));
                xb_1[6] = t0_22;
                float t0_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_23) : "f"(xb_1[7]), "f"(0.0f));
                xb_1[7] = t0_23;
                float t0_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_24) : "f"(xb_1[8]), "f"(0.0f));
                xb_1[8] = t0_24;
                float t0_25;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_25) : "f"(xb_1[9]), "f"(0.0f));
                xb_1[9] = t0_25;
                float t0_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_26) : "f"(xb_1[10]), "f"(0.0f));
                xb_1[10] = t0_26;
                float t0_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_27) : "f"(xb_1[11]), "f"(0.0f));
                xb_1[11] = t0_27;
                float t0_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_28) : "f"(xb_1[12]), "f"(0.0f));
                xb_1[12] = t0_28;
                float t0_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_29) : "f"(xb_1[13]), "f"(0.0f));
                xb_1[13] = t0_29;
                float t0_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_30) : "f"(xb_1[14]), "f"(0.0f));
                xb_1[14] = t0_30;
                float t0_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_31) : "f"(xb_1[15]), "f"(0.0f));
                xb_1[15] = t0_31;
                float t_74;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_74) : "f"(xb_1[0]), "f"(xb_1[8]));
                xb_1[0] = t_74;
                float t_75;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_75) : "f"(xb_1[1]), "f"(xb_1[9]));
                xb_1[1] = t_75;
                float t_76;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_76) : "f"(xb_1[2]), "f"(xb_1[10]));
                xb_1[2] = t_76;
                float t_77;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_77) : "f"(xb_1[3]), "f"(xb_1[11]));
                xb_1[3] = t_77;
                float t_78;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_78) : "f"(xb_1[4]), "f"(xb_1[12]));
                xb_1[4] = t_78;
                float t_79;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_79) : "f"(xb_1[5]), "f"(xb_1[13]));
                xb_1[5] = t_79;
                float t_80;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_80) : "f"(xb_1[6]), "f"(xb_1[14]));
                xb_1[6] = t_80;
                float t_81;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_81) : "f"(xb_1[7]), "f"(xb_1[15]));
                xb_1[7] = t_81;
                float t_82;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_82) : "f"(xb_1[0]), "f"(xb_1[4]));
                xb_1[0] = t_82;
                float t_83;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_83) : "f"(xb_1[1]), "f"(xb_1[5]));
                xb_1[1] = t_83;
                float t_84;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_84) : "f"(xb_1[2]), "f"(xb_1[6]));
                xb_1[2] = t_84;
                float t_85;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_85) : "f"(xb_1[3]), "f"(xb_1[7]));
                xb_1[3] = t_85;
                float t_86;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_86) : "f"(xb_1[0]), "f"(xb_1[2]));
                xb_1[0] = t_86;
                float t_87;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_87) : "f"(xb_1[1]), "f"(xb_1[3]));
                xb_1[1] = t_87;
                sums[0] = xa_1[0];
                sums[1] = xa_1[1];
                sums[2] = xb_1[0];
                sums[3] = xb_1[1];
            } else {
                float xa_2[32];
                float xb_2[32];
                long long row_2 = (((long long)(s1 * 128 + j) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long step_2 = (long long)num_heads * 2048ll;
                long long end_1 = (((long long)(nc - 1) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long at_8 = ((s1 * 128 + j < nc) ? row_2 : end_1);
                {
                    float2 _v2_48 = *reinterpret_cast<const float2*>(dA_part + at_8);
                    xa_2[0] = _v2_48.x;
                    xa_2[0 + 1] = _v2_48.y;
                }
                {
                    float2 _v2_49 = *reinterpret_cast<const float2*>(db_part + at_8);
                    xb_2[0] = _v2_49.x;
                    xb_2[0 + 1] = _v2_49.y;
                }
                long long at_9 = ((s1 * 128 + j + 8 < nc) ? row_2 + step_2 : end_1);
                {
                    float2 _v2_50 = *reinterpret_cast<const float2*>(dA_part + at_9);
                    xa_2[2] = _v2_50.x;
                    xa_2[2 + 1] = _v2_50.y;
                }
                {
                    float2 _v2_51 = *reinterpret_cast<const float2*>(db_part + at_9);
                    xb_2[2] = _v2_51.x;
                    xb_2[2 + 1] = _v2_51.y;
                }
                long long at_10 = ((s1 * 128 + j + 16 < nc) ? row_2 + step_2 * 2ll : end_1);
                {
                    float2 _v2_52 = *reinterpret_cast<const float2*>(dA_part + at_10);
                    xa_2[4] = _v2_52.x;
                    xa_2[4 + 1] = _v2_52.y;
                }
                {
                    float2 _v2_53 = *reinterpret_cast<const float2*>(db_part + at_10);
                    xb_2[4] = _v2_53.x;
                    xb_2[4 + 1] = _v2_53.y;
                }
                long long at_11 = ((s1 * 128 + j + 24 < nc) ? row_2 + step_2 * 3ll : end_1);
                {
                    float2 _v2_54 = *reinterpret_cast<const float2*>(dA_part + at_11);
                    xa_2[6] = _v2_54.x;
                    xa_2[6 + 1] = _v2_54.y;
                }
                {
                    float2 _v2_55 = *reinterpret_cast<const float2*>(db_part + at_11);
                    xb_2[6] = _v2_55.x;
                    xb_2[6 + 1] = _v2_55.y;
                }
                long long at_12 = ((s1 * 128 + j + 32 < nc) ? row_2 + step_2 * 4ll : end_1);
                {
                    float2 _v2_56 = *reinterpret_cast<const float2*>(dA_part + at_12);
                    xa_2[8] = _v2_56.x;
                    xa_2[8 + 1] = _v2_56.y;
                }
                {
                    float2 _v2_57 = *reinterpret_cast<const float2*>(db_part + at_12);
                    xb_2[8] = _v2_57.x;
                    xb_2[8 + 1] = _v2_57.y;
                }
                long long at_13 = ((s1 * 128 + j + 40 < nc) ? row_2 + step_2 * 5ll : end_1);
                {
                    float2 _v2_58 = *reinterpret_cast<const float2*>(dA_part + at_13);
                    xa_2[10] = _v2_58.x;
                    xa_2[10 + 1] = _v2_58.y;
                }
                {
                    float2 _v2_59 = *reinterpret_cast<const float2*>(db_part + at_13);
                    xb_2[10] = _v2_59.x;
                    xb_2[10 + 1] = _v2_59.y;
                }
                long long at_14 = ((s1 * 128 + j + 48 < nc) ? row_2 + step_2 * 6ll : end_1);
                {
                    float2 _v2_60 = *reinterpret_cast<const float2*>(dA_part + at_14);
                    xa_2[12] = _v2_60.x;
                    xa_2[12 + 1] = _v2_60.y;
                }
                {
                    float2 _v2_61 = *reinterpret_cast<const float2*>(db_part + at_14);
                    xb_2[12] = _v2_61.x;
                    xb_2[12 + 1] = _v2_61.y;
                }
                long long at_15 = ((s1 * 128 + j + 56 < nc) ? row_2 + step_2 * 7ll : end_1);
                {
                    float2 _v2_62 = *reinterpret_cast<const float2*>(dA_part + at_15);
                    xa_2[14] = _v2_62.x;
                    xa_2[14 + 1] = _v2_62.y;
                }
                {
                    float2 _v2_63 = *reinterpret_cast<const float2*>(db_part + at_15);
                    xb_2[14] = _v2_63.x;
                    xb_2[14 + 1] = _v2_63.y;
                }
                long long at_16 = ((s1 * 128 + j + 64 < nc) ? row_2 + step_2 * 8ll : end_1);
                {
                    float2 _v2_64 = *reinterpret_cast<const float2*>(dA_part + at_16);
                    xa_2[16] = _v2_64.x;
                    xa_2[16 + 1] = _v2_64.y;
                }
                {
                    float2 _v2_65 = *reinterpret_cast<const float2*>(db_part + at_16);
                    xb_2[16] = _v2_65.x;
                    xb_2[16 + 1] = _v2_65.y;
                }
                long long at_17 = ((s1 * 128 + j + 72 < nc) ? row_2 + step_2 * 9ll : end_1);
                {
                    float2 _v2_66 = *reinterpret_cast<const float2*>(dA_part + at_17);
                    xa_2[18] = _v2_66.x;
                    xa_2[18 + 1] = _v2_66.y;
                }
                {
                    float2 _v2_67 = *reinterpret_cast<const float2*>(db_part + at_17);
                    xb_2[18] = _v2_67.x;
                    xb_2[18 + 1] = _v2_67.y;
                }
                long long at_18 = ((s1 * 128 + j + 80 < nc) ? row_2 + step_2 * 10ll : end_1);
                {
                    float2 _v2_68 = *reinterpret_cast<const float2*>(dA_part + at_18);
                    xa_2[20] = _v2_68.x;
                    xa_2[20 + 1] = _v2_68.y;
                }
                {
                    float2 _v2_69 = *reinterpret_cast<const float2*>(db_part + at_18);
                    xb_2[20] = _v2_69.x;
                    xb_2[20 + 1] = _v2_69.y;
                }
                long long at_19 = ((s1 * 128 + j + 88 < nc) ? row_2 + step_2 * 11ll : end_1);
                {
                    float2 _v2_70 = *reinterpret_cast<const float2*>(dA_part + at_19);
                    xa_2[22] = _v2_70.x;
                    xa_2[22 + 1] = _v2_70.y;
                }
                {
                    float2 _v2_71 = *reinterpret_cast<const float2*>(db_part + at_19);
                    xb_2[22] = _v2_71.x;
                    xb_2[22 + 1] = _v2_71.y;
                }
                long long at_20 = ((s1 * 128 + j + 96 < nc) ? row_2 + step_2 * 12ll : end_1);
                {
                    float2 _v2_72 = *reinterpret_cast<const float2*>(dA_part + at_20);
                    xa_2[24] = _v2_72.x;
                    xa_2[24 + 1] = _v2_72.y;
                }
                {
                    float2 _v2_73 = *reinterpret_cast<const float2*>(db_part + at_20);
                    xb_2[24] = _v2_73.x;
                    xb_2[24 + 1] = _v2_73.y;
                }
                long long at_21 = ((s1 * 128 + j + 104 < nc) ? row_2 + step_2 * 13ll : end_1);
                {
                    float2 _v2_74 = *reinterpret_cast<const float2*>(dA_part + at_21);
                    xa_2[26] = _v2_74.x;
                    xa_2[26 + 1] = _v2_74.y;
                }
                {
                    float2 _v2_75 = *reinterpret_cast<const float2*>(db_part + at_21);
                    xb_2[26] = _v2_75.x;
                    xb_2[26 + 1] = _v2_75.y;
                }
                long long at_22 = ((s1 * 128 + j + 112 < nc) ? row_2 + step_2 * 14ll : end_1);
                {
                    float2 _v2_76 = *reinterpret_cast<const float2*>(dA_part + at_22);
                    xa_2[28] = _v2_76.x;
                    xa_2[28 + 1] = _v2_76.y;
                }
                {
                    float2 _v2_77 = *reinterpret_cast<const float2*>(db_part + at_22);
                    xb_2[28] = _v2_77.x;
                    xb_2[28 + 1] = _v2_77.y;
                }
                long long at_23 = ((s1 * 128 + j + 120 < nc) ? row_2 + step_2 * 15ll : end_1);
                {
                    float2 _v2_78 = *reinterpret_cast<const float2*>(dA_part + at_23);
                    xa_2[30] = _v2_78.x;
                    xa_2[30 + 1] = _v2_78.y;
                }
                {
                    float2 _v2_79 = *reinterpret_cast<const float2*>(db_part + at_23);
                    xb_2[30] = _v2_79.x;
                    xb_2[30 + 1] = _v2_79.y;
                }
                xa_2[0] = ((s1 * 128 + j < nc) ? xa_2[0] : 0.0f);
                xb_2[0] = ((s1 * 128 + j < nc) ? xb_2[0] : 0.0f);
                xa_2[1] = ((s1 * 128 + j < nc) ? xa_2[1] : 0.0f);
                xb_2[1] = ((s1 * 128 + j < nc) ? xb_2[1] : 0.0f);
                xa_2[2] = ((s1 * 128 + j + 8 < nc) ? xa_2[2] : 0.0f);
                xb_2[2] = ((s1 * 128 + j + 8 < nc) ? xb_2[2] : 0.0f);
                xa_2[3] = ((s1 * 128 + j + 8 < nc) ? xa_2[3] : 0.0f);
                xb_2[3] = ((s1 * 128 + j + 8 < nc) ? xb_2[3] : 0.0f);
                xa_2[4] = ((s1 * 128 + j + 16 < nc) ? xa_2[4] : 0.0f);
                xb_2[4] = ((s1 * 128 + j + 16 < nc) ? xb_2[4] : 0.0f);
                xa_2[5] = ((s1 * 128 + j + 16 < nc) ? xa_2[5] : 0.0f);
                xb_2[5] = ((s1 * 128 + j + 16 < nc) ? xb_2[5] : 0.0f);
                xa_2[6] = ((s1 * 128 + j + 24 < nc) ? xa_2[6] : 0.0f);
                xb_2[6] = ((s1 * 128 + j + 24 < nc) ? xb_2[6] : 0.0f);
                xa_2[7] = ((s1 * 128 + j + 24 < nc) ? xa_2[7] : 0.0f);
                xb_2[7] = ((s1 * 128 + j + 24 < nc) ? xb_2[7] : 0.0f);
                xa_2[8] = ((s1 * 128 + j + 32 < nc) ? xa_2[8] : 0.0f);
                xb_2[8] = ((s1 * 128 + j + 32 < nc) ? xb_2[8] : 0.0f);
                xa_2[9] = ((s1 * 128 + j + 32 < nc) ? xa_2[9] : 0.0f);
                xb_2[9] = ((s1 * 128 + j + 32 < nc) ? xb_2[9] : 0.0f);
                xa_2[10] = ((s1 * 128 + j + 40 < nc) ? xa_2[10] : 0.0f);
                xb_2[10] = ((s1 * 128 + j + 40 < nc) ? xb_2[10] : 0.0f);
                xa_2[11] = ((s1 * 128 + j + 40 < nc) ? xa_2[11] : 0.0f);
                xb_2[11] = ((s1 * 128 + j + 40 < nc) ? xb_2[11] : 0.0f);
                xa_2[12] = ((s1 * 128 + j + 48 < nc) ? xa_2[12] : 0.0f);
                xb_2[12] = ((s1 * 128 + j + 48 < nc) ? xb_2[12] : 0.0f);
                xa_2[13] = ((s1 * 128 + j + 48 < nc) ? xa_2[13] : 0.0f);
                xb_2[13] = ((s1 * 128 + j + 48 < nc) ? xb_2[13] : 0.0f);
                xa_2[14] = ((s1 * 128 + j + 56 < nc) ? xa_2[14] : 0.0f);
                xb_2[14] = ((s1 * 128 + j + 56 < nc) ? xb_2[14] : 0.0f);
                xa_2[15] = ((s1 * 128 + j + 56 < nc) ? xa_2[15] : 0.0f);
                xb_2[15] = ((s1 * 128 + j + 56 < nc) ? xb_2[15] : 0.0f);
                xa_2[16] = ((s1 * 128 + j + 64 < nc) ? xa_2[16] : 0.0f);
                xb_2[16] = ((s1 * 128 + j + 64 < nc) ? xb_2[16] : 0.0f);
                xa_2[17] = ((s1 * 128 + j + 64 < nc) ? xa_2[17] : 0.0f);
                xb_2[17] = ((s1 * 128 + j + 64 < nc) ? xb_2[17] : 0.0f);
                xa_2[18] = ((s1 * 128 + j + 72 < nc) ? xa_2[18] : 0.0f);
                xb_2[18] = ((s1 * 128 + j + 72 < nc) ? xb_2[18] : 0.0f);
                xa_2[19] = ((s1 * 128 + j + 72 < nc) ? xa_2[19] : 0.0f);
                xb_2[19] = ((s1 * 128 + j + 72 < nc) ? xb_2[19] : 0.0f);
                xa_2[20] = ((s1 * 128 + j + 80 < nc) ? xa_2[20] : 0.0f);
                xb_2[20] = ((s1 * 128 + j + 80 < nc) ? xb_2[20] : 0.0f);
                xa_2[21] = ((s1 * 128 + j + 80 < nc) ? xa_2[21] : 0.0f);
                xb_2[21] = ((s1 * 128 + j + 80 < nc) ? xb_2[21] : 0.0f);
                xa_2[22] = ((s1 * 128 + j + 88 < nc) ? xa_2[22] : 0.0f);
                xb_2[22] = ((s1 * 128 + j + 88 < nc) ? xb_2[22] : 0.0f);
                xa_2[23] = ((s1 * 128 + j + 88 < nc) ? xa_2[23] : 0.0f);
                xb_2[23] = ((s1 * 128 + j + 88 < nc) ? xb_2[23] : 0.0f);
                xa_2[24] = ((s1 * 128 + j + 96 < nc) ? xa_2[24] : 0.0f);
                xb_2[24] = ((s1 * 128 + j + 96 < nc) ? xb_2[24] : 0.0f);
                xa_2[25] = ((s1 * 128 + j + 96 < nc) ? xa_2[25] : 0.0f);
                xb_2[25] = ((s1 * 128 + j + 96 < nc) ? xb_2[25] : 0.0f);
                xa_2[26] = ((s1 * 128 + j + 104 < nc) ? xa_2[26] : 0.0f);
                xb_2[26] = ((s1 * 128 + j + 104 < nc) ? xb_2[26] : 0.0f);
                xa_2[27] = ((s1 * 128 + j + 104 < nc) ? xa_2[27] : 0.0f);
                xb_2[27] = ((s1 * 128 + j + 104 < nc) ? xb_2[27] : 0.0f);
                xa_2[28] = ((s1 * 128 + j + 112 < nc) ? xa_2[28] : 0.0f);
                xb_2[28] = ((s1 * 128 + j + 112 < nc) ? xb_2[28] : 0.0f);
                xa_2[29] = ((s1 * 128 + j + 112 < nc) ? xa_2[29] : 0.0f);
                xb_2[29] = ((s1 * 128 + j + 112 < nc) ? xb_2[29] : 0.0f);
                xa_2[30] = ((s1 * 128 + j + 120 < nc) ? xa_2[30] : 0.0f);
                xb_2[30] = ((s1 * 128 + j + 120 < nc) ? xb_2[30] : 0.0f);
                xa_2[31] = ((s1 * 128 + j + 120 < nc) ? xa_2[31] : 0.0f);
                xb_2[31] = ((s1 * 128 + j + 120 < nc) ? xb_2[31] : 0.0f);
                float t_88;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_88) : "f"(xa_2[0]), "f"(xa_2[16]));
                xa_2[0] = t_88;
                float t_89;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_89) : "f"(xa_2[1]), "f"(xa_2[17]));
                xa_2[1] = t_89;
                float t_90;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_90) : "f"(xa_2[2]), "f"(xa_2[18]));
                xa_2[2] = t_90;
                float t_91;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_91) : "f"(xa_2[3]), "f"(xa_2[19]));
                xa_2[3] = t_91;
                float t_92;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_92) : "f"(xa_2[4]), "f"(xa_2[20]));
                xa_2[4] = t_92;
                float t_93;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_93) : "f"(xa_2[5]), "f"(xa_2[21]));
                xa_2[5] = t_93;
                float t_94;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_94) : "f"(xa_2[6]), "f"(xa_2[22]));
                xa_2[6] = t_94;
                float t_95;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_95) : "f"(xa_2[7]), "f"(xa_2[23]));
                xa_2[7] = t_95;
                float t_96;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_96) : "f"(xa_2[8]), "f"(xa_2[24]));
                xa_2[8] = t_96;
                float t_97;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_97) : "f"(xa_2[9]), "f"(xa_2[25]));
                xa_2[9] = t_97;
                float t_98;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_98) : "f"(xa_2[10]), "f"(xa_2[26]));
                xa_2[10] = t_98;
                float t_99;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_99) : "f"(xa_2[11]), "f"(xa_2[27]));
                xa_2[11] = t_99;
                float t_100;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_100) : "f"(xa_2[12]), "f"(xa_2[28]));
                xa_2[12] = t_100;
                float t_101;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_101) : "f"(xa_2[13]), "f"(xa_2[29]));
                xa_2[13] = t_101;
                float t_102;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_102) : "f"(xa_2[14]), "f"(xa_2[30]));
                xa_2[14] = t_102;
                float t_103;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_103) : "f"(xa_2[15]), "f"(xa_2[31]));
                xa_2[15] = t_103;
                float t_104;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_104) : "f"(xa_2[0]), "f"(xa_2[8]));
                xa_2[0] = t_104;
                float t_105;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_105) : "f"(xa_2[1]), "f"(xa_2[9]));
                xa_2[1] = t_105;
                float t_106;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_106) : "f"(xa_2[2]), "f"(xa_2[10]));
                xa_2[2] = t_106;
                float t_107;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_107) : "f"(xa_2[3]), "f"(xa_2[11]));
                xa_2[3] = t_107;
                float t_108;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_108) : "f"(xa_2[4]), "f"(xa_2[12]));
                xa_2[4] = t_108;
                float t_109;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_109) : "f"(xa_2[5]), "f"(xa_2[13]));
                xa_2[5] = t_109;
                float t_110;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_110) : "f"(xa_2[6]), "f"(xa_2[14]));
                xa_2[6] = t_110;
                float t_111;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_111) : "f"(xa_2[7]), "f"(xa_2[15]));
                xa_2[7] = t_111;
                float t_112;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_112) : "f"(xa_2[0]), "f"(xa_2[4]));
                xa_2[0] = t_112;
                float t_113;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_113) : "f"(xa_2[1]), "f"(xa_2[5]));
                xa_2[1] = t_113;
                float t_114;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_114) : "f"(xa_2[2]), "f"(xa_2[6]));
                xa_2[2] = t_114;
                float t_115;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_115) : "f"(xa_2[3]), "f"(xa_2[7]));
                xa_2[3] = t_115;
                float t_116;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_116) : "f"(xa_2[0]), "f"(xa_2[2]));
                xa_2[0] = t_116;
                float t_117;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_117) : "f"(xa_2[1]), "f"(xa_2[3]));
                xa_2[1] = t_117;
                float t_118;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_118) : "f"(xb_2[0]), "f"(xb_2[16]));
                xb_2[0] = t_118;
                float t_119;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_119) : "f"(xb_2[1]), "f"(xb_2[17]));
                xb_2[1] = t_119;
                float t_120;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_120) : "f"(xb_2[2]), "f"(xb_2[18]));
                xb_2[2] = t_120;
                float t_121;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_121) : "f"(xb_2[3]), "f"(xb_2[19]));
                xb_2[3] = t_121;
                float t_122;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_122) : "f"(xb_2[4]), "f"(xb_2[20]));
                xb_2[4] = t_122;
                float t_123;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_123) : "f"(xb_2[5]), "f"(xb_2[21]));
                xb_2[5] = t_123;
                float t_124;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_124) : "f"(xb_2[6]), "f"(xb_2[22]));
                xb_2[6] = t_124;
                float t_125;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_125) : "f"(xb_2[7]), "f"(xb_2[23]));
                xb_2[7] = t_125;
                float t_126;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_126) : "f"(xb_2[8]), "f"(xb_2[24]));
                xb_2[8] = t_126;
                float t_127;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_127) : "f"(xb_2[9]), "f"(xb_2[25]));
                xb_2[9] = t_127;
                float t_128;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_128) : "f"(xb_2[10]), "f"(xb_2[26]));
                xb_2[10] = t_128;
                float t_129;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_129) : "f"(xb_2[11]), "f"(xb_2[27]));
                xb_2[11] = t_129;
                float t_130;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_130) : "f"(xb_2[12]), "f"(xb_2[28]));
                xb_2[12] = t_130;
                float t_131;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_131) : "f"(xb_2[13]), "f"(xb_2[29]));
                xb_2[13] = t_131;
                float t_132;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_132) : "f"(xb_2[14]), "f"(xb_2[30]));
                xb_2[14] = t_132;
                float t_133;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_133) : "f"(xb_2[15]), "f"(xb_2[31]));
                xb_2[15] = t_133;
                float t_134;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_134) : "f"(xb_2[0]), "f"(xb_2[8]));
                xb_2[0] = t_134;
                float t_135;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_135) : "f"(xb_2[1]), "f"(xb_2[9]));
                xb_2[1] = t_135;
                float t_136;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_136) : "f"(xb_2[2]), "f"(xb_2[10]));
                xb_2[2] = t_136;
                float t_137;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_137) : "f"(xb_2[3]), "f"(xb_2[11]));
                xb_2[3] = t_137;
                float t_138;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_138) : "f"(xb_2[4]), "f"(xb_2[12]));
                xb_2[4] = t_138;
                float t_139;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_139) : "f"(xb_2[5]), "f"(xb_2[13]));
                xb_2[5] = t_139;
                float t_140;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_140) : "f"(xb_2[6]), "f"(xb_2[14]));
                xb_2[6] = t_140;
                float t_141;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_141) : "f"(xb_2[7]), "f"(xb_2[15]));
                xb_2[7] = t_141;
                float t_142;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_142) : "f"(xb_2[0]), "f"(xb_2[4]));
                xb_2[0] = t_142;
                float t_143;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_143) : "f"(xb_2[1]), "f"(xb_2[5]));
                xb_2[1] = t_143;
                float t_144;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_144) : "f"(xb_2[2]), "f"(xb_2[6]));
                xb_2[2] = t_144;
                float t_145;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_145) : "f"(xb_2[3]), "f"(xb_2[7]));
                xb_2[3] = t_145;
                float t_146;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_146) : "f"(xb_2[0]), "f"(xb_2[2]));
                xb_2[0] = t_146;
                float t_147;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_147) : "f"(xb_2[1]), "f"(xb_2[3]));
                xb_2[1] = t_147;
                sums[0] = xa_2[0];
                sums[1] = xa_2[1];
                sums[2] = xb_2[0];
                sums[3] = xb_2[1];
            }
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(slot_addr), "f"(sums[0]), "f"(sums[1]), "f"(sums[2]), "f"(sums[3]) : "memory");
            __syncthreads();
            if (j == 0) {
                float t[32];
                t[0] = sums[0];
                t[1] = sums[1];
                t[2] = sums[2];
                t[3] = sums[3];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t[4])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 3]))
                    : "r"(part_addr + (unsigned int)((128 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t[8])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 3]))
                    : "r"(part_addr + (unsigned int)((256 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t[12])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 3]))
                    : "r"(part_addr + (unsigned int)((384 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t[16])), "=r"(*reinterpret_cast<uint32_t*>(&t[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(16) + 3]))
                    : "r"(part_addr + (unsigned int)((512 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t[20])), "=r"(*reinterpret_cast<uint32_t*>(&t[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(20) + 3]))
                    : "r"(part_addr + (unsigned int)((640 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t[24])), "=r"(*reinterpret_cast<uint32_t*>(&t[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(24) + 3]))
                    : "r"(part_addr + (unsigned int)((768 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t[28])), "=r"(*reinterpret_cast<uint32_t*>(&t[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(28) + 3]))
                    : "r"(part_addr + (unsigned int)((896 + d) * 16)));
                float u_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_0) : "f"(t[0]), "f"(t[16]));
                t[0] = u_0;
                float u_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_1) : "f"(t[1]), "f"(t[17]));
                t[1] = u_1;
                float u_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_2) : "f"(t[2]), "f"(t[18]));
                t[2] = u_2;
                float u_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_3) : "f"(t[3]), "f"(t[19]));
                t[3] = u_3;
                float u_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_4) : "f"(t[4]), "f"(t[20]));
                t[4] = u_4;
                float u_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_5) : "f"(t[5]), "f"(t[21]));
                t[5] = u_5;
                float u_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_6) : "f"(t[6]), "f"(t[22]));
                t[6] = u_6;
                float u_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_7) : "f"(t[7]), "f"(t[23]));
                t[7] = u_7;
                float u_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_8) : "f"(t[8]), "f"(t[24]));
                t[8] = u_8;
                float u_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_9) : "f"(t[9]), "f"(t[25]));
                t[9] = u_9;
                float u_10;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_10) : "f"(t[10]), "f"(t[26]));
                t[10] = u_10;
                float u_11;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_11) : "f"(t[11]), "f"(t[27]));
                t[11] = u_11;
                float u_12;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_12) : "f"(t[12]), "f"(t[28]));
                t[12] = u_12;
                float u_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_13) : "f"(t[13]), "f"(t[29]));
                t[13] = u_13;
                float u_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_14) : "f"(t[14]), "f"(t[30]));
                t[14] = u_14;
                float u_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_15) : "f"(t[15]), "f"(t[31]));
                t[15] = u_15;
                float u_16;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_16) : "f"(t[0]), "f"(t[8]));
                t[0] = u_16;
                float u_17;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_17) : "f"(t[1]), "f"(t[9]));
                t[1] = u_17;
                float u_18;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_18) : "f"(t[2]), "f"(t[10]));
                t[2] = u_18;
                float u_19;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_19) : "f"(t[3]), "f"(t[11]));
                t[3] = u_19;
                float u_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_20) : "f"(t[4]), "f"(t[12]));
                t[4] = u_20;
                float u_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_21) : "f"(t[5]), "f"(t[13]));
                t[5] = u_21;
                float u_22;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_22) : "f"(t[6]), "f"(t[14]));
                t[6] = u_22;
                float u_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_23) : "f"(t[7]), "f"(t[15]));
                t[7] = u_23;
                float u_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_24) : "f"(t[0]), "f"(t[4]));
                t[0] = u_24;
                float u_25;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_25) : "f"(t[1]), "f"(t[5]));
                t[1] = u_25;
                float u_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_26) : "f"(t[2]), "f"(t[6]));
                t[2] = u_26;
                float u_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_27) : "f"(t[3]), "f"(t[7]));
                t[3] = u_27;
                float b01;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(b01) : "f"(t[2]), "f"(t[3]));
                sums[0] = t[0];
                float _add_80;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(b01), "f"(t[1]));
                sums[1] = _add_80;
                if (nseg > 1) {
                    float held[2];
                    float zero[2];
                    zero[0] = 0.0f;
                    zero[1] = 0.0f;
                    unsigned int at_24 = counter_addr + (unsigned int)(d * 8);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&held[0])), "=r"(*reinterpret_cast<uint32_t*>(&held[(0) + 1]))
                        : "r"(at_24));
                    int low_0 = s1 & 1;
                    float merged_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_0) : "f"(held[0]), "f"(sums[0]));
                    sums[0] = ((low_0 == 1) ? merged_0 : sums[0]);
                    float merged_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_1) : "f"(held[1]), "f"(sums[1]));
                    sums[1] = ((low_0 == 1) ? merged_1 : sums[1]);
                    if (low_0 == 1) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_24), "f"(zero[0]), "f"(zero[1]) : "memory");
                    }
                    if (low_0 == 0) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_24), "f"(sums[0]), "f"(sums[1]) : "memory");
                    }
                    unsigned int at_25 = counter_addr + (unsigned int)((128 + d) * 8);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&held[0])), "=r"(*reinterpret_cast<uint32_t*>(&held[(0) + 1]))
                        : "r"(at_25));
                    int low_1 = s1 & 3;
                    float merged_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_2) : "f"(held[0]), "f"(sums[0]));
                    sums[0] = ((low_1 == 3) ? merged_2 : sums[0]);
                    float merged_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_3) : "f"(held[1]), "f"(sums[1]));
                    sums[1] = ((low_1 == 3) ? merged_3 : sums[1]);
                    if (low_1 == 3) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_25), "f"(zero[0]), "f"(zero[1]) : "memory");
                    }
                    if (low_1 == 1) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_25), "f"(sums[0]), "f"(sums[1]) : "memory");
                    }
                    unsigned int at_26 = counter_addr + (unsigned int)((256 + d) * 8);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&held[0])), "=r"(*reinterpret_cast<uint32_t*>(&held[(0) + 1]))
                        : "r"(at_26));
                    int low_2 = s1 & 7;
                    float merged_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_4) : "f"(held[0]), "f"(sums[0]));
                    sums[0] = ((low_2 == 7) ? merged_4 : sums[0]);
                    float merged_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_5) : "f"(held[1]), "f"(sums[1]));
                    sums[1] = ((low_2 == 7) ? merged_5 : sums[1]);
                    if (low_2 == 7) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_26), "f"(zero[0]), "f"(zero[1]) : "memory");
                    }
                    if (low_2 == 3) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_26), "f"(sums[0]), "f"(sums[1]) : "memory");
                    }
                    unsigned int at_27 = counter_addr + (unsigned int)((384 + d) * 8);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&held[0])), "=r"(*reinterpret_cast<uint32_t*>(&held[(0) + 1]))
                        : "r"(at_27));
                    int low_3 = s1 & 15;
                    float merged_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_6) : "f"(held[0]), "f"(sums[0]));
                    sums[0] = ((low_3 == 15) ? merged_6 : sums[0]);
                    float merged_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_7) : "f"(held[1]), "f"(sums[1]));
                    sums[1] = ((low_3 == 15) ? merged_7 : sums[1]);
                    if (low_3 == 15) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_27), "f"(zero[0]), "f"(zero[1]) : "memory");
                    }
                    if (low_3 == 7) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_27), "f"(sums[0]), "f"(sums[1]) : "memory");
                    }
                    unsigned int at_28 = counter_addr + (unsigned int)((512 + d) * 8);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&held[0])), "=r"(*reinterpret_cast<uint32_t*>(&held[(0) + 1]))
                        : "r"(at_28));
                    int low_4 = s1 & 31;
                    float merged_8;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_8) : "f"(held[0]), "f"(sums[0]));
                    sums[0] = ((low_4 == 31) ? merged_8 : sums[0]);
                    float merged_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_9) : "f"(held[1]), "f"(sums[1]));
                    sums[1] = ((low_4 == 31) ? merged_9 : sums[1]);
                    if (low_4 == 31) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_28), "f"(zero[0]), "f"(zero[1]) : "memory");
                    }
                    if (low_4 == 15) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_28), "f"(sums[0]), "f"(sums[1]) : "memory");
                    }
                    unsigned int at_29 = counter_addr + (unsigned int)((640 + d) * 8);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&held[0])), "=r"(*reinterpret_cast<uint32_t*>(&held[(0) + 1]))
                        : "r"(at_29));
                    int low_5 = s1 & 63;
                    float merged_10;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_10) : "f"(held[0]), "f"(sums[0]));
                    sums[0] = ((low_5 == 63) ? merged_10 : sums[0]);
                    float merged_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_11) : "f"(held[1]), "f"(sums[1]));
                    sums[1] = ((low_5 == 63) ? merged_11 : sums[1]);
                    if (low_5 == 63) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_29), "f"(zero[0]), "f"(zero[1]) : "memory");
                    }
                    if (low_5 == 31) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_29), "f"(sums[0]), "f"(sums[1]) : "memory");
                    }
                    unsigned int at_30 = counter_addr + (unsigned int)((768 + d) * 8);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&held[0])), "=r"(*reinterpret_cast<uint32_t*>(&held[(0) + 1]))
                        : "r"(at_30));
                    int low_6 = s1 & 127;
                    float merged_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_12) : "f"(held[0]), "f"(sums[0]));
                    sums[0] = ((low_6 == 127) ? merged_12 : sums[0]);
                    float merged_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_13) : "f"(held[1]), "f"(sums[1]));
                    sums[1] = ((low_6 == 127) ? merged_13 : sums[1]);
                    if (low_6 == 127) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_30), "f"(zero[0]), "f"(zero[1]) : "memory");
                    }
                    if (low_6 == 63) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_30), "f"(sums[0]), "f"(sums[1]) : "memory");
                    }
                    unsigned int at_31 = counter_addr + (unsigned int)((896 + d) * 8);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&held[0])), "=r"(*reinterpret_cast<uint32_t*>(&held[(0) + 1]))
                        : "r"(at_31));
                    int low_7 = s1 & 255;
                    float merged_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_14) : "f"(held[0]), "f"(sums[0]));
                    sums[0] = ((low_7 == 255) ? merged_14 : sums[0]);
                    float merged_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_15) : "f"(held[1]), "f"(sums[1]));
                    sums[1] = ((low_7 == 255) ? merged_15 : sums[1]);
                    if (low_7 == 255) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_31), "f"(zero[0]), "f"(zero[1]) : "memory");
                    }
                    if (low_7 == 127) {
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(at_31), "f"(sums[0]), "f"(sums[1]) : "memory");
                    }
                    float carried_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_0) : "f"(top[0]), "f"(sums[0]));
                    top[0] = (((s1 & 255) == 255) ? carried_0 : top[0]);
                    float carried_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_1) : "f"(top[1]), "f"(sums[1]));
                    top[1] = (((s1 & 255) == 255) ? carried_1 : top[1]);
                }
            }
            __syncthreads();
        }
        if (j == 0) {
            if (nseg > 1) {
                float held_1[2];
                sums[0] = 0.0f;
                sums[1] = 0.0f;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&held_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&held_1[(0) + 1]))
                    : "r"(counter_addr + (unsigned int)(d * 8)));
                float got_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_0) : "f"(held_1[0]), "f"(sums[0]));
                sums[0] = got_0;
                float got_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_1) : "f"(held_1[1]), "f"(sums[1]));
                sums[1] = got_1;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&held_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&held_1[(0) + 1]))
                    : "r"(counter_addr + (unsigned int)((128 + d) * 8)));
                float got_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_2) : "f"(held_1[0]), "f"(sums[0]));
                sums[0] = got_2;
                float got_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_3) : "f"(held_1[1]), "f"(sums[1]));
                sums[1] = got_3;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&held_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&held_1[(0) + 1]))
                    : "r"(counter_addr + (unsigned int)((256 + d) * 8)));
                float got_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_4) : "f"(held_1[0]), "f"(sums[0]));
                sums[0] = got_4;
                float got_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_5) : "f"(held_1[1]), "f"(sums[1]));
                sums[1] = got_5;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&held_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&held_1[(0) + 1]))
                    : "r"(counter_addr + (unsigned int)((384 + d) * 8)));
                float got_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_6) : "f"(held_1[0]), "f"(sums[0]));
                sums[0] = got_6;
                float got_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_7) : "f"(held_1[1]), "f"(sums[1]));
                sums[1] = got_7;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&held_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&held_1[(0) + 1]))
                    : "r"(counter_addr + (unsigned int)((512 + d) * 8)));
                float got_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_8) : "f"(held_1[0]), "f"(sums[0]));
                sums[0] = got_8;
                float got_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_9) : "f"(held_1[1]), "f"(sums[1]));
                sums[1] = got_9;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&held_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&held_1[(0) + 1]))
                    : "r"(counter_addr + (unsigned int)((640 + d) * 8)));
                float got_10;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_10) : "f"(held_1[0]), "f"(sums[0]));
                sums[0] = got_10;
                float got_11;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_11) : "f"(held_1[1]), "f"(sums[1]));
                sums[1] = got_11;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&held_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&held_1[(0) + 1]))
                    : "r"(counter_addr + (unsigned int)((768 + d) * 8)));
                float got_12;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_12) : "f"(held_1[0]), "f"(sums[0]));
                sums[0] = got_12;
                float got_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_13) : "f"(held_1[1]), "f"(sums[1]));
                sums[1] = got_13;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&held_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&held_1[(0) + 1]))
                    : "r"(counter_addr + (unsigned int)((896 + d) * 8)));
                float got_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_14) : "f"(held_1[0]), "f"(sums[0]));
                sums[0] = got_14;
                float got_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_15) : "f"(held_1[1]), "f"(sums[1]));
                sums[1] = got_15;
                float whole_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(whole_0) : "f"(top[0]), "f"(sums[0]));
                sums[0] = whole_0;
                float whole_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(whole_1) : "f"(top[1]), "f"(sums[1]));
                sums[1] = whole_1;
            } else if (nseg == 1) {
                float one_0;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(one_0) : "f"(sums[0]), "f"(0.0f));
                sums[0] = one_0;
                float one_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(one_1) : "f"(sums[1]), "f"(0.0f));
                sums[1] = one_1;
            } else {
                sums[0] = 0.0f;
                sums[1] = 0.0f;
            }
        }
    } else {
        if (p * 128 + 128 > nc) {
            int pc = p * 128 + (tid >> 3);
            long long pw = ((long long)pc * (long long)num_heads + (long long)head) * 256ll + (long long)((tid & 7) * 32);
            if (pc < nc) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dA_part + pw))); }
            if (pc < nc) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(db_part + pw))); }
        }
        for (int s = p; s < nseg; s += g) {
            if (s * 128 + 128 <= nc) {
                float xa_3[32];
                float xb_3[32];
                long long row_3 = (((long long)(s * 128 + j) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long step_3 = (long long)num_heads * 2048ll;
                {
                    float2 _v2_81 = *reinterpret_cast<const float2*>(dA_part + row_3);
                    xa_3[0] = _v2_81.x;
                    xa_3[0 + 1] = _v2_81.y;
                }
                {
                    float2 _v2_82 = *reinterpret_cast<const float2*>(db_part + row_3);
                    xb_3[0] = _v2_82.x;
                    xb_3[0 + 1] = _v2_82.y;
                }
                {
                    float2 _v2_83 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3);
                    xa_3[2] = _v2_83.x;
                    xa_3[2 + 1] = _v2_83.y;
                }
                {
                    float2 _v2_84 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3);
                    xb_3[2] = _v2_84.x;
                    xb_3[2 + 1] = _v2_84.y;
                }
                {
                    float2 _v2_85 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 2ll);
                    xa_3[4] = _v2_85.x;
                    xa_3[4 + 1] = _v2_85.y;
                }
                {
                    float2 _v2_86 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 2ll);
                    xb_3[4] = _v2_86.x;
                    xb_3[4 + 1] = _v2_86.y;
                }
                {
                    float2 _v2_87 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 3ll);
                    xa_3[6] = _v2_87.x;
                    xa_3[6 + 1] = _v2_87.y;
                }
                {
                    float2 _v2_88 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 3ll);
                    xb_3[6] = _v2_88.x;
                    xb_3[6 + 1] = _v2_88.y;
                }
                {
                    float2 _v2_89 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 4ll);
                    xa_3[8] = _v2_89.x;
                    xa_3[8 + 1] = _v2_89.y;
                }
                {
                    float2 _v2_90 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 4ll);
                    xb_3[8] = _v2_90.x;
                    xb_3[8 + 1] = _v2_90.y;
                }
                {
                    float2 _v2_91 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 5ll);
                    xa_3[10] = _v2_91.x;
                    xa_3[10 + 1] = _v2_91.y;
                }
                {
                    float2 _v2_92 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 5ll);
                    xb_3[10] = _v2_92.x;
                    xb_3[10 + 1] = _v2_92.y;
                }
                {
                    float2 _v2_93 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 6ll);
                    xa_3[12] = _v2_93.x;
                    xa_3[12 + 1] = _v2_93.y;
                }
                {
                    float2 _v2_94 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 6ll);
                    xb_3[12] = _v2_94.x;
                    xb_3[12 + 1] = _v2_94.y;
                }
                {
                    float2 _v2_95 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 7ll);
                    xa_3[14] = _v2_95.x;
                    xa_3[14 + 1] = _v2_95.y;
                }
                {
                    float2 _v2_96 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 7ll);
                    xb_3[14] = _v2_96.x;
                    xb_3[14 + 1] = _v2_96.y;
                }
                {
                    float2 _v2_97 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 8ll);
                    xa_3[16] = _v2_97.x;
                    xa_3[16 + 1] = _v2_97.y;
                }
                {
                    float2 _v2_98 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 8ll);
                    xb_3[16] = _v2_98.x;
                    xb_3[16 + 1] = _v2_98.y;
                }
                {
                    float2 _v2_99 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 9ll);
                    xa_3[18] = _v2_99.x;
                    xa_3[18 + 1] = _v2_99.y;
                }
                {
                    float2 _v2_100 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 9ll);
                    xb_3[18] = _v2_100.x;
                    xb_3[18 + 1] = _v2_100.y;
                }
                {
                    float2 _v2_101 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 10ll);
                    xa_3[20] = _v2_101.x;
                    xa_3[20 + 1] = _v2_101.y;
                }
                {
                    float2 _v2_102 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 10ll);
                    xb_3[20] = _v2_102.x;
                    xb_3[20 + 1] = _v2_102.y;
                }
                {
                    float2 _v2_103 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 11ll);
                    xa_3[22] = _v2_103.x;
                    xa_3[22 + 1] = _v2_103.y;
                }
                {
                    float2 _v2_104 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 11ll);
                    xb_3[22] = _v2_104.x;
                    xb_3[22 + 1] = _v2_104.y;
                }
                {
                    float2 _v2_105 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 12ll);
                    xa_3[24] = _v2_105.x;
                    xa_3[24 + 1] = _v2_105.y;
                }
                {
                    float2 _v2_106 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 12ll);
                    xb_3[24] = _v2_106.x;
                    xb_3[24 + 1] = _v2_106.y;
                }
                {
                    float2 _v2_107 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 13ll);
                    xa_3[26] = _v2_107.x;
                    xa_3[26 + 1] = _v2_107.y;
                }
                {
                    float2 _v2_108 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 13ll);
                    xb_3[26] = _v2_108.x;
                    xb_3[26 + 1] = _v2_108.y;
                }
                {
                    float2 _v2_109 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 14ll);
                    xa_3[28] = _v2_109.x;
                    xa_3[28 + 1] = _v2_109.y;
                }
                {
                    float2 _v2_110 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 14ll);
                    xb_3[28] = _v2_110.x;
                    xb_3[28 + 1] = _v2_110.y;
                }
                {
                    float2 _v2_111 = *reinterpret_cast<const float2*>(dA_part + row_3 + step_3 * 15ll);
                    xa_3[30] = _v2_111.x;
                    xa_3[30 + 1] = _v2_111.y;
                }
                {
                    float2 _v2_112 = *reinterpret_cast<const float2*>(db_part + row_3 + step_3 * 15ll);
                    xb_3[30] = _v2_112.x;
                    xb_3[30 + 1] = _v2_112.y;
                }
                float t_148;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_148) : "f"(xa_3[0]), "f"(xa_3[16]));
                xa_3[0] = t_148;
                float t_149;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_149) : "f"(xa_3[1]), "f"(xa_3[17]));
                xa_3[1] = t_149;
                float t_150;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_150) : "f"(xa_3[2]), "f"(xa_3[18]));
                xa_3[2] = t_150;
                float t_151;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_151) : "f"(xa_3[3]), "f"(xa_3[19]));
                xa_3[3] = t_151;
                float t_152;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_152) : "f"(xa_3[4]), "f"(xa_3[20]));
                xa_3[4] = t_152;
                float t_153;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_153) : "f"(xa_3[5]), "f"(xa_3[21]));
                xa_3[5] = t_153;
                float t_154;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_154) : "f"(xa_3[6]), "f"(xa_3[22]));
                xa_3[6] = t_154;
                float t_155;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_155) : "f"(xa_3[7]), "f"(xa_3[23]));
                xa_3[7] = t_155;
                float t_156;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_156) : "f"(xa_3[8]), "f"(xa_3[24]));
                xa_3[8] = t_156;
                float t_157;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_157) : "f"(xa_3[9]), "f"(xa_3[25]));
                xa_3[9] = t_157;
                float t_158;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_158) : "f"(xa_3[10]), "f"(xa_3[26]));
                xa_3[10] = t_158;
                float t_159;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_159) : "f"(xa_3[11]), "f"(xa_3[27]));
                xa_3[11] = t_159;
                float t_160;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_160) : "f"(xa_3[12]), "f"(xa_3[28]));
                xa_3[12] = t_160;
                float t_161;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_161) : "f"(xa_3[13]), "f"(xa_3[29]));
                xa_3[13] = t_161;
                float t_162;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_162) : "f"(xa_3[14]), "f"(xa_3[30]));
                xa_3[14] = t_162;
                float t_163;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_163) : "f"(xa_3[15]), "f"(xa_3[31]));
                xa_3[15] = t_163;
                float t_164;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_164) : "f"(xa_3[0]), "f"(xa_3[8]));
                xa_3[0] = t_164;
                float t_165;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_165) : "f"(xa_3[1]), "f"(xa_3[9]));
                xa_3[1] = t_165;
                float t_166;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_166) : "f"(xa_3[2]), "f"(xa_3[10]));
                xa_3[2] = t_166;
                float t_167;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_167) : "f"(xa_3[3]), "f"(xa_3[11]));
                xa_3[3] = t_167;
                float t_168;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_168) : "f"(xa_3[4]), "f"(xa_3[12]));
                xa_3[4] = t_168;
                float t_169;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_169) : "f"(xa_3[5]), "f"(xa_3[13]));
                xa_3[5] = t_169;
                float t_170;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_170) : "f"(xa_3[6]), "f"(xa_3[14]));
                xa_3[6] = t_170;
                float t_171;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_171) : "f"(xa_3[7]), "f"(xa_3[15]));
                xa_3[7] = t_171;
                float t_172;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_172) : "f"(xa_3[0]), "f"(xa_3[4]));
                xa_3[0] = t_172;
                float t_173;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_173) : "f"(xa_3[1]), "f"(xa_3[5]));
                xa_3[1] = t_173;
                float t_174;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_174) : "f"(xa_3[2]), "f"(xa_3[6]));
                xa_3[2] = t_174;
                float t_175;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_175) : "f"(xa_3[3]), "f"(xa_3[7]));
                xa_3[3] = t_175;
                float t_176;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_176) : "f"(xa_3[0]), "f"(xa_3[2]));
                xa_3[0] = t_176;
                float t_177;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_177) : "f"(xa_3[1]), "f"(xa_3[3]));
                xa_3[1] = t_177;
                float t_178;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_178) : "f"(xb_3[0]), "f"(xb_3[16]));
                xb_3[0] = t_178;
                float t_179;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_179) : "f"(xb_3[1]), "f"(xb_3[17]));
                xb_3[1] = t_179;
                float t_180;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_180) : "f"(xb_3[2]), "f"(xb_3[18]));
                xb_3[2] = t_180;
                float t_181;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_181) : "f"(xb_3[3]), "f"(xb_3[19]));
                xb_3[3] = t_181;
                float t_182;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_182) : "f"(xb_3[4]), "f"(xb_3[20]));
                xb_3[4] = t_182;
                float t_183;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_183) : "f"(xb_3[5]), "f"(xb_3[21]));
                xb_3[5] = t_183;
                float t_184;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_184) : "f"(xb_3[6]), "f"(xb_3[22]));
                xb_3[6] = t_184;
                float t_185;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_185) : "f"(xb_3[7]), "f"(xb_3[23]));
                xb_3[7] = t_185;
                float t_186;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_186) : "f"(xb_3[8]), "f"(xb_3[24]));
                xb_3[8] = t_186;
                float t_187;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_187) : "f"(xb_3[9]), "f"(xb_3[25]));
                xb_3[9] = t_187;
                float t_188;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_188) : "f"(xb_3[10]), "f"(xb_3[26]));
                xb_3[10] = t_188;
                float t_189;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_189) : "f"(xb_3[11]), "f"(xb_3[27]));
                xb_3[11] = t_189;
                float t_190;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_190) : "f"(xb_3[12]), "f"(xb_3[28]));
                xb_3[12] = t_190;
                float t_191;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_191) : "f"(xb_3[13]), "f"(xb_3[29]));
                xb_3[13] = t_191;
                float t_192;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_192) : "f"(xb_3[14]), "f"(xb_3[30]));
                xb_3[14] = t_192;
                float t_193;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_193) : "f"(xb_3[15]), "f"(xb_3[31]));
                xb_3[15] = t_193;
                float t_194;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_194) : "f"(xb_3[0]), "f"(xb_3[8]));
                xb_3[0] = t_194;
                float t_195;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_195) : "f"(xb_3[1]), "f"(xb_3[9]));
                xb_3[1] = t_195;
                float t_196;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_196) : "f"(xb_3[2]), "f"(xb_3[10]));
                xb_3[2] = t_196;
                float t_197;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_197) : "f"(xb_3[3]), "f"(xb_3[11]));
                xb_3[3] = t_197;
                float t_198;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_198) : "f"(xb_3[4]), "f"(xb_3[12]));
                xb_3[4] = t_198;
                float t_199;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_199) : "f"(xb_3[5]), "f"(xb_3[13]));
                xb_3[5] = t_199;
                float t_200;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_200) : "f"(xb_3[6]), "f"(xb_3[14]));
                xb_3[6] = t_200;
                float t_201;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_201) : "f"(xb_3[7]), "f"(xb_3[15]));
                xb_3[7] = t_201;
                float t_202;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_202) : "f"(xb_3[0]), "f"(xb_3[4]));
                xb_3[0] = t_202;
                float t_203;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_203) : "f"(xb_3[1]), "f"(xb_3[5]));
                xb_3[1] = t_203;
                float t_204;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_204) : "f"(xb_3[2]), "f"(xb_3[6]));
                xb_3[2] = t_204;
                float t_205;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_205) : "f"(xb_3[3]), "f"(xb_3[7]));
                xb_3[3] = t_205;
                float t_206;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_206) : "f"(xb_3[0]), "f"(xb_3[2]));
                xb_3[0] = t_206;
                float t_207;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_207) : "f"(xb_3[1]), "f"(xb_3[3]));
                xb_3[1] = t_207;
                sums[0] = xa_3[0];
                sums[1] = xa_3[1];
                sums[2] = xb_3[0];
                sums[3] = xb_3[1];
            } else if (s * 128 + j + 64 >= nc) {
                float xa_4[16];
                float xb_4[16];
                long long row_4 = (((long long)(s * 128 + j) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long step_4 = (long long)num_heads * 2048ll;
                long long end_2 = (((long long)(nc - 1) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long at_32 = ((s * 128 + j < nc) ? row_4 : end_2);
                {
                    float2 _v2_113 = *reinterpret_cast<const float2*>(dA_part + at_32);
                    xa_4[0] = _v2_113.x;
                    xa_4[0 + 1] = _v2_113.y;
                }
                {
                    float2 _v2_114 = *reinterpret_cast<const float2*>(db_part + at_32);
                    xb_4[0] = _v2_114.x;
                    xb_4[0 + 1] = _v2_114.y;
                }
                long long at_33 = ((s * 128 + j + 8 < nc) ? row_4 + step_4 : end_2);
                {
                    float2 _v2_115 = *reinterpret_cast<const float2*>(dA_part + at_33);
                    xa_4[2] = _v2_115.x;
                    xa_4[2 + 1] = _v2_115.y;
                }
                {
                    float2 _v2_116 = *reinterpret_cast<const float2*>(db_part + at_33);
                    xb_4[2] = _v2_116.x;
                    xb_4[2 + 1] = _v2_116.y;
                }
                long long at_34 = ((s * 128 + j + 16 < nc) ? row_4 + step_4 * 2ll : end_2);
                {
                    float2 _v2_117 = *reinterpret_cast<const float2*>(dA_part + at_34);
                    xa_4[4] = _v2_117.x;
                    xa_4[4 + 1] = _v2_117.y;
                }
                {
                    float2 _v2_118 = *reinterpret_cast<const float2*>(db_part + at_34);
                    xb_4[4] = _v2_118.x;
                    xb_4[4 + 1] = _v2_118.y;
                }
                long long at_35 = ((s * 128 + j + 24 < nc) ? row_4 + step_4 * 3ll : end_2);
                {
                    float2 _v2_119 = *reinterpret_cast<const float2*>(dA_part + at_35);
                    xa_4[6] = _v2_119.x;
                    xa_4[6 + 1] = _v2_119.y;
                }
                {
                    float2 _v2_120 = *reinterpret_cast<const float2*>(db_part + at_35);
                    xb_4[6] = _v2_120.x;
                    xb_4[6 + 1] = _v2_120.y;
                }
                long long at_36 = ((s * 128 + j + 32 < nc) ? row_4 + step_4 * 4ll : end_2);
                {
                    float2 _v2_121 = *reinterpret_cast<const float2*>(dA_part + at_36);
                    xa_4[8] = _v2_121.x;
                    xa_4[8 + 1] = _v2_121.y;
                }
                {
                    float2 _v2_122 = *reinterpret_cast<const float2*>(db_part + at_36);
                    xb_4[8] = _v2_122.x;
                    xb_4[8 + 1] = _v2_122.y;
                }
                long long at_37 = ((s * 128 + j + 40 < nc) ? row_4 + step_4 * 5ll : end_2);
                {
                    float2 _v2_123 = *reinterpret_cast<const float2*>(dA_part + at_37);
                    xa_4[10] = _v2_123.x;
                    xa_4[10 + 1] = _v2_123.y;
                }
                {
                    float2 _v2_124 = *reinterpret_cast<const float2*>(db_part + at_37);
                    xb_4[10] = _v2_124.x;
                    xb_4[10 + 1] = _v2_124.y;
                }
                long long at_38 = ((s * 128 + j + 48 < nc) ? row_4 + step_4 * 6ll : end_2);
                {
                    float2 _v2_125 = *reinterpret_cast<const float2*>(dA_part + at_38);
                    xa_4[12] = _v2_125.x;
                    xa_4[12 + 1] = _v2_125.y;
                }
                {
                    float2 _v2_126 = *reinterpret_cast<const float2*>(db_part + at_38);
                    xb_4[12] = _v2_126.x;
                    xb_4[12 + 1] = _v2_126.y;
                }
                long long at_39 = ((s * 128 + j + 56 < nc) ? row_4 + step_4 * 7ll : end_2);
                {
                    float2 _v2_127 = *reinterpret_cast<const float2*>(dA_part + at_39);
                    xa_4[14] = _v2_127.x;
                    xa_4[14 + 1] = _v2_127.y;
                }
                {
                    float2 _v2_128 = *reinterpret_cast<const float2*>(db_part + at_39);
                    xb_4[14] = _v2_128.x;
                    xb_4[14 + 1] = _v2_128.y;
                }
                xa_4[0] = ((s * 128 + j < nc) ? xa_4[0] : 0.0f);
                xb_4[0] = ((s * 128 + j < nc) ? xb_4[0] : 0.0f);
                xa_4[1] = ((s * 128 + j < nc) ? xa_4[1] : 0.0f);
                xb_4[1] = ((s * 128 + j < nc) ? xb_4[1] : 0.0f);
                xa_4[2] = ((s * 128 + j + 8 < nc) ? xa_4[2] : 0.0f);
                xb_4[2] = ((s * 128 + j + 8 < nc) ? xb_4[2] : 0.0f);
                xa_4[3] = ((s * 128 + j + 8 < nc) ? xa_4[3] : 0.0f);
                xb_4[3] = ((s * 128 + j + 8 < nc) ? xb_4[3] : 0.0f);
                xa_4[4] = ((s * 128 + j + 16 < nc) ? xa_4[4] : 0.0f);
                xb_4[4] = ((s * 128 + j + 16 < nc) ? xb_4[4] : 0.0f);
                xa_4[5] = ((s * 128 + j + 16 < nc) ? xa_4[5] : 0.0f);
                xb_4[5] = ((s * 128 + j + 16 < nc) ? xb_4[5] : 0.0f);
                xa_4[6] = ((s * 128 + j + 24 < nc) ? xa_4[6] : 0.0f);
                xb_4[6] = ((s * 128 + j + 24 < nc) ? xb_4[6] : 0.0f);
                xa_4[7] = ((s * 128 + j + 24 < nc) ? xa_4[7] : 0.0f);
                xb_4[7] = ((s * 128 + j + 24 < nc) ? xb_4[7] : 0.0f);
                xa_4[8] = ((s * 128 + j + 32 < nc) ? xa_4[8] : 0.0f);
                xb_4[8] = ((s * 128 + j + 32 < nc) ? xb_4[8] : 0.0f);
                xa_4[9] = ((s * 128 + j + 32 < nc) ? xa_4[9] : 0.0f);
                xb_4[9] = ((s * 128 + j + 32 < nc) ? xb_4[9] : 0.0f);
                xa_4[10] = ((s * 128 + j + 40 < nc) ? xa_4[10] : 0.0f);
                xb_4[10] = ((s * 128 + j + 40 < nc) ? xb_4[10] : 0.0f);
                xa_4[11] = ((s * 128 + j + 40 < nc) ? xa_4[11] : 0.0f);
                xb_4[11] = ((s * 128 + j + 40 < nc) ? xb_4[11] : 0.0f);
                xa_4[12] = ((s * 128 + j + 48 < nc) ? xa_4[12] : 0.0f);
                xb_4[12] = ((s * 128 + j + 48 < nc) ? xb_4[12] : 0.0f);
                xa_4[13] = ((s * 128 + j + 48 < nc) ? xa_4[13] : 0.0f);
                xb_4[13] = ((s * 128 + j + 48 < nc) ? xb_4[13] : 0.0f);
                xa_4[14] = ((s * 128 + j + 56 < nc) ? xa_4[14] : 0.0f);
                xb_4[14] = ((s * 128 + j + 56 < nc) ? xb_4[14] : 0.0f);
                xa_4[15] = ((s * 128 + j + 56 < nc) ? xa_4[15] : 0.0f);
                xb_4[15] = ((s * 128 + j + 56 < nc) ? xb_4[15] : 0.0f);
                float t0_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_32) : "f"(xa_4[0]), "f"(0.0f));
                xa_4[0] = t0_32;
                float t0_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_33) : "f"(xa_4[1]), "f"(0.0f));
                xa_4[1] = t0_33;
                float t0_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_34) : "f"(xa_4[2]), "f"(0.0f));
                xa_4[2] = t0_34;
                float t0_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_35) : "f"(xa_4[3]), "f"(0.0f));
                xa_4[3] = t0_35;
                float t0_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_36) : "f"(xa_4[4]), "f"(0.0f));
                xa_4[4] = t0_36;
                float t0_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_37) : "f"(xa_4[5]), "f"(0.0f));
                xa_4[5] = t0_37;
                float t0_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_38) : "f"(xa_4[6]), "f"(0.0f));
                xa_4[6] = t0_38;
                float t0_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_39) : "f"(xa_4[7]), "f"(0.0f));
                xa_4[7] = t0_39;
                float t0_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_40) : "f"(xa_4[8]), "f"(0.0f));
                xa_4[8] = t0_40;
                float t0_41;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_41) : "f"(xa_4[9]), "f"(0.0f));
                xa_4[9] = t0_41;
                float t0_42;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_42) : "f"(xa_4[10]), "f"(0.0f));
                xa_4[10] = t0_42;
                float t0_43;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_43) : "f"(xa_4[11]), "f"(0.0f));
                xa_4[11] = t0_43;
                float t0_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_44) : "f"(xa_4[12]), "f"(0.0f));
                xa_4[12] = t0_44;
                float t0_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_45) : "f"(xa_4[13]), "f"(0.0f));
                xa_4[13] = t0_45;
                float t0_46;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_46) : "f"(xa_4[14]), "f"(0.0f));
                xa_4[14] = t0_46;
                float t0_47;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_47) : "f"(xa_4[15]), "f"(0.0f));
                xa_4[15] = t0_47;
                float t_208;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_208) : "f"(xa_4[0]), "f"(xa_4[8]));
                xa_4[0] = t_208;
                float t_209;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_209) : "f"(xa_4[1]), "f"(xa_4[9]));
                xa_4[1] = t_209;
                float t_210;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_210) : "f"(xa_4[2]), "f"(xa_4[10]));
                xa_4[2] = t_210;
                float t_211;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_211) : "f"(xa_4[3]), "f"(xa_4[11]));
                xa_4[3] = t_211;
                float t_212;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_212) : "f"(xa_4[4]), "f"(xa_4[12]));
                xa_4[4] = t_212;
                float t_213;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_213) : "f"(xa_4[5]), "f"(xa_4[13]));
                xa_4[5] = t_213;
                float t_214;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_214) : "f"(xa_4[6]), "f"(xa_4[14]));
                xa_4[6] = t_214;
                float t_215;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_215) : "f"(xa_4[7]), "f"(xa_4[15]));
                xa_4[7] = t_215;
                float t_216;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_216) : "f"(xa_4[0]), "f"(xa_4[4]));
                xa_4[0] = t_216;
                float t_217;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_217) : "f"(xa_4[1]), "f"(xa_4[5]));
                xa_4[1] = t_217;
                float t_218;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_218) : "f"(xa_4[2]), "f"(xa_4[6]));
                xa_4[2] = t_218;
                float t_219;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_219) : "f"(xa_4[3]), "f"(xa_4[7]));
                xa_4[3] = t_219;
                float t_220;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_220) : "f"(xa_4[0]), "f"(xa_4[2]));
                xa_4[0] = t_220;
                float t_221;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_221) : "f"(xa_4[1]), "f"(xa_4[3]));
                xa_4[1] = t_221;
                float t0_48;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_48) : "f"(xb_4[0]), "f"(0.0f));
                xb_4[0] = t0_48;
                float t0_49;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_49) : "f"(xb_4[1]), "f"(0.0f));
                xb_4[1] = t0_49;
                float t0_50;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_50) : "f"(xb_4[2]), "f"(0.0f));
                xb_4[2] = t0_50;
                float t0_51;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_51) : "f"(xb_4[3]), "f"(0.0f));
                xb_4[3] = t0_51;
                float t0_52;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_52) : "f"(xb_4[4]), "f"(0.0f));
                xb_4[4] = t0_52;
                float t0_53;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_53) : "f"(xb_4[5]), "f"(0.0f));
                xb_4[5] = t0_53;
                float t0_54;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_54) : "f"(xb_4[6]), "f"(0.0f));
                xb_4[6] = t0_54;
                float t0_55;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_55) : "f"(xb_4[7]), "f"(0.0f));
                xb_4[7] = t0_55;
                float t0_56;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_56) : "f"(xb_4[8]), "f"(0.0f));
                xb_4[8] = t0_56;
                float t0_57;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_57) : "f"(xb_4[9]), "f"(0.0f));
                xb_4[9] = t0_57;
                float t0_58;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_58) : "f"(xb_4[10]), "f"(0.0f));
                xb_4[10] = t0_58;
                float t0_59;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_59) : "f"(xb_4[11]), "f"(0.0f));
                xb_4[11] = t0_59;
                float t0_60;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_60) : "f"(xb_4[12]), "f"(0.0f));
                xb_4[12] = t0_60;
                float t0_61;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_61) : "f"(xb_4[13]), "f"(0.0f));
                xb_4[13] = t0_61;
                float t0_62;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_62) : "f"(xb_4[14]), "f"(0.0f));
                xb_4[14] = t0_62;
                float t0_63;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t0_63) : "f"(xb_4[15]), "f"(0.0f));
                xb_4[15] = t0_63;
                float t_222;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_222) : "f"(xb_4[0]), "f"(xb_4[8]));
                xb_4[0] = t_222;
                float t_223;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_223) : "f"(xb_4[1]), "f"(xb_4[9]));
                xb_4[1] = t_223;
                float t_224;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_224) : "f"(xb_4[2]), "f"(xb_4[10]));
                xb_4[2] = t_224;
                float t_225;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_225) : "f"(xb_4[3]), "f"(xb_4[11]));
                xb_4[3] = t_225;
                float t_226;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_226) : "f"(xb_4[4]), "f"(xb_4[12]));
                xb_4[4] = t_226;
                float t_227;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_227) : "f"(xb_4[5]), "f"(xb_4[13]));
                xb_4[5] = t_227;
                float t_228;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_228) : "f"(xb_4[6]), "f"(xb_4[14]));
                xb_4[6] = t_228;
                float t_229;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_229) : "f"(xb_4[7]), "f"(xb_4[15]));
                xb_4[7] = t_229;
                float t_230;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_230) : "f"(xb_4[0]), "f"(xb_4[4]));
                xb_4[0] = t_230;
                float t_231;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_231) : "f"(xb_4[1]), "f"(xb_4[5]));
                xb_4[1] = t_231;
                float t_232;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_232) : "f"(xb_4[2]), "f"(xb_4[6]));
                xb_4[2] = t_232;
                float t_233;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_233) : "f"(xb_4[3]), "f"(xb_4[7]));
                xb_4[3] = t_233;
                float t_234;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_234) : "f"(xb_4[0]), "f"(xb_4[2]));
                xb_4[0] = t_234;
                float t_235;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_235) : "f"(xb_4[1]), "f"(xb_4[3]));
                xb_4[1] = t_235;
                sums[0] = xa_4[0];
                sums[1] = xa_4[1];
                sums[2] = xb_4[0];
                sums[3] = xb_4[1];
            } else {
                float xa_5[32];
                float xb_5[32];
                long long row_5 = (((long long)(s * 128 + j) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long step_5 = (long long)num_heads * 2048ll;
                long long end_3 = (((long long)(nc - 1) * (long long)num_heads + (long long)head) * 128ll + (long long)d) * 2ll;
                long long at_40 = ((s * 128 + j < nc) ? row_5 : end_3);
                {
                    float2 _v2_129 = *reinterpret_cast<const float2*>(dA_part + at_40);
                    xa_5[0] = _v2_129.x;
                    xa_5[0 + 1] = _v2_129.y;
                }
                {
                    float2 _v2_130 = *reinterpret_cast<const float2*>(db_part + at_40);
                    xb_5[0] = _v2_130.x;
                    xb_5[0 + 1] = _v2_130.y;
                }
                long long at_41 = ((s * 128 + j + 8 < nc) ? row_5 + step_5 : end_3);
                {
                    float2 _v2_131 = *reinterpret_cast<const float2*>(dA_part + at_41);
                    xa_5[2] = _v2_131.x;
                    xa_5[2 + 1] = _v2_131.y;
                }
                {
                    float2 _v2_132 = *reinterpret_cast<const float2*>(db_part + at_41);
                    xb_5[2] = _v2_132.x;
                    xb_5[2 + 1] = _v2_132.y;
                }
                long long at_42 = ((s * 128 + j + 16 < nc) ? row_5 + step_5 * 2ll : end_3);
                {
                    float2 _v2_133 = *reinterpret_cast<const float2*>(dA_part + at_42);
                    xa_5[4] = _v2_133.x;
                    xa_5[4 + 1] = _v2_133.y;
                }
                {
                    float2 _v2_134 = *reinterpret_cast<const float2*>(db_part + at_42);
                    xb_5[4] = _v2_134.x;
                    xb_5[4 + 1] = _v2_134.y;
                }
                long long at_43 = ((s * 128 + j + 24 < nc) ? row_5 + step_5 * 3ll : end_3);
                {
                    float2 _v2_135 = *reinterpret_cast<const float2*>(dA_part + at_43);
                    xa_5[6] = _v2_135.x;
                    xa_5[6 + 1] = _v2_135.y;
                }
                {
                    float2 _v2_136 = *reinterpret_cast<const float2*>(db_part + at_43);
                    xb_5[6] = _v2_136.x;
                    xb_5[6 + 1] = _v2_136.y;
                }
                long long at_44 = ((s * 128 + j + 32 < nc) ? row_5 + step_5 * 4ll : end_3);
                {
                    float2 _v2_137 = *reinterpret_cast<const float2*>(dA_part + at_44);
                    xa_5[8] = _v2_137.x;
                    xa_5[8 + 1] = _v2_137.y;
                }
                {
                    float2 _v2_138 = *reinterpret_cast<const float2*>(db_part + at_44);
                    xb_5[8] = _v2_138.x;
                    xb_5[8 + 1] = _v2_138.y;
                }
                long long at_45 = ((s * 128 + j + 40 < nc) ? row_5 + step_5 * 5ll : end_3);
                {
                    float2 _v2_139 = *reinterpret_cast<const float2*>(dA_part + at_45);
                    xa_5[10] = _v2_139.x;
                    xa_5[10 + 1] = _v2_139.y;
                }
                {
                    float2 _v2_140 = *reinterpret_cast<const float2*>(db_part + at_45);
                    xb_5[10] = _v2_140.x;
                    xb_5[10 + 1] = _v2_140.y;
                }
                long long at_46 = ((s * 128 + j + 48 < nc) ? row_5 + step_5 * 6ll : end_3);
                {
                    float2 _v2_141 = *reinterpret_cast<const float2*>(dA_part + at_46);
                    xa_5[12] = _v2_141.x;
                    xa_5[12 + 1] = _v2_141.y;
                }
                {
                    float2 _v2_142 = *reinterpret_cast<const float2*>(db_part + at_46);
                    xb_5[12] = _v2_142.x;
                    xb_5[12 + 1] = _v2_142.y;
                }
                long long at_47 = ((s * 128 + j + 56 < nc) ? row_5 + step_5 * 7ll : end_3);
                {
                    float2 _v2_143 = *reinterpret_cast<const float2*>(dA_part + at_47);
                    xa_5[14] = _v2_143.x;
                    xa_5[14 + 1] = _v2_143.y;
                }
                {
                    float2 _v2_144 = *reinterpret_cast<const float2*>(db_part + at_47);
                    xb_5[14] = _v2_144.x;
                    xb_5[14 + 1] = _v2_144.y;
                }
                long long at_48 = ((s * 128 + j + 64 < nc) ? row_5 + step_5 * 8ll : end_3);
                {
                    float2 _v2_145 = *reinterpret_cast<const float2*>(dA_part + at_48);
                    xa_5[16] = _v2_145.x;
                    xa_5[16 + 1] = _v2_145.y;
                }
                {
                    float2 _v2_146 = *reinterpret_cast<const float2*>(db_part + at_48);
                    xb_5[16] = _v2_146.x;
                    xb_5[16 + 1] = _v2_146.y;
                }
                long long at_49 = ((s * 128 + j + 72 < nc) ? row_5 + step_5 * 9ll : end_3);
                {
                    float2 _v2_147 = *reinterpret_cast<const float2*>(dA_part + at_49);
                    xa_5[18] = _v2_147.x;
                    xa_5[18 + 1] = _v2_147.y;
                }
                {
                    float2 _v2_148 = *reinterpret_cast<const float2*>(db_part + at_49);
                    xb_5[18] = _v2_148.x;
                    xb_5[18 + 1] = _v2_148.y;
                }
                long long at_50 = ((s * 128 + j + 80 < nc) ? row_5 + step_5 * 10ll : end_3);
                {
                    float2 _v2_149 = *reinterpret_cast<const float2*>(dA_part + at_50);
                    xa_5[20] = _v2_149.x;
                    xa_5[20 + 1] = _v2_149.y;
                }
                {
                    float2 _v2_150 = *reinterpret_cast<const float2*>(db_part + at_50);
                    xb_5[20] = _v2_150.x;
                    xb_5[20 + 1] = _v2_150.y;
                }
                long long at_51 = ((s * 128 + j + 88 < nc) ? row_5 + step_5 * 11ll : end_3);
                {
                    float2 _v2_151 = *reinterpret_cast<const float2*>(dA_part + at_51);
                    xa_5[22] = _v2_151.x;
                    xa_5[22 + 1] = _v2_151.y;
                }
                {
                    float2 _v2_152 = *reinterpret_cast<const float2*>(db_part + at_51);
                    xb_5[22] = _v2_152.x;
                    xb_5[22 + 1] = _v2_152.y;
                }
                long long at_52 = ((s * 128 + j + 96 < nc) ? row_5 + step_5 * 12ll : end_3);
                {
                    float2 _v2_153 = *reinterpret_cast<const float2*>(dA_part + at_52);
                    xa_5[24] = _v2_153.x;
                    xa_5[24 + 1] = _v2_153.y;
                }
                {
                    float2 _v2_154 = *reinterpret_cast<const float2*>(db_part + at_52);
                    xb_5[24] = _v2_154.x;
                    xb_5[24 + 1] = _v2_154.y;
                }
                long long at_53 = ((s * 128 + j + 104 < nc) ? row_5 + step_5 * 13ll : end_3);
                {
                    float2 _v2_155 = *reinterpret_cast<const float2*>(dA_part + at_53);
                    xa_5[26] = _v2_155.x;
                    xa_5[26 + 1] = _v2_155.y;
                }
                {
                    float2 _v2_156 = *reinterpret_cast<const float2*>(db_part + at_53);
                    xb_5[26] = _v2_156.x;
                    xb_5[26 + 1] = _v2_156.y;
                }
                long long at_54 = ((s * 128 + j + 112 < nc) ? row_5 + step_5 * 14ll : end_3);
                {
                    float2 _v2_157 = *reinterpret_cast<const float2*>(dA_part + at_54);
                    xa_5[28] = _v2_157.x;
                    xa_5[28 + 1] = _v2_157.y;
                }
                {
                    float2 _v2_158 = *reinterpret_cast<const float2*>(db_part + at_54);
                    xb_5[28] = _v2_158.x;
                    xb_5[28 + 1] = _v2_158.y;
                }
                long long at_55 = ((s * 128 + j + 120 < nc) ? row_5 + step_5 * 15ll : end_3);
                {
                    float2 _v2_159 = *reinterpret_cast<const float2*>(dA_part + at_55);
                    xa_5[30] = _v2_159.x;
                    xa_5[30 + 1] = _v2_159.y;
                }
                {
                    float2 _v2_160 = *reinterpret_cast<const float2*>(db_part + at_55);
                    xb_5[30] = _v2_160.x;
                    xb_5[30 + 1] = _v2_160.y;
                }
                xa_5[0] = ((s * 128 + j < nc) ? xa_5[0] : 0.0f);
                xb_5[0] = ((s * 128 + j < nc) ? xb_5[0] : 0.0f);
                xa_5[1] = ((s * 128 + j < nc) ? xa_5[1] : 0.0f);
                xb_5[1] = ((s * 128 + j < nc) ? xb_5[1] : 0.0f);
                xa_5[2] = ((s * 128 + j + 8 < nc) ? xa_5[2] : 0.0f);
                xb_5[2] = ((s * 128 + j + 8 < nc) ? xb_5[2] : 0.0f);
                xa_5[3] = ((s * 128 + j + 8 < nc) ? xa_5[3] : 0.0f);
                xb_5[3] = ((s * 128 + j + 8 < nc) ? xb_5[3] : 0.0f);
                xa_5[4] = ((s * 128 + j + 16 < nc) ? xa_5[4] : 0.0f);
                xb_5[4] = ((s * 128 + j + 16 < nc) ? xb_5[4] : 0.0f);
                xa_5[5] = ((s * 128 + j + 16 < nc) ? xa_5[5] : 0.0f);
                xb_5[5] = ((s * 128 + j + 16 < nc) ? xb_5[5] : 0.0f);
                xa_5[6] = ((s * 128 + j + 24 < nc) ? xa_5[6] : 0.0f);
                xb_5[6] = ((s * 128 + j + 24 < nc) ? xb_5[6] : 0.0f);
                xa_5[7] = ((s * 128 + j + 24 < nc) ? xa_5[7] : 0.0f);
                xb_5[7] = ((s * 128 + j + 24 < nc) ? xb_5[7] : 0.0f);
                xa_5[8] = ((s * 128 + j + 32 < nc) ? xa_5[8] : 0.0f);
                xb_5[8] = ((s * 128 + j + 32 < nc) ? xb_5[8] : 0.0f);
                xa_5[9] = ((s * 128 + j + 32 < nc) ? xa_5[9] : 0.0f);
                xb_5[9] = ((s * 128 + j + 32 < nc) ? xb_5[9] : 0.0f);
                xa_5[10] = ((s * 128 + j + 40 < nc) ? xa_5[10] : 0.0f);
                xb_5[10] = ((s * 128 + j + 40 < nc) ? xb_5[10] : 0.0f);
                xa_5[11] = ((s * 128 + j + 40 < nc) ? xa_5[11] : 0.0f);
                xb_5[11] = ((s * 128 + j + 40 < nc) ? xb_5[11] : 0.0f);
                xa_5[12] = ((s * 128 + j + 48 < nc) ? xa_5[12] : 0.0f);
                xb_5[12] = ((s * 128 + j + 48 < nc) ? xb_5[12] : 0.0f);
                xa_5[13] = ((s * 128 + j + 48 < nc) ? xa_5[13] : 0.0f);
                xb_5[13] = ((s * 128 + j + 48 < nc) ? xb_5[13] : 0.0f);
                xa_5[14] = ((s * 128 + j + 56 < nc) ? xa_5[14] : 0.0f);
                xb_5[14] = ((s * 128 + j + 56 < nc) ? xb_5[14] : 0.0f);
                xa_5[15] = ((s * 128 + j + 56 < nc) ? xa_5[15] : 0.0f);
                xb_5[15] = ((s * 128 + j + 56 < nc) ? xb_5[15] : 0.0f);
                xa_5[16] = ((s * 128 + j + 64 < nc) ? xa_5[16] : 0.0f);
                xb_5[16] = ((s * 128 + j + 64 < nc) ? xb_5[16] : 0.0f);
                xa_5[17] = ((s * 128 + j + 64 < nc) ? xa_5[17] : 0.0f);
                xb_5[17] = ((s * 128 + j + 64 < nc) ? xb_5[17] : 0.0f);
                xa_5[18] = ((s * 128 + j + 72 < nc) ? xa_5[18] : 0.0f);
                xb_5[18] = ((s * 128 + j + 72 < nc) ? xb_5[18] : 0.0f);
                xa_5[19] = ((s * 128 + j + 72 < nc) ? xa_5[19] : 0.0f);
                xb_5[19] = ((s * 128 + j + 72 < nc) ? xb_5[19] : 0.0f);
                xa_5[20] = ((s * 128 + j + 80 < nc) ? xa_5[20] : 0.0f);
                xb_5[20] = ((s * 128 + j + 80 < nc) ? xb_5[20] : 0.0f);
                xa_5[21] = ((s * 128 + j + 80 < nc) ? xa_5[21] : 0.0f);
                xb_5[21] = ((s * 128 + j + 80 < nc) ? xb_5[21] : 0.0f);
                xa_5[22] = ((s * 128 + j + 88 < nc) ? xa_5[22] : 0.0f);
                xb_5[22] = ((s * 128 + j + 88 < nc) ? xb_5[22] : 0.0f);
                xa_5[23] = ((s * 128 + j + 88 < nc) ? xa_5[23] : 0.0f);
                xb_5[23] = ((s * 128 + j + 88 < nc) ? xb_5[23] : 0.0f);
                xa_5[24] = ((s * 128 + j + 96 < nc) ? xa_5[24] : 0.0f);
                xb_5[24] = ((s * 128 + j + 96 < nc) ? xb_5[24] : 0.0f);
                xa_5[25] = ((s * 128 + j + 96 < nc) ? xa_5[25] : 0.0f);
                xb_5[25] = ((s * 128 + j + 96 < nc) ? xb_5[25] : 0.0f);
                xa_5[26] = ((s * 128 + j + 104 < nc) ? xa_5[26] : 0.0f);
                xb_5[26] = ((s * 128 + j + 104 < nc) ? xb_5[26] : 0.0f);
                xa_5[27] = ((s * 128 + j + 104 < nc) ? xa_5[27] : 0.0f);
                xb_5[27] = ((s * 128 + j + 104 < nc) ? xb_5[27] : 0.0f);
                xa_5[28] = ((s * 128 + j + 112 < nc) ? xa_5[28] : 0.0f);
                xb_5[28] = ((s * 128 + j + 112 < nc) ? xb_5[28] : 0.0f);
                xa_5[29] = ((s * 128 + j + 112 < nc) ? xa_5[29] : 0.0f);
                xb_5[29] = ((s * 128 + j + 112 < nc) ? xb_5[29] : 0.0f);
                xa_5[30] = ((s * 128 + j + 120 < nc) ? xa_5[30] : 0.0f);
                xb_5[30] = ((s * 128 + j + 120 < nc) ? xb_5[30] : 0.0f);
                xa_5[31] = ((s * 128 + j + 120 < nc) ? xa_5[31] : 0.0f);
                xb_5[31] = ((s * 128 + j + 120 < nc) ? xb_5[31] : 0.0f);
                float t_236;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_236) : "f"(xa_5[0]), "f"(xa_5[16]));
                xa_5[0] = t_236;
                float t_237;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_237) : "f"(xa_5[1]), "f"(xa_5[17]));
                xa_5[1] = t_237;
                float t_238;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_238) : "f"(xa_5[2]), "f"(xa_5[18]));
                xa_5[2] = t_238;
                float t_239;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_239) : "f"(xa_5[3]), "f"(xa_5[19]));
                xa_5[3] = t_239;
                float t_240;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_240) : "f"(xa_5[4]), "f"(xa_5[20]));
                xa_5[4] = t_240;
                float t_241;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_241) : "f"(xa_5[5]), "f"(xa_5[21]));
                xa_5[5] = t_241;
                float t_242;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_242) : "f"(xa_5[6]), "f"(xa_5[22]));
                xa_5[6] = t_242;
                float t_243;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_243) : "f"(xa_5[7]), "f"(xa_5[23]));
                xa_5[7] = t_243;
                float t_244;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_244) : "f"(xa_5[8]), "f"(xa_5[24]));
                xa_5[8] = t_244;
                float t_245;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_245) : "f"(xa_5[9]), "f"(xa_5[25]));
                xa_5[9] = t_245;
                float t_246;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_246) : "f"(xa_5[10]), "f"(xa_5[26]));
                xa_5[10] = t_246;
                float t_247;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_247) : "f"(xa_5[11]), "f"(xa_5[27]));
                xa_5[11] = t_247;
                float t_248;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_248) : "f"(xa_5[12]), "f"(xa_5[28]));
                xa_5[12] = t_248;
                float t_249;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_249) : "f"(xa_5[13]), "f"(xa_5[29]));
                xa_5[13] = t_249;
                float t_250;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_250) : "f"(xa_5[14]), "f"(xa_5[30]));
                xa_5[14] = t_250;
                float t_251;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_251) : "f"(xa_5[15]), "f"(xa_5[31]));
                xa_5[15] = t_251;
                float t_252;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_252) : "f"(xa_5[0]), "f"(xa_5[8]));
                xa_5[0] = t_252;
                float t_253;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_253) : "f"(xa_5[1]), "f"(xa_5[9]));
                xa_5[1] = t_253;
                float t_254;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_254) : "f"(xa_5[2]), "f"(xa_5[10]));
                xa_5[2] = t_254;
                float t_255;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_255) : "f"(xa_5[3]), "f"(xa_5[11]));
                xa_5[3] = t_255;
                float t_256;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_256) : "f"(xa_5[4]), "f"(xa_5[12]));
                xa_5[4] = t_256;
                float t_257;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_257) : "f"(xa_5[5]), "f"(xa_5[13]));
                xa_5[5] = t_257;
                float t_258;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_258) : "f"(xa_5[6]), "f"(xa_5[14]));
                xa_5[6] = t_258;
                float t_259;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_259) : "f"(xa_5[7]), "f"(xa_5[15]));
                xa_5[7] = t_259;
                float t_260;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_260) : "f"(xa_5[0]), "f"(xa_5[4]));
                xa_5[0] = t_260;
                float t_261;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_261) : "f"(xa_5[1]), "f"(xa_5[5]));
                xa_5[1] = t_261;
                float t_262;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_262) : "f"(xa_5[2]), "f"(xa_5[6]));
                xa_5[2] = t_262;
                float t_263;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_263) : "f"(xa_5[3]), "f"(xa_5[7]));
                xa_5[3] = t_263;
                float t_264;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_264) : "f"(xa_5[0]), "f"(xa_5[2]));
                xa_5[0] = t_264;
                float t_265;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_265) : "f"(xa_5[1]), "f"(xa_5[3]));
                xa_5[1] = t_265;
                float t_266;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_266) : "f"(xb_5[0]), "f"(xb_5[16]));
                xb_5[0] = t_266;
                float t_267;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_267) : "f"(xb_5[1]), "f"(xb_5[17]));
                xb_5[1] = t_267;
                float t_268;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_268) : "f"(xb_5[2]), "f"(xb_5[18]));
                xb_5[2] = t_268;
                float t_269;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_269) : "f"(xb_5[3]), "f"(xb_5[19]));
                xb_5[3] = t_269;
                float t_270;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_270) : "f"(xb_5[4]), "f"(xb_5[20]));
                xb_5[4] = t_270;
                float t_271;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_271) : "f"(xb_5[5]), "f"(xb_5[21]));
                xb_5[5] = t_271;
                float t_272;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_272) : "f"(xb_5[6]), "f"(xb_5[22]));
                xb_5[6] = t_272;
                float t_273;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_273) : "f"(xb_5[7]), "f"(xb_5[23]));
                xb_5[7] = t_273;
                float t_274;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_274) : "f"(xb_5[8]), "f"(xb_5[24]));
                xb_5[8] = t_274;
                float t_275;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_275) : "f"(xb_5[9]), "f"(xb_5[25]));
                xb_5[9] = t_275;
                float t_276;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_276) : "f"(xb_5[10]), "f"(xb_5[26]));
                xb_5[10] = t_276;
                float t_277;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_277) : "f"(xb_5[11]), "f"(xb_5[27]));
                xb_5[11] = t_277;
                float t_278;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_278) : "f"(xb_5[12]), "f"(xb_5[28]));
                xb_5[12] = t_278;
                float t_279;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_279) : "f"(xb_5[13]), "f"(xb_5[29]));
                xb_5[13] = t_279;
                float t_280;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_280) : "f"(xb_5[14]), "f"(xb_5[30]));
                xb_5[14] = t_280;
                float t_281;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_281) : "f"(xb_5[15]), "f"(xb_5[31]));
                xb_5[15] = t_281;
                float t_282;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_282) : "f"(xb_5[0]), "f"(xb_5[8]));
                xb_5[0] = t_282;
                float t_283;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_283) : "f"(xb_5[1]), "f"(xb_5[9]));
                xb_5[1] = t_283;
                float t_284;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_284) : "f"(xb_5[2]), "f"(xb_5[10]));
                xb_5[2] = t_284;
                float t_285;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_285) : "f"(xb_5[3]), "f"(xb_5[11]));
                xb_5[3] = t_285;
                float t_286;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_286) : "f"(xb_5[4]), "f"(xb_5[12]));
                xb_5[4] = t_286;
                float t_287;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_287) : "f"(xb_5[5]), "f"(xb_5[13]));
                xb_5[5] = t_287;
                float t_288;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_288) : "f"(xb_5[6]), "f"(xb_5[14]));
                xb_5[6] = t_288;
                float t_289;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_289) : "f"(xb_5[7]), "f"(xb_5[15]));
                xb_5[7] = t_289;
                float t_290;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_290) : "f"(xb_5[0]), "f"(xb_5[4]));
                xb_5[0] = t_290;
                float t_291;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_291) : "f"(xb_5[1]), "f"(xb_5[5]));
                xb_5[1] = t_291;
                float t_292;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_292) : "f"(xb_5[2]), "f"(xb_5[6]));
                xb_5[2] = t_292;
                float t_293;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_293) : "f"(xb_5[3]), "f"(xb_5[7]));
                xb_5[3] = t_293;
                float t_294;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_294) : "f"(xb_5[0]), "f"(xb_5[2]));
                xb_5[0] = t_294;
                float t_295;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(t_295) : "f"(xb_5[1]), "f"(xb_5[3]));
                xb_5[1] = t_295;
                sums[0] = xa_5[0];
                sums[1] = xa_5[1];
                sums[2] = xb_5[0];
                sums[3] = xb_5[1];
            }
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(slot_addr), "f"(sums[0]), "f"(sums[1]), "f"(sums[2]), "f"(sums[3]) : "memory");
            __syncthreads();
            if (j == 0) {
                float t_296[32];
                t_296[0] = sums[0];
                t_296[1] = sums[1];
                t_296[2] = sums[2];
                t_296[3] = sums[3];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t_296[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(4) + 3]))
                    : "r"(part_addr + (unsigned int)((128 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t_296[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(8) + 3]))
                    : "r"(part_addr + (unsigned int)((256 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t_296[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(12) + 3]))
                    : "r"(part_addr + (unsigned int)((384 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t_296[16])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(16) + 3]))
                    : "r"(part_addr + (unsigned int)((512 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t_296[20])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(20) + 3]))
                    : "r"(part_addr + (unsigned int)((640 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t_296[24])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(24) + 3]))
                    : "r"(part_addr + (unsigned int)((768 + d) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&t_296[28])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_296[(28) + 3]))
                    : "r"(part_addr + (unsigned int)((896 + d) * 16)));
                float u_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_28) : "f"(t_296[0]), "f"(t_296[16]));
                t_296[0] = u_28;
                float u_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_29) : "f"(t_296[1]), "f"(t_296[17]));
                t_296[1] = u_29;
                float u_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_30) : "f"(t_296[2]), "f"(t_296[18]));
                t_296[2] = u_30;
                float u_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_31) : "f"(t_296[3]), "f"(t_296[19]));
                t_296[3] = u_31;
                float u_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_32) : "f"(t_296[4]), "f"(t_296[20]));
                t_296[4] = u_32;
                float u_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_33) : "f"(t_296[5]), "f"(t_296[21]));
                t_296[5] = u_33;
                float u_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_34) : "f"(t_296[6]), "f"(t_296[22]));
                t_296[6] = u_34;
                float u_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_35) : "f"(t_296[7]), "f"(t_296[23]));
                t_296[7] = u_35;
                float u_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_36) : "f"(t_296[8]), "f"(t_296[24]));
                t_296[8] = u_36;
                float u_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_37) : "f"(t_296[9]), "f"(t_296[25]));
                t_296[9] = u_37;
                float u_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_38) : "f"(t_296[10]), "f"(t_296[26]));
                t_296[10] = u_38;
                float u_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_39) : "f"(t_296[11]), "f"(t_296[27]));
                t_296[11] = u_39;
                float u_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_40) : "f"(t_296[12]), "f"(t_296[28]));
                t_296[12] = u_40;
                float u_41;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_41) : "f"(t_296[13]), "f"(t_296[29]));
                t_296[13] = u_41;
                float u_42;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_42) : "f"(t_296[14]), "f"(t_296[30]));
                t_296[14] = u_42;
                float u_43;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_43) : "f"(t_296[15]), "f"(t_296[31]));
                t_296[15] = u_43;
                float u_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_44) : "f"(t_296[0]), "f"(t_296[8]));
                t_296[0] = u_44;
                float u_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_45) : "f"(t_296[1]), "f"(t_296[9]));
                t_296[1] = u_45;
                float u_46;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_46) : "f"(t_296[2]), "f"(t_296[10]));
                t_296[2] = u_46;
                float u_47;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_47) : "f"(t_296[3]), "f"(t_296[11]));
                t_296[3] = u_47;
                float u_48;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_48) : "f"(t_296[4]), "f"(t_296[12]));
                t_296[4] = u_48;
                float u_49;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_49) : "f"(t_296[5]), "f"(t_296[13]));
                t_296[5] = u_49;
                float u_50;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_50) : "f"(t_296[6]), "f"(t_296[14]));
                t_296[6] = u_50;
                float u_51;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_51) : "f"(t_296[7]), "f"(t_296[15]));
                t_296[7] = u_51;
                float u_52;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_52) : "f"(t_296[0]), "f"(t_296[4]));
                t_296[0] = u_52;
                float u_53;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_53) : "f"(t_296[1]), "f"(t_296[5]));
                t_296[1] = u_53;
                float u_54;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_54) : "f"(t_296[2]), "f"(t_296[6]));
                t_296[2] = u_54;
                float u_55;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(u_55) : "f"(t_296[3]), "f"(t_296[7]));
                t_296[3] = u_55;
                float b01_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(b01_1) : "f"(t_296[2]), "f"(t_296[3]));
                sums[0] = t_296[0];
                float _add_161;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(b01_1), "f"(t_296[1]));
                sums[1] = _add_161;
                long long q = (((long long)head * (long long)seg_cap + (long long)s) * 128ll + (long long)d) * 2ll;
                part2[q] = sums[0];
                part2[q + 1ll] = sums[1];
            }
            __syncthreads();
        }
        asm volatile("fence.release.gpu;" ::: "memory");
        __syncthreads();
        if (tid == 0) {
            unsigned int old = atomicAdd(&ticket[head], 1);
            flag[0] = old;
            if ((int)old == g - 1) {
                asm volatile("fence.acquire.gpu;" ::: "memory");
                ticket[head] = 0;
            }
        }
        __syncthreads();
        last = flag[0];
        if (last == (unsigned int)(g - 1)) {
            asm volatile("fence.acquire.gpu;" ::: "memory");
            int groups = nseg >> 3;
            float levels[16];
            levels[0] = 0.0f;
            levels[1] = 0.0f;
            levels[2] = 0.0f;
            levels[3] = 0.0f;
            levels[4] = 0.0f;
            levels[5] = 0.0f;
            levels[6] = 0.0f;
            levels[7] = 0.0f;
            levels[8] = 0.0f;
            levels[9] = 0.0f;
            levels[10] = 0.0f;
            levels[11] = 0.0f;
            levels[12] = 0.0f;
            levels[13] = 0.0f;
            levels[14] = 0.0f;
            levels[15] = 0.0f;
            top[0] = 0.0f;
            top[1] = 0.0f;
            float rest[14];
            if (j == 0 && groups * 8 < nseg) {
                int sr_0 = ((groups * 8) < (nseg - 1) ? (groups * 8) : (nseg - 1));
                long long qr_0 = (((long long)head * (long long)seg_cap + (long long)sr_0) * 128ll + (long long)d) * 2ll;
                {
                    float2 _v2_162 = *reinterpret_cast<const float2*>(part2 + qr_0);
                    rest[0] = _v2_162.x;
                    rest[0 + 1] = _v2_162.y;
                }
                int sr_1 = ((groups * 8 + 1) < (nseg - 1) ? (groups * 8 + 1) : (nseg - 1));
                long long qr_1 = (((long long)head * (long long)seg_cap + (long long)sr_1) * 128ll + (long long)d) * 2ll;
                {
                    float2 _v2_163 = *reinterpret_cast<const float2*>(part2 + qr_1);
                    rest[2] = _v2_163.x;
                    rest[2 + 1] = _v2_163.y;
                }
                int sr_2 = ((groups * 8 + 2) < (nseg - 1) ? (groups * 8 + 2) : (nseg - 1));
                long long qr_2 = (((long long)head * (long long)seg_cap + (long long)sr_2) * 128ll + (long long)d) * 2ll;
                {
                    float2 _v2_164 = *reinterpret_cast<const float2*>(part2 + qr_2);
                    rest[4] = _v2_164.x;
                    rest[4 + 1] = _v2_164.y;
                }
                int sr_3 = ((groups * 8 + 3) < (nseg - 1) ? (groups * 8 + 3) : (nseg - 1));
                long long qr_3 = (((long long)head * (long long)seg_cap + (long long)sr_3) * 128ll + (long long)d) * 2ll;
                {
                    float2 _v2_165 = *reinterpret_cast<const float2*>(part2 + qr_3);
                    rest[6] = _v2_165.x;
                    rest[6 + 1] = _v2_165.y;
                }
                int sr_4 = ((groups * 8 + 4) < (nseg - 1) ? (groups * 8 + 4) : (nseg - 1));
                long long qr_4 = (((long long)head * (long long)seg_cap + (long long)sr_4) * 128ll + (long long)d) * 2ll;
                {
                    float2 _v2_166 = *reinterpret_cast<const float2*>(part2 + qr_4);
                    rest[8] = _v2_166.x;
                    rest[8 + 1] = _v2_166.y;
                }
                int sr_5 = ((groups * 8 + 5) < (nseg - 1) ? (groups * 8 + 5) : (nseg - 1));
                long long qr_5 = (((long long)head * (long long)seg_cap + (long long)sr_5) * 128ll + (long long)d) * 2ll;
                {
                    float2 _v2_167 = *reinterpret_cast<const float2*>(part2 + qr_5);
                    rest[10] = _v2_167.x;
                    rest[10 + 1] = _v2_167.y;
                }
                int sr_6 = ((groups * 8 + 6) < (nseg - 1) ? (groups * 8 + 6) : (nseg - 1));
                long long qr_6 = (((long long)head * (long long)seg_cap + (long long)sr_6) * 128ll + (long long)d) * 2ll;
                {
                    float2 _v2_168 = *reinterpret_cast<const float2*>(part2 + qr_6);
                    rest[12] = _v2_168.x;
                    rest[12 + 1] = _v2_168.y;
                }
            }
            float gy[2];
            for (int r = 0; r < groups; r += 8) {
                if (r + j < groups) {
                    float x[16];
                    long long q_0 = (((long long)head * (long long)seg_cap + (long long)((r + j) * 8)) * 128ll + (long long)d) * 2ll;
                    {
                        float2 _v2_169 = *reinterpret_cast<const float2*>(part2 + q_0);
                        x[0] = _v2_169.x;
                        x[0 + 1] = _v2_169.y;
                    }
                    long long q_1 = (((long long)head * (long long)seg_cap + (long long)((r + j) * 8 + 1)) * 128ll + (long long)d) * 2ll;
                    {
                        float2 _v2_170 = *reinterpret_cast<const float2*>(part2 + q_1);
                        x[2] = _v2_170.x;
                        x[2 + 1] = _v2_170.y;
                    }
                    long long q_2 = (((long long)head * (long long)seg_cap + (long long)((r + j) * 8 + 2)) * 128ll + (long long)d) * 2ll;
                    {
                        float2 _v2_171 = *reinterpret_cast<const float2*>(part2 + q_2);
                        x[4] = _v2_171.x;
                        x[4 + 1] = _v2_171.y;
                    }
                    long long q_3 = (((long long)head * (long long)seg_cap + (long long)((r + j) * 8 + 3)) * 128ll + (long long)d) * 2ll;
                    {
                        float2 _v2_172 = *reinterpret_cast<const float2*>(part2 + q_3);
                        x[6] = _v2_172.x;
                        x[6 + 1] = _v2_172.y;
                    }
                    long long q_4 = (((long long)head * (long long)seg_cap + (long long)((r + j) * 8 + 4)) * 128ll + (long long)d) * 2ll;
                    {
                        float2 _v2_173 = *reinterpret_cast<const float2*>(part2 + q_4);
                        x[8] = _v2_173.x;
                        x[8 + 1] = _v2_173.y;
                    }
                    long long q_5 = (((long long)head * (long long)seg_cap + (long long)((r + j) * 8 + 5)) * 128ll + (long long)d) * 2ll;
                    {
                        float2 _v2_174 = *reinterpret_cast<const float2*>(part2 + q_5);
                        x[10] = _v2_174.x;
                        x[10 + 1] = _v2_174.y;
                    }
                    long long q_6 = (((long long)head * (long long)seg_cap + (long long)((r + j) * 8 + 6)) * 128ll + (long long)d) * 2ll;
                    {
                        float2 _v2_175 = *reinterpret_cast<const float2*>(part2 + q_6);
                        x[12] = _v2_175.x;
                        x[12 + 1] = _v2_175.y;
                    }
                    long long q_7 = (((long long)head * (long long)seg_cap + (long long)((r + j) * 8 + 7)) * 128ll + (long long)d) * 2ll;
                    {
                        float2 _v2_176 = *reinterpret_cast<const float2*>(part2 + q_7);
                        x[14] = _v2_176.x;
                        x[14 + 1] = _v2_176.y;
                    }
                    float t_296_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_296_1) : "f"(x[0]), "f"(x[2]));
                    x[0] = t_296_1;
                    float t_297;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_297) : "f"(x[1]), "f"(x[3]));
                    x[1] = t_297;
                    float t_298;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_298) : "f"(x[4]), "f"(x[6]));
                    x[4] = t_298;
                    float t_299;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_299) : "f"(x[5]), "f"(x[7]));
                    x[5] = t_299;
                    float t_300;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_300) : "f"(x[8]), "f"(x[10]));
                    x[8] = t_300;
                    float t_301;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_301) : "f"(x[9]), "f"(x[11]));
                    x[9] = t_301;
                    float t_302;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_302) : "f"(x[12]), "f"(x[14]));
                    x[12] = t_302;
                    float t_303;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_303) : "f"(x[13]), "f"(x[15]));
                    x[13] = t_303;
                    float t_304;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_304) : "f"(x[0]), "f"(x[4]));
                    x[0] = t_304;
                    float t_305;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_305) : "f"(x[1]), "f"(x[5]));
                    x[1] = t_305;
                    float t_306;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_306) : "f"(x[8]), "f"(x[12]));
                    x[8] = t_306;
                    float t_307;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_307) : "f"(x[9]), "f"(x[13]));
                    x[9] = t_307;
                    float t_308;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_308) : "f"(x[0]), "f"(x[8]));
                    x[0] = t_308;
                    float t_309;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_309) : "f"(x[1]), "f"(x[9]));
                    x[1] = t_309;
                    gy[0] = x[0];
                    gy[1] = x[1];
                    asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(part_addr + (unsigned int)((j * 128 + d) * 8)), "f"(gy[0]), "f"(gy[1]) : "memory");
                }
                __syncthreads();
                if (j == 0) {
                    if (r < groups) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gy[0])), "=r"(*reinterpret_cast<uint32_t*>(&gy[(0) + 1]))
                            : "r"(part_addr + (unsigned int)(d * 8)));
                        int low_8 = r * 8 + 8 - 1 & 15;
                        float merged_16;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_16) : "f"(levels[6]), "f"(gy[0]));
                        gy[0] = ((low_8 == 15) ? merged_16 : gy[0]);
                        float merged_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_17) : "f"(levels[7]), "f"(gy[1]));
                        gy[1] = ((low_8 == 15) ? merged_17 : gy[1]);
                        levels[6] = ((low_8 == 15) ? 0.0f : levels[6]);
                        levels[6] = ((low_8 == 7) ? gy[0] : levels[6]);
                        levels[7] = ((low_8 == 15) ? 0.0f : levels[7]);
                        levels[7] = ((low_8 == 7) ? gy[1] : levels[7]);
                        int low_9 = r * 8 + 8 - 1 & 31;
                        float merged_18;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_18) : "f"(levels[8]), "f"(gy[0]));
                        gy[0] = ((low_9 == 31) ? merged_18 : gy[0]);
                        float merged_19;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_19) : "f"(levels[9]), "f"(gy[1]));
                        gy[1] = ((low_9 == 31) ? merged_19 : gy[1]);
                        levels[8] = ((low_9 == 31) ? 0.0f : levels[8]);
                        levels[8] = ((low_9 == 15) ? gy[0] : levels[8]);
                        levels[9] = ((low_9 == 31) ? 0.0f : levels[9]);
                        levels[9] = ((low_9 == 15) ? gy[1] : levels[9]);
                        int low_10 = r * 8 + 8 - 1 & 63;
                        float merged_20;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_20) : "f"(levels[10]), "f"(gy[0]));
                        gy[0] = ((low_10 == 63) ? merged_20 : gy[0]);
                        float merged_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_21) : "f"(levels[11]), "f"(gy[1]));
                        gy[1] = ((low_10 == 63) ? merged_21 : gy[1]);
                        levels[10] = ((low_10 == 63) ? 0.0f : levels[10]);
                        levels[10] = ((low_10 == 31) ? gy[0] : levels[10]);
                        levels[11] = ((low_10 == 63) ? 0.0f : levels[11]);
                        levels[11] = ((low_10 == 31) ? gy[1] : levels[11]);
                        int low_11 = r * 8 + 8 - 1 & 127;
                        float merged_22;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_22) : "f"(levels[12]), "f"(gy[0]));
                        gy[0] = ((low_11 == 127) ? merged_22 : gy[0]);
                        float merged_23;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_23) : "f"(levels[13]), "f"(gy[1]));
                        gy[1] = ((low_11 == 127) ? merged_23 : gy[1]);
                        levels[12] = ((low_11 == 127) ? 0.0f : levels[12]);
                        levels[12] = ((low_11 == 63) ? gy[0] : levels[12]);
                        levels[13] = ((low_11 == 127) ? 0.0f : levels[13]);
                        levels[13] = ((low_11 == 63) ? gy[1] : levels[13]);
                        int low_12 = r * 8 + 8 - 1 & 255;
                        float merged_24;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_24) : "f"(levels[14]), "f"(gy[0]));
                        gy[0] = ((low_12 == 255) ? merged_24 : gy[0]);
                        float merged_25;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_25) : "f"(levels[15]), "f"(gy[1]));
                        gy[1] = ((low_12 == 255) ? merged_25 : gy[1]);
                        levels[14] = ((low_12 == 255) ? 0.0f : levels[14]);
                        levels[14] = ((low_12 == 127) ? gy[0] : levels[14]);
                        levels[15] = ((low_12 == 255) ? 0.0f : levels[15]);
                        levels[15] = ((low_12 == 127) ? gy[1] : levels[15]);
                        float carried_2;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_2) : "f"(top[0]), "f"(gy[0]));
                        top[0] = (((r * 8 + 8 - 1 & 255) == 255) ? carried_2 : top[0]);
                        float carried_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_3) : "f"(top[1]), "f"(gy[1]));
                        top[1] = (((r * 8 + 8 - 1 & 255) == 255) ? carried_3 : top[1]);
                    }
                    if (r + 1 < groups) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gy[0])), "=r"(*reinterpret_cast<uint32_t*>(&gy[(0) + 1]))
                            : "r"(part_addr + (unsigned int)((128 + d) * 8)));
                        int low_13 = (r + 1) * 8 + 8 - 1 & 15;
                        float merged_26;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_26) : "f"(levels[6]), "f"(gy[0]));
                        gy[0] = ((low_13 == 15) ? merged_26 : gy[0]);
                        float merged_27;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_27) : "f"(levels[7]), "f"(gy[1]));
                        gy[1] = ((low_13 == 15) ? merged_27 : gy[1]);
                        levels[6] = ((low_13 == 15) ? 0.0f : levels[6]);
                        levels[6] = ((low_13 == 7) ? gy[0] : levels[6]);
                        levels[7] = ((low_13 == 15) ? 0.0f : levels[7]);
                        levels[7] = ((low_13 == 7) ? gy[1] : levels[7]);
                        int low_14 = (r + 1) * 8 + 8 - 1 & 31;
                        float merged_28;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_28) : "f"(levels[8]), "f"(gy[0]));
                        gy[0] = ((low_14 == 31) ? merged_28 : gy[0]);
                        float merged_29;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_29) : "f"(levels[9]), "f"(gy[1]));
                        gy[1] = ((low_14 == 31) ? merged_29 : gy[1]);
                        levels[8] = ((low_14 == 31) ? 0.0f : levels[8]);
                        levels[8] = ((low_14 == 15) ? gy[0] : levels[8]);
                        levels[9] = ((low_14 == 31) ? 0.0f : levels[9]);
                        levels[9] = ((low_14 == 15) ? gy[1] : levels[9]);
                        int low_15 = (r + 1) * 8 + 8 - 1 & 63;
                        float merged_30;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_30) : "f"(levels[10]), "f"(gy[0]));
                        gy[0] = ((low_15 == 63) ? merged_30 : gy[0]);
                        float merged_31;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_31) : "f"(levels[11]), "f"(gy[1]));
                        gy[1] = ((low_15 == 63) ? merged_31 : gy[1]);
                        levels[10] = ((low_15 == 63) ? 0.0f : levels[10]);
                        levels[10] = ((low_15 == 31) ? gy[0] : levels[10]);
                        levels[11] = ((low_15 == 63) ? 0.0f : levels[11]);
                        levels[11] = ((low_15 == 31) ? gy[1] : levels[11]);
                        int low_16 = (r + 1) * 8 + 8 - 1 & 127;
                        float merged_32;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_32) : "f"(levels[12]), "f"(gy[0]));
                        gy[0] = ((low_16 == 127) ? merged_32 : gy[0]);
                        float merged_33;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_33) : "f"(levels[13]), "f"(gy[1]));
                        gy[1] = ((low_16 == 127) ? merged_33 : gy[1]);
                        levels[12] = ((low_16 == 127) ? 0.0f : levels[12]);
                        levels[12] = ((low_16 == 63) ? gy[0] : levels[12]);
                        levels[13] = ((low_16 == 127) ? 0.0f : levels[13]);
                        levels[13] = ((low_16 == 63) ? gy[1] : levels[13]);
                        int low_17 = (r + 1) * 8 + 8 - 1 & 255;
                        float merged_34;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_34) : "f"(levels[14]), "f"(gy[0]));
                        gy[0] = ((low_17 == 255) ? merged_34 : gy[0]);
                        float merged_35;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_35) : "f"(levels[15]), "f"(gy[1]));
                        gy[1] = ((low_17 == 255) ? merged_35 : gy[1]);
                        levels[14] = ((low_17 == 255) ? 0.0f : levels[14]);
                        levels[14] = ((low_17 == 127) ? gy[0] : levels[14]);
                        levels[15] = ((low_17 == 255) ? 0.0f : levels[15]);
                        levels[15] = ((low_17 == 127) ? gy[1] : levels[15]);
                        float carried_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_4) : "f"(top[0]), "f"(gy[0]));
                        top[0] = ((((r + 1) * 8 + 8 - 1 & 255) == 255) ? carried_4 : top[0]);
                        float carried_5;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_5) : "f"(top[1]), "f"(gy[1]));
                        top[1] = ((((r + 1) * 8 + 8 - 1 & 255) == 255) ? carried_5 : top[1]);
                    }
                    if (r + 2 < groups) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gy[0])), "=r"(*reinterpret_cast<uint32_t*>(&gy[(0) + 1]))
                            : "r"(part_addr + (unsigned int)((256 + d) * 8)));
                        int low_18 = (r + 2) * 8 + 8 - 1 & 15;
                        float merged_36;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_36) : "f"(levels[6]), "f"(gy[0]));
                        gy[0] = ((low_18 == 15) ? merged_36 : gy[0]);
                        float merged_37;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_37) : "f"(levels[7]), "f"(gy[1]));
                        gy[1] = ((low_18 == 15) ? merged_37 : gy[1]);
                        levels[6] = ((low_18 == 15) ? 0.0f : levels[6]);
                        levels[6] = ((low_18 == 7) ? gy[0] : levels[6]);
                        levels[7] = ((low_18 == 15) ? 0.0f : levels[7]);
                        levels[7] = ((low_18 == 7) ? gy[1] : levels[7]);
                        int low_19 = (r + 2) * 8 + 8 - 1 & 31;
                        float merged_38;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_38) : "f"(levels[8]), "f"(gy[0]));
                        gy[0] = ((low_19 == 31) ? merged_38 : gy[0]);
                        float merged_39;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_39) : "f"(levels[9]), "f"(gy[1]));
                        gy[1] = ((low_19 == 31) ? merged_39 : gy[1]);
                        levels[8] = ((low_19 == 31) ? 0.0f : levels[8]);
                        levels[8] = ((low_19 == 15) ? gy[0] : levels[8]);
                        levels[9] = ((low_19 == 31) ? 0.0f : levels[9]);
                        levels[9] = ((low_19 == 15) ? gy[1] : levels[9]);
                        int low_20 = (r + 2) * 8 + 8 - 1 & 63;
                        float merged_40;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_40) : "f"(levels[10]), "f"(gy[0]));
                        gy[0] = ((low_20 == 63) ? merged_40 : gy[0]);
                        float merged_41;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_41) : "f"(levels[11]), "f"(gy[1]));
                        gy[1] = ((low_20 == 63) ? merged_41 : gy[1]);
                        levels[10] = ((low_20 == 63) ? 0.0f : levels[10]);
                        levels[10] = ((low_20 == 31) ? gy[0] : levels[10]);
                        levels[11] = ((low_20 == 63) ? 0.0f : levels[11]);
                        levels[11] = ((low_20 == 31) ? gy[1] : levels[11]);
                        int low_21 = (r + 2) * 8 + 8 - 1 & 127;
                        float merged_42;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_42) : "f"(levels[12]), "f"(gy[0]));
                        gy[0] = ((low_21 == 127) ? merged_42 : gy[0]);
                        float merged_43;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_43) : "f"(levels[13]), "f"(gy[1]));
                        gy[1] = ((low_21 == 127) ? merged_43 : gy[1]);
                        levels[12] = ((low_21 == 127) ? 0.0f : levels[12]);
                        levels[12] = ((low_21 == 63) ? gy[0] : levels[12]);
                        levels[13] = ((low_21 == 127) ? 0.0f : levels[13]);
                        levels[13] = ((low_21 == 63) ? gy[1] : levels[13]);
                        int low_22 = (r + 2) * 8 + 8 - 1 & 255;
                        float merged_44;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_44) : "f"(levels[14]), "f"(gy[0]));
                        gy[0] = ((low_22 == 255) ? merged_44 : gy[0]);
                        float merged_45;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_45) : "f"(levels[15]), "f"(gy[1]));
                        gy[1] = ((low_22 == 255) ? merged_45 : gy[1]);
                        levels[14] = ((low_22 == 255) ? 0.0f : levels[14]);
                        levels[14] = ((low_22 == 127) ? gy[0] : levels[14]);
                        levels[15] = ((low_22 == 255) ? 0.0f : levels[15]);
                        levels[15] = ((low_22 == 127) ? gy[1] : levels[15]);
                        float carried_6;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_6) : "f"(top[0]), "f"(gy[0]));
                        top[0] = ((((r + 2) * 8 + 8 - 1 & 255) == 255) ? carried_6 : top[0]);
                        float carried_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_7) : "f"(top[1]), "f"(gy[1]));
                        top[1] = ((((r + 2) * 8 + 8 - 1 & 255) == 255) ? carried_7 : top[1]);
                    }
                    if (r + 3 < groups) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gy[0])), "=r"(*reinterpret_cast<uint32_t*>(&gy[(0) + 1]))
                            : "r"(part_addr + (unsigned int)((384 + d) * 8)));
                        int low_23 = (r + 3) * 8 + 8 - 1 & 15;
                        float merged_46;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_46) : "f"(levels[6]), "f"(gy[0]));
                        gy[0] = ((low_23 == 15) ? merged_46 : gy[0]);
                        float merged_47;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_47) : "f"(levels[7]), "f"(gy[1]));
                        gy[1] = ((low_23 == 15) ? merged_47 : gy[1]);
                        levels[6] = ((low_23 == 15) ? 0.0f : levels[6]);
                        levels[6] = ((low_23 == 7) ? gy[0] : levels[6]);
                        levels[7] = ((low_23 == 15) ? 0.0f : levels[7]);
                        levels[7] = ((low_23 == 7) ? gy[1] : levels[7]);
                        int low_24 = (r + 3) * 8 + 8 - 1 & 31;
                        float merged_48;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_48) : "f"(levels[8]), "f"(gy[0]));
                        gy[0] = ((low_24 == 31) ? merged_48 : gy[0]);
                        float merged_49;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_49) : "f"(levels[9]), "f"(gy[1]));
                        gy[1] = ((low_24 == 31) ? merged_49 : gy[1]);
                        levels[8] = ((low_24 == 31) ? 0.0f : levels[8]);
                        levels[8] = ((low_24 == 15) ? gy[0] : levels[8]);
                        levels[9] = ((low_24 == 31) ? 0.0f : levels[9]);
                        levels[9] = ((low_24 == 15) ? gy[1] : levels[9]);
                        int low_25 = (r + 3) * 8 + 8 - 1 & 63;
                        float merged_50;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_50) : "f"(levels[10]), "f"(gy[0]));
                        gy[0] = ((low_25 == 63) ? merged_50 : gy[0]);
                        float merged_51;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_51) : "f"(levels[11]), "f"(gy[1]));
                        gy[1] = ((low_25 == 63) ? merged_51 : gy[1]);
                        levels[10] = ((low_25 == 63) ? 0.0f : levels[10]);
                        levels[10] = ((low_25 == 31) ? gy[0] : levels[10]);
                        levels[11] = ((low_25 == 63) ? 0.0f : levels[11]);
                        levels[11] = ((low_25 == 31) ? gy[1] : levels[11]);
                        int low_26 = (r + 3) * 8 + 8 - 1 & 127;
                        float merged_52;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_52) : "f"(levels[12]), "f"(gy[0]));
                        gy[0] = ((low_26 == 127) ? merged_52 : gy[0]);
                        float merged_53;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_53) : "f"(levels[13]), "f"(gy[1]));
                        gy[1] = ((low_26 == 127) ? merged_53 : gy[1]);
                        levels[12] = ((low_26 == 127) ? 0.0f : levels[12]);
                        levels[12] = ((low_26 == 63) ? gy[0] : levels[12]);
                        levels[13] = ((low_26 == 127) ? 0.0f : levels[13]);
                        levels[13] = ((low_26 == 63) ? gy[1] : levels[13]);
                        int low_27 = (r + 3) * 8 + 8 - 1 & 255;
                        float merged_54;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_54) : "f"(levels[14]), "f"(gy[0]));
                        gy[0] = ((low_27 == 255) ? merged_54 : gy[0]);
                        float merged_55;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_55) : "f"(levels[15]), "f"(gy[1]));
                        gy[1] = ((low_27 == 255) ? merged_55 : gy[1]);
                        levels[14] = ((low_27 == 255) ? 0.0f : levels[14]);
                        levels[14] = ((low_27 == 127) ? gy[0] : levels[14]);
                        levels[15] = ((low_27 == 255) ? 0.0f : levels[15]);
                        levels[15] = ((low_27 == 127) ? gy[1] : levels[15]);
                        float carried_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_8) : "f"(top[0]), "f"(gy[0]));
                        top[0] = ((((r + 3) * 8 + 8 - 1 & 255) == 255) ? carried_8 : top[0]);
                        float carried_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_9) : "f"(top[1]), "f"(gy[1]));
                        top[1] = ((((r + 3) * 8 + 8 - 1 & 255) == 255) ? carried_9 : top[1]);
                    }
                    if (r + 4 < groups) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gy[0])), "=r"(*reinterpret_cast<uint32_t*>(&gy[(0) + 1]))
                            : "r"(part_addr + (unsigned int)((512 + d) * 8)));
                        int low_28 = (r + 4) * 8 + 8 - 1 & 15;
                        float merged_56;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_56) : "f"(levels[6]), "f"(gy[0]));
                        gy[0] = ((low_28 == 15) ? merged_56 : gy[0]);
                        float merged_57;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_57) : "f"(levels[7]), "f"(gy[1]));
                        gy[1] = ((low_28 == 15) ? merged_57 : gy[1]);
                        levels[6] = ((low_28 == 15) ? 0.0f : levels[6]);
                        levels[6] = ((low_28 == 7) ? gy[0] : levels[6]);
                        levels[7] = ((low_28 == 15) ? 0.0f : levels[7]);
                        levels[7] = ((low_28 == 7) ? gy[1] : levels[7]);
                        int low_29 = (r + 4) * 8 + 8 - 1 & 31;
                        float merged_58;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_58) : "f"(levels[8]), "f"(gy[0]));
                        gy[0] = ((low_29 == 31) ? merged_58 : gy[0]);
                        float merged_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_59) : "f"(levels[9]), "f"(gy[1]));
                        gy[1] = ((low_29 == 31) ? merged_59 : gy[1]);
                        levels[8] = ((low_29 == 31) ? 0.0f : levels[8]);
                        levels[8] = ((low_29 == 15) ? gy[0] : levels[8]);
                        levels[9] = ((low_29 == 31) ? 0.0f : levels[9]);
                        levels[9] = ((low_29 == 15) ? gy[1] : levels[9]);
                        int low_30 = (r + 4) * 8 + 8 - 1 & 63;
                        float merged_60;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_60) : "f"(levels[10]), "f"(gy[0]));
                        gy[0] = ((low_30 == 63) ? merged_60 : gy[0]);
                        float merged_61;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_61) : "f"(levels[11]), "f"(gy[1]));
                        gy[1] = ((low_30 == 63) ? merged_61 : gy[1]);
                        levels[10] = ((low_30 == 63) ? 0.0f : levels[10]);
                        levels[10] = ((low_30 == 31) ? gy[0] : levels[10]);
                        levels[11] = ((low_30 == 63) ? 0.0f : levels[11]);
                        levels[11] = ((low_30 == 31) ? gy[1] : levels[11]);
                        int low_31 = (r + 4) * 8 + 8 - 1 & 127;
                        float merged_62;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_62) : "f"(levels[12]), "f"(gy[0]));
                        gy[0] = ((low_31 == 127) ? merged_62 : gy[0]);
                        float merged_63;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_63) : "f"(levels[13]), "f"(gy[1]));
                        gy[1] = ((low_31 == 127) ? merged_63 : gy[1]);
                        levels[12] = ((low_31 == 127) ? 0.0f : levels[12]);
                        levels[12] = ((low_31 == 63) ? gy[0] : levels[12]);
                        levels[13] = ((low_31 == 127) ? 0.0f : levels[13]);
                        levels[13] = ((low_31 == 63) ? gy[1] : levels[13]);
                        int low_32 = (r + 4) * 8 + 8 - 1 & 255;
                        float merged_64;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_64) : "f"(levels[14]), "f"(gy[0]));
                        gy[0] = ((low_32 == 255) ? merged_64 : gy[0]);
                        float merged_65;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_65) : "f"(levels[15]), "f"(gy[1]));
                        gy[1] = ((low_32 == 255) ? merged_65 : gy[1]);
                        levels[14] = ((low_32 == 255) ? 0.0f : levels[14]);
                        levels[14] = ((low_32 == 127) ? gy[0] : levels[14]);
                        levels[15] = ((low_32 == 255) ? 0.0f : levels[15]);
                        levels[15] = ((low_32 == 127) ? gy[1] : levels[15]);
                        float carried_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_10) : "f"(top[0]), "f"(gy[0]));
                        top[0] = ((((r + 4) * 8 + 8 - 1 & 255) == 255) ? carried_10 : top[0]);
                        float carried_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_11) : "f"(top[1]), "f"(gy[1]));
                        top[1] = ((((r + 4) * 8 + 8 - 1 & 255) == 255) ? carried_11 : top[1]);
                    }
                    if (r + 5 < groups) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gy[0])), "=r"(*reinterpret_cast<uint32_t*>(&gy[(0) + 1]))
                            : "r"(part_addr + (unsigned int)((640 + d) * 8)));
                        int low_33 = (r + 5) * 8 + 8 - 1 & 15;
                        float merged_66;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_66) : "f"(levels[6]), "f"(gy[0]));
                        gy[0] = ((low_33 == 15) ? merged_66 : gy[0]);
                        float merged_67;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_67) : "f"(levels[7]), "f"(gy[1]));
                        gy[1] = ((low_33 == 15) ? merged_67 : gy[1]);
                        levels[6] = ((low_33 == 15) ? 0.0f : levels[6]);
                        levels[6] = ((low_33 == 7) ? gy[0] : levels[6]);
                        levels[7] = ((low_33 == 15) ? 0.0f : levels[7]);
                        levels[7] = ((low_33 == 7) ? gy[1] : levels[7]);
                        int low_34 = (r + 5) * 8 + 8 - 1 & 31;
                        float merged_68;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_68) : "f"(levels[8]), "f"(gy[0]));
                        gy[0] = ((low_34 == 31) ? merged_68 : gy[0]);
                        float merged_69;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_69) : "f"(levels[9]), "f"(gy[1]));
                        gy[1] = ((low_34 == 31) ? merged_69 : gy[1]);
                        levels[8] = ((low_34 == 31) ? 0.0f : levels[8]);
                        levels[8] = ((low_34 == 15) ? gy[0] : levels[8]);
                        levels[9] = ((low_34 == 31) ? 0.0f : levels[9]);
                        levels[9] = ((low_34 == 15) ? gy[1] : levels[9]);
                        int low_35 = (r + 5) * 8 + 8 - 1 & 63;
                        float merged_70;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_70) : "f"(levels[10]), "f"(gy[0]));
                        gy[0] = ((low_35 == 63) ? merged_70 : gy[0]);
                        float merged_71;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_71) : "f"(levels[11]), "f"(gy[1]));
                        gy[1] = ((low_35 == 63) ? merged_71 : gy[1]);
                        levels[10] = ((low_35 == 63) ? 0.0f : levels[10]);
                        levels[10] = ((low_35 == 31) ? gy[0] : levels[10]);
                        levels[11] = ((low_35 == 63) ? 0.0f : levels[11]);
                        levels[11] = ((low_35 == 31) ? gy[1] : levels[11]);
                        int low_36 = (r + 5) * 8 + 8 - 1 & 127;
                        float merged_72;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_72) : "f"(levels[12]), "f"(gy[0]));
                        gy[0] = ((low_36 == 127) ? merged_72 : gy[0]);
                        float merged_73;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_73) : "f"(levels[13]), "f"(gy[1]));
                        gy[1] = ((low_36 == 127) ? merged_73 : gy[1]);
                        levels[12] = ((low_36 == 127) ? 0.0f : levels[12]);
                        levels[12] = ((low_36 == 63) ? gy[0] : levels[12]);
                        levels[13] = ((low_36 == 127) ? 0.0f : levels[13]);
                        levels[13] = ((low_36 == 63) ? gy[1] : levels[13]);
                        int low_37 = (r + 5) * 8 + 8 - 1 & 255;
                        float merged_74;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_74) : "f"(levels[14]), "f"(gy[0]));
                        gy[0] = ((low_37 == 255) ? merged_74 : gy[0]);
                        float merged_75;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_75) : "f"(levels[15]), "f"(gy[1]));
                        gy[1] = ((low_37 == 255) ? merged_75 : gy[1]);
                        levels[14] = ((low_37 == 255) ? 0.0f : levels[14]);
                        levels[14] = ((low_37 == 127) ? gy[0] : levels[14]);
                        levels[15] = ((low_37 == 255) ? 0.0f : levels[15]);
                        levels[15] = ((low_37 == 127) ? gy[1] : levels[15]);
                        float carried_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_12) : "f"(top[0]), "f"(gy[0]));
                        top[0] = ((((r + 5) * 8 + 8 - 1 & 255) == 255) ? carried_12 : top[0]);
                        float carried_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_13) : "f"(top[1]), "f"(gy[1]));
                        top[1] = ((((r + 5) * 8 + 8 - 1 & 255) == 255) ? carried_13 : top[1]);
                    }
                    if (r + 6 < groups) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gy[0])), "=r"(*reinterpret_cast<uint32_t*>(&gy[(0) + 1]))
                            : "r"(part_addr + (unsigned int)((768 + d) * 8)));
                        int low_38 = (r + 6) * 8 + 8 - 1 & 15;
                        float merged_76;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_76) : "f"(levels[6]), "f"(gy[0]));
                        gy[0] = ((low_38 == 15) ? merged_76 : gy[0]);
                        float merged_77;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_77) : "f"(levels[7]), "f"(gy[1]));
                        gy[1] = ((low_38 == 15) ? merged_77 : gy[1]);
                        levels[6] = ((low_38 == 15) ? 0.0f : levels[6]);
                        levels[6] = ((low_38 == 7) ? gy[0] : levels[6]);
                        levels[7] = ((low_38 == 15) ? 0.0f : levels[7]);
                        levels[7] = ((low_38 == 7) ? gy[1] : levels[7]);
                        int low_39 = (r + 6) * 8 + 8 - 1 & 31;
                        float merged_78;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_78) : "f"(levels[8]), "f"(gy[0]));
                        gy[0] = ((low_39 == 31) ? merged_78 : gy[0]);
                        float merged_79;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_79) : "f"(levels[9]), "f"(gy[1]));
                        gy[1] = ((low_39 == 31) ? merged_79 : gy[1]);
                        levels[8] = ((low_39 == 31) ? 0.0f : levels[8]);
                        levels[8] = ((low_39 == 15) ? gy[0] : levels[8]);
                        levels[9] = ((low_39 == 31) ? 0.0f : levels[9]);
                        levels[9] = ((low_39 == 15) ? gy[1] : levels[9]);
                        int low_40 = (r + 6) * 8 + 8 - 1 & 63;
                        float merged_80;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_80) : "f"(levels[10]), "f"(gy[0]));
                        gy[0] = ((low_40 == 63) ? merged_80 : gy[0]);
                        float merged_81;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_81) : "f"(levels[11]), "f"(gy[1]));
                        gy[1] = ((low_40 == 63) ? merged_81 : gy[1]);
                        levels[10] = ((low_40 == 63) ? 0.0f : levels[10]);
                        levels[10] = ((low_40 == 31) ? gy[0] : levels[10]);
                        levels[11] = ((low_40 == 63) ? 0.0f : levels[11]);
                        levels[11] = ((low_40 == 31) ? gy[1] : levels[11]);
                        int low_41 = (r + 6) * 8 + 8 - 1 & 127;
                        float merged_82;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_82) : "f"(levels[12]), "f"(gy[0]));
                        gy[0] = ((low_41 == 127) ? merged_82 : gy[0]);
                        float merged_83;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_83) : "f"(levels[13]), "f"(gy[1]));
                        gy[1] = ((low_41 == 127) ? merged_83 : gy[1]);
                        levels[12] = ((low_41 == 127) ? 0.0f : levels[12]);
                        levels[12] = ((low_41 == 63) ? gy[0] : levels[12]);
                        levels[13] = ((low_41 == 127) ? 0.0f : levels[13]);
                        levels[13] = ((low_41 == 63) ? gy[1] : levels[13]);
                        int low_42 = (r + 6) * 8 + 8 - 1 & 255;
                        float merged_84;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_84) : "f"(levels[14]), "f"(gy[0]));
                        gy[0] = ((low_42 == 255) ? merged_84 : gy[0]);
                        float merged_85;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_85) : "f"(levels[15]), "f"(gy[1]));
                        gy[1] = ((low_42 == 255) ? merged_85 : gy[1]);
                        levels[14] = ((low_42 == 255) ? 0.0f : levels[14]);
                        levels[14] = ((low_42 == 127) ? gy[0] : levels[14]);
                        levels[15] = ((low_42 == 255) ? 0.0f : levels[15]);
                        levels[15] = ((low_42 == 127) ? gy[1] : levels[15]);
                        float carried_14;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_14) : "f"(top[0]), "f"(gy[0]));
                        top[0] = ((((r + 6) * 8 + 8 - 1 & 255) == 255) ? carried_14 : top[0]);
                        float carried_15;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_15) : "f"(top[1]), "f"(gy[1]));
                        top[1] = ((((r + 6) * 8 + 8 - 1 & 255) == 255) ? carried_15 : top[1]);
                    }
                    if (r + 7 < groups) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gy[0])), "=r"(*reinterpret_cast<uint32_t*>(&gy[(0) + 1]))
                            : "r"(part_addr + (unsigned int)((896 + d) * 8)));
                        int low_43 = (r + 7) * 8 + 8 - 1 & 15;
                        float merged_86;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_86) : "f"(levels[6]), "f"(gy[0]));
                        gy[0] = ((low_43 == 15) ? merged_86 : gy[0]);
                        float merged_87;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_87) : "f"(levels[7]), "f"(gy[1]));
                        gy[1] = ((low_43 == 15) ? merged_87 : gy[1]);
                        levels[6] = ((low_43 == 15) ? 0.0f : levels[6]);
                        levels[6] = ((low_43 == 7) ? gy[0] : levels[6]);
                        levels[7] = ((low_43 == 15) ? 0.0f : levels[7]);
                        levels[7] = ((low_43 == 7) ? gy[1] : levels[7]);
                        int low_44 = (r + 7) * 8 + 8 - 1 & 31;
                        float merged_88;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_88) : "f"(levels[8]), "f"(gy[0]));
                        gy[0] = ((low_44 == 31) ? merged_88 : gy[0]);
                        float merged_89;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_89) : "f"(levels[9]), "f"(gy[1]));
                        gy[1] = ((low_44 == 31) ? merged_89 : gy[1]);
                        levels[8] = ((low_44 == 31) ? 0.0f : levels[8]);
                        levels[8] = ((low_44 == 15) ? gy[0] : levels[8]);
                        levels[9] = ((low_44 == 31) ? 0.0f : levels[9]);
                        levels[9] = ((low_44 == 15) ? gy[1] : levels[9]);
                        int low_45 = (r + 7) * 8 + 8 - 1 & 63;
                        float merged_90;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_90) : "f"(levels[10]), "f"(gy[0]));
                        gy[0] = ((low_45 == 63) ? merged_90 : gy[0]);
                        float merged_91;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_91) : "f"(levels[11]), "f"(gy[1]));
                        gy[1] = ((low_45 == 63) ? merged_91 : gy[1]);
                        levels[10] = ((low_45 == 63) ? 0.0f : levels[10]);
                        levels[10] = ((low_45 == 31) ? gy[0] : levels[10]);
                        levels[11] = ((low_45 == 63) ? 0.0f : levels[11]);
                        levels[11] = ((low_45 == 31) ? gy[1] : levels[11]);
                        int low_46 = (r + 7) * 8 + 8 - 1 & 127;
                        float merged_92;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_92) : "f"(levels[12]), "f"(gy[0]));
                        gy[0] = ((low_46 == 127) ? merged_92 : gy[0]);
                        float merged_93;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_93) : "f"(levels[13]), "f"(gy[1]));
                        gy[1] = ((low_46 == 127) ? merged_93 : gy[1]);
                        levels[12] = ((low_46 == 127) ? 0.0f : levels[12]);
                        levels[12] = ((low_46 == 63) ? gy[0] : levels[12]);
                        levels[13] = ((low_46 == 127) ? 0.0f : levels[13]);
                        levels[13] = ((low_46 == 63) ? gy[1] : levels[13]);
                        int low_47 = (r + 7) * 8 + 8 - 1 & 255;
                        float merged_94;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_94) : "f"(levels[14]), "f"(gy[0]));
                        gy[0] = ((low_47 == 255) ? merged_94 : gy[0]);
                        float merged_95;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_95) : "f"(levels[15]), "f"(gy[1]));
                        gy[1] = ((low_47 == 255) ? merged_95 : gy[1]);
                        levels[14] = ((low_47 == 255) ? 0.0f : levels[14]);
                        levels[14] = ((low_47 == 127) ? gy[0] : levels[14]);
                        levels[15] = ((low_47 == 255) ? 0.0f : levels[15]);
                        levels[15] = ((low_47 == 127) ? gy[1] : levels[15]);
                        float carried_16;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_16) : "f"(top[0]), "f"(gy[0]));
                        top[0] = ((((r + 7) * 8 + 8 - 1 & 255) == 255) ? carried_16 : top[0]);
                        float carried_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_17) : "f"(top[1]), "f"(gy[1]));
                        top[1] = ((((r + 7) * 8 + 8 - 1 & 255) == 255) ? carried_17 : top[1]);
                    }
                }
                __syncthreads();
            }
            if (j == 0) {
                if (groups * 8 < nseg) {
                    sums[0] = rest[0];
                    sums[1] = rest[1];
                    int low_48 = groups * 8 & 1;
                    float merged_96;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_96) : "f"(levels[0]), "f"(sums[0]));
                    sums[0] = ((low_48 == 1) ? merged_96 : sums[0]);
                    float merged_97;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_97) : "f"(levels[1]), "f"(sums[1]));
                    sums[1] = ((low_48 == 1) ? merged_97 : sums[1]);
                    levels[0] = ((low_48 == 1) ? 0.0f : levels[0]);
                    levels[0] = ((low_48 == 0) ? sums[0] : levels[0]);
                    levels[1] = ((low_48 == 1) ? 0.0f : levels[1]);
                    levels[1] = ((low_48 == 0) ? sums[1] : levels[1]);
                    int low_49 = groups * 8 & 3;
                    float merged_98;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_98) : "f"(levels[2]), "f"(sums[0]));
                    sums[0] = ((low_49 == 3) ? merged_98 : sums[0]);
                    float merged_99;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_99) : "f"(levels[3]), "f"(sums[1]));
                    sums[1] = ((low_49 == 3) ? merged_99 : sums[1]);
                    levels[2] = ((low_49 == 3) ? 0.0f : levels[2]);
                    levels[2] = ((low_49 == 1) ? sums[0] : levels[2]);
                    levels[3] = ((low_49 == 3) ? 0.0f : levels[3]);
                    levels[3] = ((low_49 == 1) ? sums[1] : levels[3]);
                    int low_50 = groups * 8 & 7;
                    float merged_100;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_100) : "f"(levels[4]), "f"(sums[0]));
                    sums[0] = ((low_50 == 7) ? merged_100 : sums[0]);
                    float merged_101;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_101) : "f"(levels[5]), "f"(sums[1]));
                    sums[1] = ((low_50 == 7) ? merged_101 : sums[1]);
                    levels[4] = ((low_50 == 7) ? 0.0f : levels[4]);
                    levels[4] = ((low_50 == 3) ? sums[0] : levels[4]);
                    levels[5] = ((low_50 == 7) ? 0.0f : levels[5]);
                    levels[5] = ((low_50 == 3) ? sums[1] : levels[5]);
                    int low_51 = groups * 8 & 15;
                    float merged_102;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_102) : "f"(levels[6]), "f"(sums[0]));
                    sums[0] = ((low_51 == 15) ? merged_102 : sums[0]);
                    float merged_103;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_103) : "f"(levels[7]), "f"(sums[1]));
                    sums[1] = ((low_51 == 15) ? merged_103 : sums[1]);
                    levels[6] = ((low_51 == 15) ? 0.0f : levels[6]);
                    levels[6] = ((low_51 == 7) ? sums[0] : levels[6]);
                    levels[7] = ((low_51 == 15) ? 0.0f : levels[7]);
                    levels[7] = ((low_51 == 7) ? sums[1] : levels[7]);
                    int low_52 = groups * 8 & 31;
                    float merged_104;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_104) : "f"(levels[8]), "f"(sums[0]));
                    sums[0] = ((low_52 == 31) ? merged_104 : sums[0]);
                    float merged_105;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_105) : "f"(levels[9]), "f"(sums[1]));
                    sums[1] = ((low_52 == 31) ? merged_105 : sums[1]);
                    levels[8] = ((low_52 == 31) ? 0.0f : levels[8]);
                    levels[8] = ((low_52 == 15) ? sums[0] : levels[8]);
                    levels[9] = ((low_52 == 31) ? 0.0f : levels[9]);
                    levels[9] = ((low_52 == 15) ? sums[1] : levels[9]);
                    int low_53 = groups * 8 & 63;
                    float merged_106;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_106) : "f"(levels[10]), "f"(sums[0]));
                    sums[0] = ((low_53 == 63) ? merged_106 : sums[0]);
                    float merged_107;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_107) : "f"(levels[11]), "f"(sums[1]));
                    sums[1] = ((low_53 == 63) ? merged_107 : sums[1]);
                    levels[10] = ((low_53 == 63) ? 0.0f : levels[10]);
                    levels[10] = ((low_53 == 31) ? sums[0] : levels[10]);
                    levels[11] = ((low_53 == 63) ? 0.0f : levels[11]);
                    levels[11] = ((low_53 == 31) ? sums[1] : levels[11]);
                    int low_54 = groups * 8 & 127;
                    float merged_108;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_108) : "f"(levels[12]), "f"(sums[0]));
                    sums[0] = ((low_54 == 127) ? merged_108 : sums[0]);
                    float merged_109;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_109) : "f"(levels[13]), "f"(sums[1]));
                    sums[1] = ((low_54 == 127) ? merged_109 : sums[1]);
                    levels[12] = ((low_54 == 127) ? 0.0f : levels[12]);
                    levels[12] = ((low_54 == 63) ? sums[0] : levels[12]);
                    levels[13] = ((low_54 == 127) ? 0.0f : levels[13]);
                    levels[13] = ((low_54 == 63) ? sums[1] : levels[13]);
                    int low_55 = groups * 8 & 255;
                    float merged_110;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_110) : "f"(levels[14]), "f"(sums[0]));
                    sums[0] = ((low_55 == 255) ? merged_110 : sums[0]);
                    float merged_111;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_111) : "f"(levels[15]), "f"(sums[1]));
                    sums[1] = ((low_55 == 255) ? merged_111 : sums[1]);
                    levels[14] = ((low_55 == 255) ? 0.0f : levels[14]);
                    levels[14] = ((low_55 == 127) ? sums[0] : levels[14]);
                    levels[15] = ((low_55 == 255) ? 0.0f : levels[15]);
                    levels[15] = ((low_55 == 127) ? sums[1] : levels[15]);
                    float carried_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_18) : "f"(top[0]), "f"(sums[0]));
                    top[0] = (((groups * 8 & 255) == 255) ? carried_18 : top[0]);
                    float carried_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_19) : "f"(top[1]), "f"(sums[1]));
                    top[1] = (((groups * 8 & 255) == 255) ? carried_19 : top[1]);
                }
                if (groups * 8 + 1 < nseg) {
                    sums[0] = rest[2];
                    sums[1] = rest[3];
                    int low_56 = groups * 8 + 1 & 1;
                    float merged_112;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_112) : "f"(levels[0]), "f"(sums[0]));
                    sums[0] = ((low_56 == 1) ? merged_112 : sums[0]);
                    float merged_113;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_113) : "f"(levels[1]), "f"(sums[1]));
                    sums[1] = ((low_56 == 1) ? merged_113 : sums[1]);
                    levels[0] = ((low_56 == 1) ? 0.0f : levels[0]);
                    levels[0] = ((low_56 == 0) ? sums[0] : levels[0]);
                    levels[1] = ((low_56 == 1) ? 0.0f : levels[1]);
                    levels[1] = ((low_56 == 0) ? sums[1] : levels[1]);
                    int low_57 = groups * 8 + 1 & 3;
                    float merged_114;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_114) : "f"(levels[2]), "f"(sums[0]));
                    sums[0] = ((low_57 == 3) ? merged_114 : sums[0]);
                    float merged_115;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_115) : "f"(levels[3]), "f"(sums[1]));
                    sums[1] = ((low_57 == 3) ? merged_115 : sums[1]);
                    levels[2] = ((low_57 == 3) ? 0.0f : levels[2]);
                    levels[2] = ((low_57 == 1) ? sums[0] : levels[2]);
                    levels[3] = ((low_57 == 3) ? 0.0f : levels[3]);
                    levels[3] = ((low_57 == 1) ? sums[1] : levels[3]);
                    int low_58 = groups * 8 + 1 & 7;
                    float merged_116;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_116) : "f"(levels[4]), "f"(sums[0]));
                    sums[0] = ((low_58 == 7) ? merged_116 : sums[0]);
                    float merged_117;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_117) : "f"(levels[5]), "f"(sums[1]));
                    sums[1] = ((low_58 == 7) ? merged_117 : sums[1]);
                    levels[4] = ((low_58 == 7) ? 0.0f : levels[4]);
                    levels[4] = ((low_58 == 3) ? sums[0] : levels[4]);
                    levels[5] = ((low_58 == 7) ? 0.0f : levels[5]);
                    levels[5] = ((low_58 == 3) ? sums[1] : levels[5]);
                    int low_59 = groups * 8 + 1 & 15;
                    float merged_118;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_118) : "f"(levels[6]), "f"(sums[0]));
                    sums[0] = ((low_59 == 15) ? merged_118 : sums[0]);
                    float merged_119;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_119) : "f"(levels[7]), "f"(sums[1]));
                    sums[1] = ((low_59 == 15) ? merged_119 : sums[1]);
                    levels[6] = ((low_59 == 15) ? 0.0f : levels[6]);
                    levels[6] = ((low_59 == 7) ? sums[0] : levels[6]);
                    levels[7] = ((low_59 == 15) ? 0.0f : levels[7]);
                    levels[7] = ((low_59 == 7) ? sums[1] : levels[7]);
                    int low_60 = groups * 8 + 1 & 31;
                    float merged_120;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_120) : "f"(levels[8]), "f"(sums[0]));
                    sums[0] = ((low_60 == 31) ? merged_120 : sums[0]);
                    float merged_121;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_121) : "f"(levels[9]), "f"(sums[1]));
                    sums[1] = ((low_60 == 31) ? merged_121 : sums[1]);
                    levels[8] = ((low_60 == 31) ? 0.0f : levels[8]);
                    levels[8] = ((low_60 == 15) ? sums[0] : levels[8]);
                    levels[9] = ((low_60 == 31) ? 0.0f : levels[9]);
                    levels[9] = ((low_60 == 15) ? sums[1] : levels[9]);
                    int low_61 = groups * 8 + 1 & 63;
                    float merged_122;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_122) : "f"(levels[10]), "f"(sums[0]));
                    sums[0] = ((low_61 == 63) ? merged_122 : sums[0]);
                    float merged_123;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_123) : "f"(levels[11]), "f"(sums[1]));
                    sums[1] = ((low_61 == 63) ? merged_123 : sums[1]);
                    levels[10] = ((low_61 == 63) ? 0.0f : levels[10]);
                    levels[10] = ((low_61 == 31) ? sums[0] : levels[10]);
                    levels[11] = ((low_61 == 63) ? 0.0f : levels[11]);
                    levels[11] = ((low_61 == 31) ? sums[1] : levels[11]);
                    int low_62 = groups * 8 + 1 & 127;
                    float merged_124;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_124) : "f"(levels[12]), "f"(sums[0]));
                    sums[0] = ((low_62 == 127) ? merged_124 : sums[0]);
                    float merged_125;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_125) : "f"(levels[13]), "f"(sums[1]));
                    sums[1] = ((low_62 == 127) ? merged_125 : sums[1]);
                    levels[12] = ((low_62 == 127) ? 0.0f : levels[12]);
                    levels[12] = ((low_62 == 63) ? sums[0] : levels[12]);
                    levels[13] = ((low_62 == 127) ? 0.0f : levels[13]);
                    levels[13] = ((low_62 == 63) ? sums[1] : levels[13]);
                    int low_63 = groups * 8 + 1 & 255;
                    float merged_126;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_126) : "f"(levels[14]), "f"(sums[0]));
                    sums[0] = ((low_63 == 255) ? merged_126 : sums[0]);
                    float merged_127;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_127) : "f"(levels[15]), "f"(sums[1]));
                    sums[1] = ((low_63 == 255) ? merged_127 : sums[1]);
                    levels[14] = ((low_63 == 255) ? 0.0f : levels[14]);
                    levels[14] = ((low_63 == 127) ? sums[0] : levels[14]);
                    levels[15] = ((low_63 == 255) ? 0.0f : levels[15]);
                    levels[15] = ((low_63 == 127) ? sums[1] : levels[15]);
                    float carried_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_20) : "f"(top[0]), "f"(sums[0]));
                    top[0] = (((groups * 8 + 1 & 255) == 255) ? carried_20 : top[0]);
                    float carried_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_21) : "f"(top[1]), "f"(sums[1]));
                    top[1] = (((groups * 8 + 1 & 255) == 255) ? carried_21 : top[1]);
                }
                if (groups * 8 + 2 < nseg) {
                    sums[0] = rest[4];
                    sums[1] = rest[5];
                    int low_64 = groups * 8 + 2 & 1;
                    float merged_128;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_128) : "f"(levels[0]), "f"(sums[0]));
                    sums[0] = ((low_64 == 1) ? merged_128 : sums[0]);
                    float merged_129;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_129) : "f"(levels[1]), "f"(sums[1]));
                    sums[1] = ((low_64 == 1) ? merged_129 : sums[1]);
                    levels[0] = ((low_64 == 1) ? 0.0f : levels[0]);
                    levels[0] = ((low_64 == 0) ? sums[0] : levels[0]);
                    levels[1] = ((low_64 == 1) ? 0.0f : levels[1]);
                    levels[1] = ((low_64 == 0) ? sums[1] : levels[1]);
                    int low_65 = groups * 8 + 2 & 3;
                    float merged_130;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_130) : "f"(levels[2]), "f"(sums[0]));
                    sums[0] = ((low_65 == 3) ? merged_130 : sums[0]);
                    float merged_131;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_131) : "f"(levels[3]), "f"(sums[1]));
                    sums[1] = ((low_65 == 3) ? merged_131 : sums[1]);
                    levels[2] = ((low_65 == 3) ? 0.0f : levels[2]);
                    levels[2] = ((low_65 == 1) ? sums[0] : levels[2]);
                    levels[3] = ((low_65 == 3) ? 0.0f : levels[3]);
                    levels[3] = ((low_65 == 1) ? sums[1] : levels[3]);
                    int low_66 = groups * 8 + 2 & 7;
                    float merged_132;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_132) : "f"(levels[4]), "f"(sums[0]));
                    sums[0] = ((low_66 == 7) ? merged_132 : sums[0]);
                    float merged_133;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_133) : "f"(levels[5]), "f"(sums[1]));
                    sums[1] = ((low_66 == 7) ? merged_133 : sums[1]);
                    levels[4] = ((low_66 == 7) ? 0.0f : levels[4]);
                    levels[4] = ((low_66 == 3) ? sums[0] : levels[4]);
                    levels[5] = ((low_66 == 7) ? 0.0f : levels[5]);
                    levels[5] = ((low_66 == 3) ? sums[1] : levels[5]);
                    int low_67 = groups * 8 + 2 & 15;
                    float merged_134;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_134) : "f"(levels[6]), "f"(sums[0]));
                    sums[0] = ((low_67 == 15) ? merged_134 : sums[0]);
                    float merged_135;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_135) : "f"(levels[7]), "f"(sums[1]));
                    sums[1] = ((low_67 == 15) ? merged_135 : sums[1]);
                    levels[6] = ((low_67 == 15) ? 0.0f : levels[6]);
                    levels[6] = ((low_67 == 7) ? sums[0] : levels[6]);
                    levels[7] = ((low_67 == 15) ? 0.0f : levels[7]);
                    levels[7] = ((low_67 == 7) ? sums[1] : levels[7]);
                    int low_68 = groups * 8 + 2 & 31;
                    float merged_136;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_136) : "f"(levels[8]), "f"(sums[0]));
                    sums[0] = ((low_68 == 31) ? merged_136 : sums[0]);
                    float merged_137;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_137) : "f"(levels[9]), "f"(sums[1]));
                    sums[1] = ((low_68 == 31) ? merged_137 : sums[1]);
                    levels[8] = ((low_68 == 31) ? 0.0f : levels[8]);
                    levels[8] = ((low_68 == 15) ? sums[0] : levels[8]);
                    levels[9] = ((low_68 == 31) ? 0.0f : levels[9]);
                    levels[9] = ((low_68 == 15) ? sums[1] : levels[9]);
                    int low_69 = groups * 8 + 2 & 63;
                    float merged_138;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_138) : "f"(levels[10]), "f"(sums[0]));
                    sums[0] = ((low_69 == 63) ? merged_138 : sums[0]);
                    float merged_139;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_139) : "f"(levels[11]), "f"(sums[1]));
                    sums[1] = ((low_69 == 63) ? merged_139 : sums[1]);
                    levels[10] = ((low_69 == 63) ? 0.0f : levels[10]);
                    levels[10] = ((low_69 == 31) ? sums[0] : levels[10]);
                    levels[11] = ((low_69 == 63) ? 0.0f : levels[11]);
                    levels[11] = ((low_69 == 31) ? sums[1] : levels[11]);
                    int low_70 = groups * 8 + 2 & 127;
                    float merged_140;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_140) : "f"(levels[12]), "f"(sums[0]));
                    sums[0] = ((low_70 == 127) ? merged_140 : sums[0]);
                    float merged_141;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_141) : "f"(levels[13]), "f"(sums[1]));
                    sums[1] = ((low_70 == 127) ? merged_141 : sums[1]);
                    levels[12] = ((low_70 == 127) ? 0.0f : levels[12]);
                    levels[12] = ((low_70 == 63) ? sums[0] : levels[12]);
                    levels[13] = ((low_70 == 127) ? 0.0f : levels[13]);
                    levels[13] = ((low_70 == 63) ? sums[1] : levels[13]);
                    int low_71 = groups * 8 + 2 & 255;
                    float merged_142;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_142) : "f"(levels[14]), "f"(sums[0]));
                    sums[0] = ((low_71 == 255) ? merged_142 : sums[0]);
                    float merged_143;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_143) : "f"(levels[15]), "f"(sums[1]));
                    sums[1] = ((low_71 == 255) ? merged_143 : sums[1]);
                    levels[14] = ((low_71 == 255) ? 0.0f : levels[14]);
                    levels[14] = ((low_71 == 127) ? sums[0] : levels[14]);
                    levels[15] = ((low_71 == 255) ? 0.0f : levels[15]);
                    levels[15] = ((low_71 == 127) ? sums[1] : levels[15]);
                    float carried_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_22) : "f"(top[0]), "f"(sums[0]));
                    top[0] = (((groups * 8 + 2 & 255) == 255) ? carried_22 : top[0]);
                    float carried_23;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_23) : "f"(top[1]), "f"(sums[1]));
                    top[1] = (((groups * 8 + 2 & 255) == 255) ? carried_23 : top[1]);
                }
                if (groups * 8 + 3 < nseg) {
                    sums[0] = rest[6];
                    sums[1] = rest[7];
                    int low_72 = groups * 8 + 3 & 1;
                    float merged_144;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_144) : "f"(levels[0]), "f"(sums[0]));
                    sums[0] = ((low_72 == 1) ? merged_144 : sums[0]);
                    float merged_145;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_145) : "f"(levels[1]), "f"(sums[1]));
                    sums[1] = ((low_72 == 1) ? merged_145 : sums[1]);
                    levels[0] = ((low_72 == 1) ? 0.0f : levels[0]);
                    levels[0] = ((low_72 == 0) ? sums[0] : levels[0]);
                    levels[1] = ((low_72 == 1) ? 0.0f : levels[1]);
                    levels[1] = ((low_72 == 0) ? sums[1] : levels[1]);
                    int low_73 = groups * 8 + 3 & 3;
                    float merged_146;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_146) : "f"(levels[2]), "f"(sums[0]));
                    sums[0] = ((low_73 == 3) ? merged_146 : sums[0]);
                    float merged_147;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_147) : "f"(levels[3]), "f"(sums[1]));
                    sums[1] = ((low_73 == 3) ? merged_147 : sums[1]);
                    levels[2] = ((low_73 == 3) ? 0.0f : levels[2]);
                    levels[2] = ((low_73 == 1) ? sums[0] : levels[2]);
                    levels[3] = ((low_73 == 3) ? 0.0f : levels[3]);
                    levels[3] = ((low_73 == 1) ? sums[1] : levels[3]);
                    int low_74 = groups * 8 + 3 & 7;
                    float merged_148;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_148) : "f"(levels[4]), "f"(sums[0]));
                    sums[0] = ((low_74 == 7) ? merged_148 : sums[0]);
                    float merged_149;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_149) : "f"(levels[5]), "f"(sums[1]));
                    sums[1] = ((low_74 == 7) ? merged_149 : sums[1]);
                    levels[4] = ((low_74 == 7) ? 0.0f : levels[4]);
                    levels[4] = ((low_74 == 3) ? sums[0] : levels[4]);
                    levels[5] = ((low_74 == 7) ? 0.0f : levels[5]);
                    levels[5] = ((low_74 == 3) ? sums[1] : levels[5]);
                    int low_75 = groups * 8 + 3 & 15;
                    float merged_150;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_150) : "f"(levels[6]), "f"(sums[0]));
                    sums[0] = ((low_75 == 15) ? merged_150 : sums[0]);
                    float merged_151;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_151) : "f"(levels[7]), "f"(sums[1]));
                    sums[1] = ((low_75 == 15) ? merged_151 : sums[1]);
                    levels[6] = ((low_75 == 15) ? 0.0f : levels[6]);
                    levels[6] = ((low_75 == 7) ? sums[0] : levels[6]);
                    levels[7] = ((low_75 == 15) ? 0.0f : levels[7]);
                    levels[7] = ((low_75 == 7) ? sums[1] : levels[7]);
                    int low_76 = groups * 8 + 3 & 31;
                    float merged_152;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_152) : "f"(levels[8]), "f"(sums[0]));
                    sums[0] = ((low_76 == 31) ? merged_152 : sums[0]);
                    float merged_153;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_153) : "f"(levels[9]), "f"(sums[1]));
                    sums[1] = ((low_76 == 31) ? merged_153 : sums[1]);
                    levels[8] = ((low_76 == 31) ? 0.0f : levels[8]);
                    levels[8] = ((low_76 == 15) ? sums[0] : levels[8]);
                    levels[9] = ((low_76 == 31) ? 0.0f : levels[9]);
                    levels[9] = ((low_76 == 15) ? sums[1] : levels[9]);
                    int low_77 = groups * 8 + 3 & 63;
                    float merged_154;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_154) : "f"(levels[10]), "f"(sums[0]));
                    sums[0] = ((low_77 == 63) ? merged_154 : sums[0]);
                    float merged_155;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_155) : "f"(levels[11]), "f"(sums[1]));
                    sums[1] = ((low_77 == 63) ? merged_155 : sums[1]);
                    levels[10] = ((low_77 == 63) ? 0.0f : levels[10]);
                    levels[10] = ((low_77 == 31) ? sums[0] : levels[10]);
                    levels[11] = ((low_77 == 63) ? 0.0f : levels[11]);
                    levels[11] = ((low_77 == 31) ? sums[1] : levels[11]);
                    int low_78 = groups * 8 + 3 & 127;
                    float merged_156;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_156) : "f"(levels[12]), "f"(sums[0]));
                    sums[0] = ((low_78 == 127) ? merged_156 : sums[0]);
                    float merged_157;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_157) : "f"(levels[13]), "f"(sums[1]));
                    sums[1] = ((low_78 == 127) ? merged_157 : sums[1]);
                    levels[12] = ((low_78 == 127) ? 0.0f : levels[12]);
                    levels[12] = ((low_78 == 63) ? sums[0] : levels[12]);
                    levels[13] = ((low_78 == 127) ? 0.0f : levels[13]);
                    levels[13] = ((low_78 == 63) ? sums[1] : levels[13]);
                    int low_79 = groups * 8 + 3 & 255;
                    float merged_158;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_158) : "f"(levels[14]), "f"(sums[0]));
                    sums[0] = ((low_79 == 255) ? merged_158 : sums[0]);
                    float merged_159;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_159) : "f"(levels[15]), "f"(sums[1]));
                    sums[1] = ((low_79 == 255) ? merged_159 : sums[1]);
                    levels[14] = ((low_79 == 255) ? 0.0f : levels[14]);
                    levels[14] = ((low_79 == 127) ? sums[0] : levels[14]);
                    levels[15] = ((low_79 == 255) ? 0.0f : levels[15]);
                    levels[15] = ((low_79 == 127) ? sums[1] : levels[15]);
                    float carried_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_24) : "f"(top[0]), "f"(sums[0]));
                    top[0] = (((groups * 8 + 3 & 255) == 255) ? carried_24 : top[0]);
                    float carried_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_25) : "f"(top[1]), "f"(sums[1]));
                    top[1] = (((groups * 8 + 3 & 255) == 255) ? carried_25 : top[1]);
                }
                if (groups * 8 + 4 < nseg) {
                    sums[0] = rest[8];
                    sums[1] = rest[9];
                    int low_80 = groups * 8 + 4 & 1;
                    float merged_160;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_160) : "f"(levels[0]), "f"(sums[0]));
                    sums[0] = ((low_80 == 1) ? merged_160 : sums[0]);
                    float merged_161;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_161) : "f"(levels[1]), "f"(sums[1]));
                    sums[1] = ((low_80 == 1) ? merged_161 : sums[1]);
                    levels[0] = ((low_80 == 1) ? 0.0f : levels[0]);
                    levels[0] = ((low_80 == 0) ? sums[0] : levels[0]);
                    levels[1] = ((low_80 == 1) ? 0.0f : levels[1]);
                    levels[1] = ((low_80 == 0) ? sums[1] : levels[1]);
                    int low_81 = groups * 8 + 4 & 3;
                    float merged_162;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_162) : "f"(levels[2]), "f"(sums[0]));
                    sums[0] = ((low_81 == 3) ? merged_162 : sums[0]);
                    float merged_163;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_163) : "f"(levels[3]), "f"(sums[1]));
                    sums[1] = ((low_81 == 3) ? merged_163 : sums[1]);
                    levels[2] = ((low_81 == 3) ? 0.0f : levels[2]);
                    levels[2] = ((low_81 == 1) ? sums[0] : levels[2]);
                    levels[3] = ((low_81 == 3) ? 0.0f : levels[3]);
                    levels[3] = ((low_81 == 1) ? sums[1] : levels[3]);
                    int low_82 = groups * 8 + 4 & 7;
                    float merged_164;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_164) : "f"(levels[4]), "f"(sums[0]));
                    sums[0] = ((low_82 == 7) ? merged_164 : sums[0]);
                    float merged_165;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_165) : "f"(levels[5]), "f"(sums[1]));
                    sums[1] = ((low_82 == 7) ? merged_165 : sums[1]);
                    levels[4] = ((low_82 == 7) ? 0.0f : levels[4]);
                    levels[4] = ((low_82 == 3) ? sums[0] : levels[4]);
                    levels[5] = ((low_82 == 7) ? 0.0f : levels[5]);
                    levels[5] = ((low_82 == 3) ? sums[1] : levels[5]);
                    int low_83 = groups * 8 + 4 & 15;
                    float merged_166;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_166) : "f"(levels[6]), "f"(sums[0]));
                    sums[0] = ((low_83 == 15) ? merged_166 : sums[0]);
                    float merged_167;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_167) : "f"(levels[7]), "f"(sums[1]));
                    sums[1] = ((low_83 == 15) ? merged_167 : sums[1]);
                    levels[6] = ((low_83 == 15) ? 0.0f : levels[6]);
                    levels[6] = ((low_83 == 7) ? sums[0] : levels[6]);
                    levels[7] = ((low_83 == 15) ? 0.0f : levels[7]);
                    levels[7] = ((low_83 == 7) ? sums[1] : levels[7]);
                    int low_84 = groups * 8 + 4 & 31;
                    float merged_168;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_168) : "f"(levels[8]), "f"(sums[0]));
                    sums[0] = ((low_84 == 31) ? merged_168 : sums[0]);
                    float merged_169;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_169) : "f"(levels[9]), "f"(sums[1]));
                    sums[1] = ((low_84 == 31) ? merged_169 : sums[1]);
                    levels[8] = ((low_84 == 31) ? 0.0f : levels[8]);
                    levels[8] = ((low_84 == 15) ? sums[0] : levels[8]);
                    levels[9] = ((low_84 == 31) ? 0.0f : levels[9]);
                    levels[9] = ((low_84 == 15) ? sums[1] : levels[9]);
                    int low_85 = groups * 8 + 4 & 63;
                    float merged_170;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_170) : "f"(levels[10]), "f"(sums[0]));
                    sums[0] = ((low_85 == 63) ? merged_170 : sums[0]);
                    float merged_171;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_171) : "f"(levels[11]), "f"(sums[1]));
                    sums[1] = ((low_85 == 63) ? merged_171 : sums[1]);
                    levels[10] = ((low_85 == 63) ? 0.0f : levels[10]);
                    levels[10] = ((low_85 == 31) ? sums[0] : levels[10]);
                    levels[11] = ((low_85 == 63) ? 0.0f : levels[11]);
                    levels[11] = ((low_85 == 31) ? sums[1] : levels[11]);
                    int low_86 = groups * 8 + 4 & 127;
                    float merged_172;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_172) : "f"(levels[12]), "f"(sums[0]));
                    sums[0] = ((low_86 == 127) ? merged_172 : sums[0]);
                    float merged_173;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_173) : "f"(levels[13]), "f"(sums[1]));
                    sums[1] = ((low_86 == 127) ? merged_173 : sums[1]);
                    levels[12] = ((low_86 == 127) ? 0.0f : levels[12]);
                    levels[12] = ((low_86 == 63) ? sums[0] : levels[12]);
                    levels[13] = ((low_86 == 127) ? 0.0f : levels[13]);
                    levels[13] = ((low_86 == 63) ? sums[1] : levels[13]);
                    int low_87 = groups * 8 + 4 & 255;
                    float merged_174;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_174) : "f"(levels[14]), "f"(sums[0]));
                    sums[0] = ((low_87 == 255) ? merged_174 : sums[0]);
                    float merged_175;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_175) : "f"(levels[15]), "f"(sums[1]));
                    sums[1] = ((low_87 == 255) ? merged_175 : sums[1]);
                    levels[14] = ((low_87 == 255) ? 0.0f : levels[14]);
                    levels[14] = ((low_87 == 127) ? sums[0] : levels[14]);
                    levels[15] = ((low_87 == 255) ? 0.0f : levels[15]);
                    levels[15] = ((low_87 == 127) ? sums[1] : levels[15]);
                    float carried_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_26) : "f"(top[0]), "f"(sums[0]));
                    top[0] = (((groups * 8 + 4 & 255) == 255) ? carried_26 : top[0]);
                    float carried_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_27) : "f"(top[1]), "f"(sums[1]));
                    top[1] = (((groups * 8 + 4 & 255) == 255) ? carried_27 : top[1]);
                }
                if (groups * 8 + 5 < nseg) {
                    sums[0] = rest[10];
                    sums[1] = rest[11];
                    int low_88 = groups * 8 + 5 & 1;
                    float merged_176;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_176) : "f"(levels[0]), "f"(sums[0]));
                    sums[0] = ((low_88 == 1) ? merged_176 : sums[0]);
                    float merged_177;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_177) : "f"(levels[1]), "f"(sums[1]));
                    sums[1] = ((low_88 == 1) ? merged_177 : sums[1]);
                    levels[0] = ((low_88 == 1) ? 0.0f : levels[0]);
                    levels[0] = ((low_88 == 0) ? sums[0] : levels[0]);
                    levels[1] = ((low_88 == 1) ? 0.0f : levels[1]);
                    levels[1] = ((low_88 == 0) ? sums[1] : levels[1]);
                    int low_89 = groups * 8 + 5 & 3;
                    float merged_178;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_178) : "f"(levels[2]), "f"(sums[0]));
                    sums[0] = ((low_89 == 3) ? merged_178 : sums[0]);
                    float merged_179;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_179) : "f"(levels[3]), "f"(sums[1]));
                    sums[1] = ((low_89 == 3) ? merged_179 : sums[1]);
                    levels[2] = ((low_89 == 3) ? 0.0f : levels[2]);
                    levels[2] = ((low_89 == 1) ? sums[0] : levels[2]);
                    levels[3] = ((low_89 == 3) ? 0.0f : levels[3]);
                    levels[3] = ((low_89 == 1) ? sums[1] : levels[3]);
                    int low_90 = groups * 8 + 5 & 7;
                    float merged_180;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_180) : "f"(levels[4]), "f"(sums[0]));
                    sums[0] = ((low_90 == 7) ? merged_180 : sums[0]);
                    float merged_181;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_181) : "f"(levels[5]), "f"(sums[1]));
                    sums[1] = ((low_90 == 7) ? merged_181 : sums[1]);
                    levels[4] = ((low_90 == 7) ? 0.0f : levels[4]);
                    levels[4] = ((low_90 == 3) ? sums[0] : levels[4]);
                    levels[5] = ((low_90 == 7) ? 0.0f : levels[5]);
                    levels[5] = ((low_90 == 3) ? sums[1] : levels[5]);
                    int low_91 = groups * 8 + 5 & 15;
                    float merged_182;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_182) : "f"(levels[6]), "f"(sums[0]));
                    sums[0] = ((low_91 == 15) ? merged_182 : sums[0]);
                    float merged_183;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_183) : "f"(levels[7]), "f"(sums[1]));
                    sums[1] = ((low_91 == 15) ? merged_183 : sums[1]);
                    levels[6] = ((low_91 == 15) ? 0.0f : levels[6]);
                    levels[6] = ((low_91 == 7) ? sums[0] : levels[6]);
                    levels[7] = ((low_91 == 15) ? 0.0f : levels[7]);
                    levels[7] = ((low_91 == 7) ? sums[1] : levels[7]);
                    int low_92 = groups * 8 + 5 & 31;
                    float merged_184;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_184) : "f"(levels[8]), "f"(sums[0]));
                    sums[0] = ((low_92 == 31) ? merged_184 : sums[0]);
                    float merged_185;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_185) : "f"(levels[9]), "f"(sums[1]));
                    sums[1] = ((low_92 == 31) ? merged_185 : sums[1]);
                    levels[8] = ((low_92 == 31) ? 0.0f : levels[8]);
                    levels[8] = ((low_92 == 15) ? sums[0] : levels[8]);
                    levels[9] = ((low_92 == 31) ? 0.0f : levels[9]);
                    levels[9] = ((low_92 == 15) ? sums[1] : levels[9]);
                    int low_93 = groups * 8 + 5 & 63;
                    float merged_186;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_186) : "f"(levels[10]), "f"(sums[0]));
                    sums[0] = ((low_93 == 63) ? merged_186 : sums[0]);
                    float merged_187;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_187) : "f"(levels[11]), "f"(sums[1]));
                    sums[1] = ((low_93 == 63) ? merged_187 : sums[1]);
                    levels[10] = ((low_93 == 63) ? 0.0f : levels[10]);
                    levels[10] = ((low_93 == 31) ? sums[0] : levels[10]);
                    levels[11] = ((low_93 == 63) ? 0.0f : levels[11]);
                    levels[11] = ((low_93 == 31) ? sums[1] : levels[11]);
                    int low_94 = groups * 8 + 5 & 127;
                    float merged_188;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_188) : "f"(levels[12]), "f"(sums[0]));
                    sums[0] = ((low_94 == 127) ? merged_188 : sums[0]);
                    float merged_189;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_189) : "f"(levels[13]), "f"(sums[1]));
                    sums[1] = ((low_94 == 127) ? merged_189 : sums[1]);
                    levels[12] = ((low_94 == 127) ? 0.0f : levels[12]);
                    levels[12] = ((low_94 == 63) ? sums[0] : levels[12]);
                    levels[13] = ((low_94 == 127) ? 0.0f : levels[13]);
                    levels[13] = ((low_94 == 63) ? sums[1] : levels[13]);
                    int low_95 = groups * 8 + 5 & 255;
                    float merged_190;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_190) : "f"(levels[14]), "f"(sums[0]));
                    sums[0] = ((low_95 == 255) ? merged_190 : sums[0]);
                    float merged_191;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_191) : "f"(levels[15]), "f"(sums[1]));
                    sums[1] = ((low_95 == 255) ? merged_191 : sums[1]);
                    levels[14] = ((low_95 == 255) ? 0.0f : levels[14]);
                    levels[14] = ((low_95 == 127) ? sums[0] : levels[14]);
                    levels[15] = ((low_95 == 255) ? 0.0f : levels[15]);
                    levels[15] = ((low_95 == 127) ? sums[1] : levels[15]);
                    float carried_28;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_28) : "f"(top[0]), "f"(sums[0]));
                    top[0] = (((groups * 8 + 5 & 255) == 255) ? carried_28 : top[0]);
                    float carried_29;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_29) : "f"(top[1]), "f"(sums[1]));
                    top[1] = (((groups * 8 + 5 & 255) == 255) ? carried_29 : top[1]);
                }
                if (groups * 8 + 6 < nseg) {
                    sums[0] = rest[12];
                    sums[1] = rest[13];
                    int low_96 = groups * 8 + 6 & 1;
                    float merged_192;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_192) : "f"(levels[0]), "f"(sums[0]));
                    sums[0] = ((low_96 == 1) ? merged_192 : sums[0]);
                    float merged_193;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_193) : "f"(levels[1]), "f"(sums[1]));
                    sums[1] = ((low_96 == 1) ? merged_193 : sums[1]);
                    levels[0] = ((low_96 == 1) ? 0.0f : levels[0]);
                    levels[0] = ((low_96 == 0) ? sums[0] : levels[0]);
                    levels[1] = ((low_96 == 1) ? 0.0f : levels[1]);
                    levels[1] = ((low_96 == 0) ? sums[1] : levels[1]);
                    int low_97 = groups * 8 + 6 & 3;
                    float merged_194;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_194) : "f"(levels[2]), "f"(sums[0]));
                    sums[0] = ((low_97 == 3) ? merged_194 : sums[0]);
                    float merged_195;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_195) : "f"(levels[3]), "f"(sums[1]));
                    sums[1] = ((low_97 == 3) ? merged_195 : sums[1]);
                    levels[2] = ((low_97 == 3) ? 0.0f : levels[2]);
                    levels[2] = ((low_97 == 1) ? sums[0] : levels[2]);
                    levels[3] = ((low_97 == 3) ? 0.0f : levels[3]);
                    levels[3] = ((low_97 == 1) ? sums[1] : levels[3]);
                    int low_98 = groups * 8 + 6 & 7;
                    float merged_196;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_196) : "f"(levels[4]), "f"(sums[0]));
                    sums[0] = ((low_98 == 7) ? merged_196 : sums[0]);
                    float merged_197;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_197) : "f"(levels[5]), "f"(sums[1]));
                    sums[1] = ((low_98 == 7) ? merged_197 : sums[1]);
                    levels[4] = ((low_98 == 7) ? 0.0f : levels[4]);
                    levels[4] = ((low_98 == 3) ? sums[0] : levels[4]);
                    levels[5] = ((low_98 == 7) ? 0.0f : levels[5]);
                    levels[5] = ((low_98 == 3) ? sums[1] : levels[5]);
                    int low_99 = groups * 8 + 6 & 15;
                    float merged_198;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_198) : "f"(levels[6]), "f"(sums[0]));
                    sums[0] = ((low_99 == 15) ? merged_198 : sums[0]);
                    float merged_199;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_199) : "f"(levels[7]), "f"(sums[1]));
                    sums[1] = ((low_99 == 15) ? merged_199 : sums[1]);
                    levels[6] = ((low_99 == 15) ? 0.0f : levels[6]);
                    levels[6] = ((low_99 == 7) ? sums[0] : levels[6]);
                    levels[7] = ((low_99 == 15) ? 0.0f : levels[7]);
                    levels[7] = ((low_99 == 7) ? sums[1] : levels[7]);
                    int low_100 = groups * 8 + 6 & 31;
                    float merged_200;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_200) : "f"(levels[8]), "f"(sums[0]));
                    sums[0] = ((low_100 == 31) ? merged_200 : sums[0]);
                    float merged_201;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_201) : "f"(levels[9]), "f"(sums[1]));
                    sums[1] = ((low_100 == 31) ? merged_201 : sums[1]);
                    levels[8] = ((low_100 == 31) ? 0.0f : levels[8]);
                    levels[8] = ((low_100 == 15) ? sums[0] : levels[8]);
                    levels[9] = ((low_100 == 31) ? 0.0f : levels[9]);
                    levels[9] = ((low_100 == 15) ? sums[1] : levels[9]);
                    int low_101 = groups * 8 + 6 & 63;
                    float merged_202;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_202) : "f"(levels[10]), "f"(sums[0]));
                    sums[0] = ((low_101 == 63) ? merged_202 : sums[0]);
                    float merged_203;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_203) : "f"(levels[11]), "f"(sums[1]));
                    sums[1] = ((low_101 == 63) ? merged_203 : sums[1]);
                    levels[10] = ((low_101 == 63) ? 0.0f : levels[10]);
                    levels[10] = ((low_101 == 31) ? sums[0] : levels[10]);
                    levels[11] = ((low_101 == 63) ? 0.0f : levels[11]);
                    levels[11] = ((low_101 == 31) ? sums[1] : levels[11]);
                    int low_102 = groups * 8 + 6 & 127;
                    float merged_204;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_204) : "f"(levels[12]), "f"(sums[0]));
                    sums[0] = ((low_102 == 127) ? merged_204 : sums[0]);
                    float merged_205;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_205) : "f"(levels[13]), "f"(sums[1]));
                    sums[1] = ((low_102 == 127) ? merged_205 : sums[1]);
                    levels[12] = ((low_102 == 127) ? 0.0f : levels[12]);
                    levels[12] = ((low_102 == 63) ? sums[0] : levels[12]);
                    levels[13] = ((low_102 == 127) ? 0.0f : levels[13]);
                    levels[13] = ((low_102 == 63) ? sums[1] : levels[13]);
                    int low_103 = groups * 8 + 6 & 255;
                    float merged_206;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_206) : "f"(levels[14]), "f"(sums[0]));
                    sums[0] = ((low_103 == 255) ? merged_206 : sums[0]);
                    float merged_207;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(merged_207) : "f"(levels[15]), "f"(sums[1]));
                    sums[1] = ((low_103 == 255) ? merged_207 : sums[1]);
                    levels[14] = ((low_103 == 255) ? 0.0f : levels[14]);
                    levels[14] = ((low_103 == 127) ? sums[0] : levels[14]);
                    levels[15] = ((low_103 == 255) ? 0.0f : levels[15]);
                    levels[15] = ((low_103 == 127) ? sums[1] : levels[15]);
                    float carried_30;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_30) : "f"(top[0]), "f"(sums[0]));
                    top[0] = (((groups * 8 + 6 & 255) == 255) ? carried_30 : top[0]);
                    float carried_31;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(carried_31) : "f"(top[1]), "f"(sums[1]));
                    top[1] = (((groups * 8 + 6 & 255) == 255) ? carried_31 : top[1]);
                }
                sums[0] = 0.0f;
                sums[1] = 0.0f;
                float got_16;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_16) : "f"(levels[0]), "f"(sums[0]));
                sums[0] = got_16;
                float got_17;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_17) : "f"(levels[1]), "f"(sums[1]));
                sums[1] = got_17;
                float got_18;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_18) : "f"(levels[2]), "f"(sums[0]));
                sums[0] = got_18;
                float got_19;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_19) : "f"(levels[3]), "f"(sums[1]));
                sums[1] = got_19;
                float got_20;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_20) : "f"(levels[4]), "f"(sums[0]));
                sums[0] = got_20;
                float got_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_21) : "f"(levels[5]), "f"(sums[1]));
                sums[1] = got_21;
                float got_22;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_22) : "f"(levels[6]), "f"(sums[0]));
                sums[0] = got_22;
                float got_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_23) : "f"(levels[7]), "f"(sums[1]));
                sums[1] = got_23;
                float got_24;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_24) : "f"(levels[8]), "f"(sums[0]));
                sums[0] = got_24;
                float got_25;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_25) : "f"(levels[9]), "f"(sums[1]));
                sums[1] = got_25;
                float got_26;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_26) : "f"(levels[10]), "f"(sums[0]));
                sums[0] = got_26;
                float got_27;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_27) : "f"(levels[11]), "f"(sums[1]));
                sums[1] = got_27;
                float got_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_28) : "f"(levels[12]), "f"(sums[0]));
                sums[0] = got_28;
                float got_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_29) : "f"(levels[13]), "f"(sums[1]));
                sums[1] = got_29;
                float got_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_30) : "f"(levels[14]), "f"(sums[0]));
                sums[0] = got_30;
                float got_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(got_31) : "f"(levels[15]), "f"(sums[1]));
                sums[1] = got_31;
                float whole_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(whole_2) : "f"(top[0]), "f"(sums[0]));
                sums[0] = whole_2;
                float whole_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(whole_3) : "f"(top[1]), "f"(sums[1]));
                sums[1] = whole_3;
            }
        }
    }
    if (last == (unsigned int)(g - 1)) {
        if (j == 0) {
            ddt_bias[head * 128 + d] = sums[1];
            float v = sums[0];
            float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, v, 16);
            float _add_177;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(v), "f"(_shfl_xor_sync_178));
            v = _add_177;
            float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, v, 8);
            float _add_179;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(v), "f"(_shfl_xor_sync_180));
            v = _add_179;
            float _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, v, 4);
            float _add_181;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(v), "f"(_shfl_xor_sync_182));
            v = _add_181;
            float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, v, 2);
            float _add_183;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(v), "f"(_shfl_xor_sync_184));
            v = _add_183;
            float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, v, 1);
            float _add_185;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(v), "f"(_shfl_xor_sync_186));
            v = _add_185;
            if (lane == 0) {
                wsum[warp] = v;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        if (last == (unsigned int)(g - 1)) {
            float w01;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(w01) : "f"(wsum[0]), "f"(wsum[1]));
            float w23;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(w23) : "f"(wsum[2]), "f"(wsum[3]));
            float _add_187;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(w01), "f"(w23));
            dA_log[head] = _add_187;
        }
    }
}

} // extern "C"
