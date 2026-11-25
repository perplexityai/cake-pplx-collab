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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_attn_res_det_train_f2e8cf286e7fa021968d(float* __restrict__ dw_rows, float* __restrict__ dw, int G)
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
    float* sums = reinterpret_cast<float*>(smem_raw + 0);
    const int sums_addr = smem + 0;

    // === Task calls (declaration order) ===
    int slice_ = tid >> 5;
    int quad = tid & 31;
    int column = bid * 128 + quad * 4;
    float acc[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        acc[c] = -0.0f;
    }
    #pragma unroll 1
    for (int g = slice_; g < G; g += 128) {
        float rows[64];
        #pragma unroll
        for (int c_0 = 0; c_0 < 4; c_0++) {
            rows[c_0] = -0.0f;
        }
        if (g < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)g * 7168ll + (long long)column) + 0);
                rows[0 + 0] = _v4.x;
                rows[0 + 1] = _v4.y;
                rows[0 + 2] = _v4.z;
                rows[0 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_1 = 0; c_1 < 4; c_1++) {
            rows[4 + c_1] = -0.0f;
        }
        if (g + 8 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 8) * 7168ll + (long long)column) + 0);
                rows[4 + 0] = _v4.x;
                rows[4 + 1] = _v4.y;
                rows[4 + 2] = _v4.z;
                rows[4 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_2 = 0; c_2 < 4; c_2++) {
            rows[8 + c_2] = -0.0f;
        }
        if (g + 16 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 16) * 7168ll + (long long)column) + 0);
                rows[8 + 0] = _v4.x;
                rows[8 + 1] = _v4.y;
                rows[8 + 2] = _v4.z;
                rows[8 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_3 = 0; c_3 < 4; c_3++) {
            rows[12 + c_3] = -0.0f;
        }
        if (g + 24 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 24) * 7168ll + (long long)column) + 0);
                rows[12 + 0] = _v4.x;
                rows[12 + 1] = _v4.y;
                rows[12 + 2] = _v4.z;
                rows[12 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_4 = 0; c_4 < 4; c_4++) {
            rows[16 + c_4] = -0.0f;
        }
        if (g + 32 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 32) * 7168ll + (long long)column) + 0);
                rows[16 + 0] = _v4.x;
                rows[16 + 1] = _v4.y;
                rows[16 + 2] = _v4.z;
                rows[16 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_5 = 0; c_5 < 4; c_5++) {
            rows[20 + c_5] = -0.0f;
        }
        if (g + 40 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 40) * 7168ll + (long long)column) + 0);
                rows[20 + 0] = _v4.x;
                rows[20 + 1] = _v4.y;
                rows[20 + 2] = _v4.z;
                rows[20 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_6 = 0; c_6 < 4; c_6++) {
            rows[24 + c_6] = -0.0f;
        }
        if (g + 48 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 48) * 7168ll + (long long)column) + 0);
                rows[24 + 0] = _v4.x;
                rows[24 + 1] = _v4.y;
                rows[24 + 2] = _v4.z;
                rows[24 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_7 = 0; c_7 < 4; c_7++) {
            rows[28 + c_7] = -0.0f;
        }
        if (g + 56 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 56) * 7168ll + (long long)column) + 0);
                rows[28 + 0] = _v4.x;
                rows[28 + 1] = _v4.y;
                rows[28 + 2] = _v4.z;
                rows[28 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_8 = 0; c_8 < 4; c_8++) {
            rows[32 + c_8] = -0.0f;
        }
        if (g + 64 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 64) * 7168ll + (long long)column) + 0);
                rows[32 + 0] = _v4.x;
                rows[32 + 1] = _v4.y;
                rows[32 + 2] = _v4.z;
                rows[32 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_9 = 0; c_9 < 4; c_9++) {
            rows[36 + c_9] = -0.0f;
        }
        if (g + 72 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 72) * 7168ll + (long long)column) + 0);
                rows[36 + 0] = _v4.x;
                rows[36 + 1] = _v4.y;
                rows[36 + 2] = _v4.z;
                rows[36 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_10 = 0; c_10 < 4; c_10++) {
            rows[40 + c_10] = -0.0f;
        }
        if (g + 80 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 80) * 7168ll + (long long)column) + 0);
                rows[40 + 0] = _v4.x;
                rows[40 + 1] = _v4.y;
                rows[40 + 2] = _v4.z;
                rows[40 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_11 = 0; c_11 < 4; c_11++) {
            rows[44 + c_11] = -0.0f;
        }
        if (g + 88 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 88) * 7168ll + (long long)column) + 0);
                rows[44 + 0] = _v4.x;
                rows[44 + 1] = _v4.y;
                rows[44 + 2] = _v4.z;
                rows[44 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_12 = 0; c_12 < 4; c_12++) {
            rows[48 + c_12] = -0.0f;
        }
        if (g + 96 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 96) * 7168ll + (long long)column) + 0);
                rows[48 + 0] = _v4.x;
                rows[48 + 1] = _v4.y;
                rows[48 + 2] = _v4.z;
                rows[48 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_13 = 0; c_13 < 4; c_13++) {
            rows[52 + c_13] = -0.0f;
        }
        if (g + 104 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 104) * 7168ll + (long long)column) + 0);
                rows[52 + 0] = _v4.x;
                rows[52 + 1] = _v4.y;
                rows[52 + 2] = _v4.z;
                rows[52 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_14 = 0; c_14 < 4; c_14++) {
            rows[56 + c_14] = -0.0f;
        }
        if (g + 112 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 112) * 7168ll + (long long)column) + 0);
                rows[56 + 0] = _v4.x;
                rows[56 + 1] = _v4.y;
                rows[56 + 2] = _v4.z;
                rows[56 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_15 = 0; c_15 < 4; c_15++) {
            rows[60 + c_15] = -0.0f;
        }
        if (g + 120 < G) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dw_rows + ((long long)(g + 120) * 7168ll + (long long)column) + 0);
                rows[60 + 0] = _v4.x;
                rows[60 + 1] = _v4.y;
                rows[60 + 2] = _v4.z;
                rows[60 + 3] = _v4.w;
            }
        }
        #pragma unroll
        for (int c_16 = 0; c_16 < 4; c_16++) {
            float _add_16;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(acc[c_16]), "f"(rows[c_16]));
            acc[c_16] = _add_16;
        }
        #pragma unroll
        for (int c_17 = 0; c_17 < 4; c_17++) {
            float _add_17;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(acc[c_17]), "f"(rows[4 + c_17]));
            acc[c_17] = _add_17;
        }
        #pragma unroll
        for (int c_18 = 0; c_18 < 4; c_18++) {
            float _add_18;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(acc[c_18]), "f"(rows[8 + c_18]));
            acc[c_18] = _add_18;
        }
        #pragma unroll
        for (int c_19 = 0; c_19 < 4; c_19++) {
            float _add_19;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(acc[c_19]), "f"(rows[12 + c_19]));
            acc[c_19] = _add_19;
        }
        #pragma unroll
        for (int c_20 = 0; c_20 < 4; c_20++) {
            float _add_20;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(acc[c_20]), "f"(rows[16 + c_20]));
            acc[c_20] = _add_20;
        }
        #pragma unroll
        for (int c_21 = 0; c_21 < 4; c_21++) {
            float _add_21;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(acc[c_21]), "f"(rows[20 + c_21]));
            acc[c_21] = _add_21;
        }
        #pragma unroll
        for (int c_22 = 0; c_22 < 4; c_22++) {
            float _add_22;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(acc[c_22]), "f"(rows[24 + c_22]));
            acc[c_22] = _add_22;
        }
        #pragma unroll
        for (int c_23 = 0; c_23 < 4; c_23++) {
            float _add_23;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(acc[c_23]), "f"(rows[28 + c_23]));
            acc[c_23] = _add_23;
        }
        #pragma unroll
        for (int c_24 = 0; c_24 < 4; c_24++) {
            float _add_24;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(acc[c_24]), "f"(rows[32 + c_24]));
            acc[c_24] = _add_24;
        }
        #pragma unroll
        for (int c_25 = 0; c_25 < 4; c_25++) {
            float _add_25;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(acc[c_25]), "f"(rows[36 + c_25]));
            acc[c_25] = _add_25;
        }
        #pragma unroll
        for (int c_26 = 0; c_26 < 4; c_26++) {
            float _add_26;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(acc[c_26]), "f"(rows[40 + c_26]));
            acc[c_26] = _add_26;
        }
        #pragma unroll
        for (int c_27 = 0; c_27 < 4; c_27++) {
            float _add_27;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(acc[c_27]), "f"(rows[44 + c_27]));
            acc[c_27] = _add_27;
        }
        #pragma unroll
        for (int c_28 = 0; c_28 < 4; c_28++) {
            float _add_28;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(acc[c_28]), "f"(rows[48 + c_28]));
            acc[c_28] = _add_28;
        }
        #pragma unroll
        for (int c_29 = 0; c_29 < 4; c_29++) {
            float _add_29;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(acc[c_29]), "f"(rows[52 + c_29]));
            acc[c_29] = _add_29;
        }
        #pragma unroll
        for (int c_30 = 0; c_30 < 4; c_30++) {
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(acc[c_30]), "f"(rows[56 + c_30]));
            acc[c_30] = _add_30;
        }
        #pragma unroll
        for (int c_31 = 0; c_31 < 4; c_31++) {
            float _add_31;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(acc[c_31]), "f"(rows[60 + c_31]));
            acc[c_31] = _add_31;
        }
    }
    #pragma unroll
    for (int c_32 = 0; c_32 < 4; c_32++) {
        sums[slice_ * 128 + quad * 4 + c_32] = acc[c_32];
    }
    __syncthreads();
    if (slice_ == 0) {
        float out[4];
        #pragma unroll
        for (int c_33 = 0; c_33 < 4; c_33++) {
            float tree[8];
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                tree[s] = sums[s * 128 + quad * 4 + c_33];
            }
            #pragma unroll
            for (int s_0 = 0; s_0 < 8; s_0 += 2) {
                float _add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(tree[s_0]), "f"(tree[s_0 + 1]));
                tree[s_0] = _add_32;
            }
            #pragma unroll
            for (int s_1 = 0; s_1 < 8; s_1 += 4) {
                float _add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(tree[s_1]), "f"(tree[s_1 + 2]));
                tree[s_1] = _add_33;
            }
            #pragma unroll
            for (int s_2 = 0; s_2 < 8; s_2 += 8) {
                float _add_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(tree[s_2]), "f"(tree[s_2 + 4]));
                tree[s_2] = _add_34;
            }
            out[c_33] = tree[0];
        }
        {
            float4 _v4 = make_float4(out[0 + 0], out[0 + 1], out[0 + 2], out[0 + 3]);
            *reinterpret_cast<float4*>((dw + column) + 0) = _v4;
        }
    }
}

} // extern "C"
