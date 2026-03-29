#include "linear_cpu.hpp"
#include <cblas.h>
#include <type_traits>
#include <vector>
#include "../../../utils.hpp"
#include <omp.h>

template <typename T>
void linear_kernel(const T *input, const T *weight, const T *bias, T *out,
                   size_t M, size_t K, size_t N) {
    if constexpr (std::is_same_v<T, float>) {
        // 🚀 【F32 狂飙模式】：交给 OpenBLAS 降维打击
        const float* f_input = (const float*)input;
        const float* f_weight = (const float*)weight;
        float* f_out = (float*)out;

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, K, 1.0f,
            f_input, K, f_weight, K,
            0.0f, f_out, N
        );

        // 加上偏置
        if (bias != nullptr) {
            const float* f_bias = (const float*)bias;
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    f_out[m * N + n] += f_bias[n];
                }
            }
        }
    } 
    // 🚨 绝对安全版：剥离 OpenMP，彻底杜绝线程死锁！
    else if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
        std::vector<float> f32_x(M * K);
        std::vector<float> f32_w(N * K);
        std::vector<float> f32_out(M * N);

        // 1. 单线程安全强转（CPU 跑这点 for 循环极快）
        for (size_t i = 0; i < M * K; i++) {
            f32_x[i] = llaisys::utils::cast<float>(input[i]);
        }
        
        for (size_t i = 0; i < N * K; i++) {
            f32_w[i] = llaisys::utils::cast<float>(weight[i]);
        }

        // 2. 将所有并发算力全部留给 OpenBLAS！
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f,
                    f32_x.data(), K,
                    f32_w.data(), K,
                    0.0f,
                    f32_out.data(), N);

        // 3. 算完后，单线程安全转回
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float val = f32_out[m * N + n];
                if (bias != nullptr) {
                    val += llaisys::utils::cast<float>(bias[n]);
                }
                out[m * N + n] = llaisys::utils::cast<T>(val);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void linear(const std::byte *input_data, const std::byte *weight_data, const std::byte *bias_data,
            std::byte *out_data, llaisysDataType_t dtype,
            size_t M, size_t K, size_t N) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<float>(
            reinterpret_cast<const float *>(input_data),
            reinterpret_cast<const float *>(weight_data),
            reinterpret_cast<const float *>(bias_data),
            reinterpret_cast<float *>(out_data),
            M, K, N);
        break;

    case LLAISYS_DTYPE_F16:
        linear_kernel<llaisys::fp16_t>(
            reinterpret_cast<const llaisys::fp16_t *>(input_data),
            reinterpret_cast<const llaisys::fp16_t *>(weight_data),
            reinterpret_cast<const llaisys::fp16_t *>(bias_data),
            reinterpret_cast<llaisys::fp16_t *>(out_data),
            M, K, N);
        break;

    case LLAISYS_DTYPE_BF16:
        linear_kernel<llaisys::bf16_t>(
            reinterpret_cast<const llaisys::bf16_t *>(input_data),
            reinterpret_cast<const llaisys::bf16_t *>(weight_data),
            reinterpret_cast<const llaisys::bf16_t *>(bias_data),
            reinterpret_cast<llaisys::bf16_t *>(out_data),
            M, K, N);
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu