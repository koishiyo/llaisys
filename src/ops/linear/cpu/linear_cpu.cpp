#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <omp.h>
template <typename T>
void linear_kernel(const T *input, const T *weight, const T *bias, T *out,
                   size_t M, size_t K, size_t N) {
#pragma omp parallel for
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {

            float sum = 0.0f;

            for (size_t k = 0; k < K; k++) {

                size_t input_idx = m * K + k;
                size_t weight_idx = n * K + k;

                float x_val = llaisys::utils::cast<float>(input[input_idx]);
                float w_val = llaisys::utils::cast<float>(weight[weight_idx]);

                sum += x_val * w_val;
            }

            // 加上偏置
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[n]);
            }

            out[m * N + n] = llaisys::utils::cast<T>(sum);
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
            reinterpret_cast<const float *>(bias_data), // 如果是空指针，转换后还是空指针
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