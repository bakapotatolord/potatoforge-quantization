#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#if defined(_WIN32)
#define PF_EXPORT __declspec(dllexport)
#else
#define PF_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr float kScaleFloor = 1.0e-10f;
constexpr int kQMin = -7;
constexpr int kQMax = 7;
constexpr int kThreads = 256;

__global__ void candidate_mse_kernel(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float scale = scales[row];
    if (scale < kScaleFloor) {
        scale = kScaleFloor;
    }

    float sum = 0.0f;
    const std::size_t row_offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(columns);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float weight = weights[row_offset + column];
        float code = nearbyintf(weight / scale);
        if (code < static_cast<float>(kQMin)) {
            code = static_cast<float>(kQMin);
        } else if (code > static_cast<float>(kQMax)) {
            code = static_cast<float>(kQMax);
        }
        const float error = weight - code * scale;
        sum += error * error;
    }

    __shared__ float partials[kThreads];
    partials[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = partials[0] / static_cast<float>(columns);
    }
}

}  // namespace

extern "C" PF_EXPORT int potatoforge_w4a4_mse_candidate(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream) {
    if (weights == nullptr || scales == nullptr || output == nullptr ||
        rows <= 0 || columns <= 0) {
        return 1;
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    candidate_mse_kernel<<<rows, kThreads, 0, cuda_stream>>>(
        weights,
        scales,
        output,
        rows,
        columns);

    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return 1000 + static_cast<int>(launch_error);
    }
    return 0;
}
