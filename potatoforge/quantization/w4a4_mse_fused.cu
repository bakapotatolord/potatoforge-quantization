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
constexpr int kThreads = 256;

__global__ void candidate_mse_kernel(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    float scale_floor,
    int qmin,
    int qmax) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float scale = scales[row];
    if (scale < scale_floor) {
        scale = scale_floor;
    }

    float sum = 0.0f;
    const std::size_t row_offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(columns);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float weight = weights[row_offset + column];
        float code = nearbyintf(weight / scale);
        if (code < static_cast<float>(qmin)) {
            code = static_cast<float>(qmin);
        } else if (code > static_cast<float>(qmax)) {
            code = static_cast<float>(qmax);
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

int launch_candidate(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream,
    float scale_floor,
    int qmin,
    int qmax) {
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
        columns,
        scale_floor,
        qmin,
        qmax);

    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return 1000 + static_cast<int>(launch_error);
    }
    return 0;
}

}  // namespace

extern "C" PF_EXPORT int potatoforge_w4a4_mse_candidate(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream) {
    return launch_candidate(
        weights,
        scales,
        output,
        rows,
        columns,
        stream,
        kScaleFloor,
        -7,
        7);
}

extern "C" PF_EXPORT int potatoforge_int5_convrot_mse_candidate(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream) {
    return launch_candidate(
        weights,
        scales,
        output,
        rows,
        columns,
        stream,
        0.0f,
        -15,
        15);
}

extern "C" PF_EXPORT int potatoforge_int6_convrot_mse_candidate(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream) {
    return launch_candidate(
        weights,
        scales,
        output,
        rows,
        columns,
        stream,
        0.0f,
        -31,
        31);
}

extern "C" PF_EXPORT int potatoforge_int7_convrot_mse_candidate(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream) {
    return launch_candidate(
        weights,
        scales,
        output,
        rows,
        columns,
        stream,
        0.0f,
        -63,
        63);
}
