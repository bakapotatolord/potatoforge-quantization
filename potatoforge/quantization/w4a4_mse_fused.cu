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

__global__ void squared_error_kernel(
    const float* weights,
    const float* scales,
    float* output,
    int columns,
    int qmin,
    int qmax,
    float scale_floor) {
    const int row = blockIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(row) * columns + column;
    const float weight = weights[index];
    float scale = scales[row];
    if (scale < scale_floor) {
        scale = scale_floor;
    }
    float code = nearbyintf(__fdiv_rn(weight, scale));
    if (code < qmin) {
        code = qmin;
    } else if (code > qmax) {
        code = qmax;
    }
    // Preserve eager Torch's separate FP32 operations, including subnormals.
    const float error = __fsub_rn(__fmul_rn(code, scale), weight);
    output[index] = __fmul_rn(error, error);
}

int launch_squared_error(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream,
    int qmin,
    int qmax,
    float scale_floor) {
    if (!weights || !scales || !output || rows <= 0 || rows > 65535 || columns <= 0) {
        return 1;
    }
    const dim3 blocks((columns + kThreads - 1) / kThreads, rows);
    squared_error_kernel<<<blocks, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        weights, scales, output, columns, qmin, qmax, scale_floor);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : 1000 + static_cast<int>(error);
}

// Whole coarse+fine rowwise MSE scale search in one launch. Each block owns
// one row and scores all 34 candidates against it, keeping the row in
// registers when it fits. The per-candidate score reproduces
// candidate_mse_kernel's arithmetic and reduction order, and the multiplier
// arithmetic reproduces the eager Torch search, so the selected multipliers
// match the per-candidate path bit for bit.
constexpr int kCoarseFirst = 2;
constexpr int kCoarseLast = 20;
constexpr int kFineSteps = 16;
constexpr int kMaxValuesPerThread = 64;
// Quotients this close to a rounding midpoint take the exact IEEE division.
constexpr float kMidpointMargin = 1.0e-4f;
// Keeps 1 / scale a normal float so the reciprocal error bound holds.
constexpr float kFastScaleLimit = 1.0e30f;

// Adding and subtracting 1.5 * 2^23 rounds |x| < 2^22 to the nearest integer,
// ties to even, like nearbyintf but on the full-rate FP32 adder. Larger
// magnitudes stay beyond every qmin/qmax and clamp identically.
constexpr float kRoundingMagic = 12582912.0f;
// Up to this many register-cached values per thread, the fast pass runs
// branch-free and a flagged thread recomputes its whole sum exactly.
constexpr int kBranchFreeMaxValues = 16;

// The fast path replaces nearbyintf(weight / scale) with weight * (1 / scale)
// rounded on the FP32 adder. That product is within ~2^-23 relative of the
// true quotient, below kMidpointMargin for |quotient| < 128; larger quotients
// clamp to the same qmin/qmax either way. Only quotients near a rounding
// midpoint can round differently: they raise near_midpoint, and the caller
// then recomputes that thread's sum with the exact division.
__device__ __forceinline__ float accumulate_code_error(
    float sum,
    float weight,
    float code,
    float scale,
    int qmin,
    int qmax) {
    if (code < static_cast<float>(qmin)) {
        code = static_cast<float>(qmin);
    } else if (code > static_cast<float>(qmax)) {
        code = static_cast<float>(qmax);
    }
    const float error = weight - code * scale;
    return sum + error * error;
}

__device__ __forceinline__ float exact_code(float weight, float scale) {
    return nearbyintf(weight / scale);
}

__device__ __forceinline__ float fast_code(
    float weight,
    float inverse_scale,
    bool& near_midpoint) {
    const float approx = __fmul_rn(weight, inverse_scale);
    const float rounded = __fsub_rn(__fadd_rn(approx, kRoundingMagic), kRoundingMagic);
    near_midpoint |= fabsf(__fsub_rn(approx, rounded)) > 0.5f - kMidpointMargin;
    return rounded;
}

template <int kValuesPerThread>
__device__ __forceinline__ float candidate_thread_sum(
    const float* row_values,
    const float (&values)[kValuesPerThread > 0 ? kValuesPerThread : 1],
    int columns,
    float scale,
    int qmin,
    int qmax) {
    const float inverse_scale = __frcp_rn(scale);
    bool exact = !(scale < kFastScaleLimit);
    float sum = 0.0f;
    if constexpr (kValuesPerThread > kBranchFreeMaxValues) {
        // Wide register-cached rows: a second unrolled pass would cost too many
        // registers, so near-midpoint values divide exactly in place.
#pragma unroll
        for (int index = 0; index < kValuesPerThread; ++index) {
            if (threadIdx.x + index * kThreads < columns) {
                const float weight = values[index];
                bool near_midpoint = exact;
                float code = exact ? 0.0f : fast_code(weight, inverse_scale, near_midpoint);
                if (near_midpoint) {
                    code = exact_code(weight, scale);
                }
                sum = accumulate_code_error(sum, weight, code, scale, qmin, qmax);
            }
        }
        return sum;
    }
    if (!exact) {
        bool near_midpoint = false;
        if constexpr (kValuesPerThread > 0) {
#pragma unroll
            for (int index = 0; index < kValuesPerThread; ++index) {
                if (threadIdx.x + index * kThreads < columns) {
                    const float weight = values[index];
                    sum = accumulate_code_error(
                        sum, weight, fast_code(weight, inverse_scale, near_midpoint),
                        scale, qmin, qmax);
                }
            }
        } else {
            for (int column = threadIdx.x; column < columns; column += kThreads) {
                const float weight = row_values[column];
                sum = accumulate_code_error(
                    sum, weight, fast_code(weight, inverse_scale, near_midpoint),
                    scale, qmin, qmax);
            }
        }
        exact = near_midpoint;
    }
    if (exact) {
        // Rare: re-read this thread's columns (same order) from global memory
        // so the register-cached loop above is not duplicated.
        sum = 0.0f;
#pragma unroll 1
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            const float weight = row_values[column];
            sum = accumulate_code_error(
                sum, weight, exact_code(weight, scale), scale, qmin, qmax);
        }
    }
    return sum;
}

// Reduces candidate_count rows of per-thread sums with candidate_mse_kernel's
// pairing (partial[t] += partial[t + stride], stride 128..1) for each
// candidate, but with one barrier per level for all candidates together.
__device__ __forceinline__ void reduce_candidate_partials(
    float (*partials)[kThreads],
    int candidate_count) {
    __syncthreads();
    for (int stride = kThreads / 2, shift = 7; stride > 0; stride /= 2, --shift) {
        for (int item = threadIdx.x; item < candidate_count * stride; item += kThreads) {
            const int candidate = item >> shift;
            const int lane = item & (stride - 1);
            partials[candidate][lane] += partials[candidate][lane + stride];
        }
        __syncthreads();
    }
}

constexpr int kCoarseCount = kCoarseLast - kCoarseFirst + 1;
constexpr int kFineCount = kFineSteps - 1;
static_assert(kThreads == 256, "reduce_candidate_partials assumes 256 threads");
static_assert(kFineCount <= kCoarseCount, "fine candidates share the coarse buffer");

__device__ __forceinline__ float candidate_scale(
    float base_scale,
    float multiplier,
    float scale_floor) {
    const float scale = __fmul_rn(base_scale, multiplier);
    return scale < scale_floor ? scale_floor : scale;
}

constexpr int kCoarseStage = 0;
constexpr int kFineStage = 1;

// Matches Torch's FP32 cast of the Python double index / 20.
__device__ __forceinline__ float coarse_multiplier(int candidate) {
    return static_cast<float>((kCoarseFirst + candidate) / 20.0);
}

// Matches the Torch FP32 expression lower + (upper - lower) * (step / 16).
__device__ __forceinline__ float fine_multiplier(float lower, float span, int candidate) {
    const float fraction =
        static_cast<float>(candidate + 1) / static_cast<float>(kFineSteps);
    return __fadd_rn(lower, __fmul_rn(span, fraction));
}

template <int kValuesPerThread>
__global__ void __launch_bounds__(kThreads) mse_search_kernel(
    const float* weights,
    const float* base_scales,
    float* best_mses,
    float* best_multipliers,
    int columns,
    int qmin,
    int qmax,
    float scale_floor,
    int stage) {
    __shared__ float partials[kCoarseCount][kThreads];

    const int row = blockIdx.x;
    const float* row_values =
        weights + static_cast<std::size_t>(row) * static_cast<std::size_t>(columns);
    float values[kValuesPerThread > 0 ? kValuesPerThread : 1];
    if constexpr (kValuesPerThread > 0) {
#pragma unroll
        for (int index = 0; index < kValuesPerThread; ++index) {
            const int column = threadIdx.x + index * kThreads;
            values[index] = column < columns ? row_values[column] : 0.0f;
        }
    }

    const float base_scale = base_scales[row];
    const float column_count = static_cast<float>(columns);

    if (stage == kCoarseStage) {
        for (int candidate = 0; candidate < kCoarseCount; ++candidate) {
            partials[candidate][threadIdx.x] = candidate_thread_sum<kValuesPerThread>(
                row_values, values, columns,
                candidate_scale(base_scale, coarse_multiplier(candidate), scale_floor),
                qmin, qmax);
        }
        reduce_candidate_partials(partials, kCoarseCount);

        if (threadIdx.x == 0) {
            // Sequential strict-less selection, as in the per-candidate Torch loop.
            float best_mse = INFINITY;
            float best_multiplier = 1.0f;
            for (int candidate = 0; candidate < kCoarseCount; ++candidate) {
                const float mse = partials[candidate][0] / column_count;
                if (mse < best_mse) {
                    best_mse = mse;
                    best_multiplier = coarse_multiplier(candidate);
                }
            }
            best_mses[row] = best_mse;
            best_multipliers[row] = best_multiplier;
        }
        return;
    }

    float best_mse = best_mses[row];
    float best_multiplier = best_multipliers[row];
    const float lower = fmaxf(__fsub_rn(best_multiplier, 0.05f), 0.10f);
    const float upper = fminf(__fadd_rn(best_multiplier, 0.05f), 1.0f);
    const float span = __fsub_rn(upper, lower);
    for (int candidate = 0; candidate < kFineCount; ++candidate) {
        partials[candidate][threadIdx.x] = candidate_thread_sum<kValuesPerThread>(
            row_values, values, columns,
            candidate_scale(base_scale, fine_multiplier(lower, span, candidate), scale_floor),
            qmin, qmax);
    }
    reduce_candidate_partials(partials, kFineCount);

    if (threadIdx.x == 0) {
        for (int candidate = 0; candidate < kFineCount; ++candidate) {
            const float mse = partials[candidate][0] / column_count;
            if (mse < best_mse) {
                best_mse = mse;
                best_multiplier = fine_multiplier(lower, span, candidate);
            }
        }
        best_mses[row] = best_mse;
        best_multipliers[row] = best_multiplier;
    }
}

template <int kValuesPerThread>
void launch_search(
    const float* weights,
    const float* base_scales,
    float* best_mses,
    float* best_multipliers,
    int rows,
    int columns,
    int qmin,
    int qmax,
    float scale_floor,
    int stage,
    cudaStream_t stream) {
    mse_search_kernel<kValuesPerThread><<<rows, kThreads, 0, stream>>>(
        weights, base_scales, best_mses, best_multipliers, columns, qmin, qmax,
        scale_floor, stage);
}

}  // namespace

// stage 0 scores the coarse grid and writes each row's best MSE and
// multiplier; stage 1 reads them, scores the fine grid around the multiplier
// and updates both. row_cache_mode -1 keeps rows of up to
// kMaxValuesPerThread * kThreads columns in registers; 0 always reads the row
// from global memory.
extern "C" PF_EXPORT int potatoforge_rowwise_mse_search(
    const float* weights,
    const float* base_scales,
    float* best_mses,
    float* best_multipliers,
    int rows,
    int columns,
    int qmin,
    int qmax,
    float scale_floor,
    int stage,
    int row_cache_mode,
    void* stream) {
    if (weights == nullptr || base_scales == nullptr || best_mses == nullptr ||
        best_multipliers == nullptr || rows <= 0 || columns <= 0 || qmin >= qmax ||
        qmin < -128 || qmax > 127 || (stage != kCoarseStage && stage != kFineStage) ||
        (row_cache_mode != -1 && row_cache_mode != 0)) {
        return 1;
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int per_thread = (columns + kThreads - 1) / kThreads;
    const bool cache = row_cache_mode == -1 && per_thread <= kMaxValuesPerThread;
#define PF_LAUNCH_SEARCH(count) \
    launch_search<count>(weights, base_scales, best_mses, best_multipliers, rows, \
                         columns, qmin, qmax, scale_floor, stage, cuda_stream)
    if (!cache) {
        PF_LAUNCH_SEARCH(0);
    } else if (per_thread <= 4) {
        PF_LAUNCH_SEARCH(4);
    } else if (per_thread <= 8) {
        PF_LAUNCH_SEARCH(8);
    } else if (per_thread <= 16) {
        PF_LAUNCH_SEARCH(16);
    } else if (per_thread <= 32) {
        PF_LAUNCH_SEARCH(32);
    } else if (per_thread <= 48) {
        PF_LAUNCH_SEARCH(48);
    } else {
        PF_LAUNCH_SEARCH(64);
    }
#undef PF_LAUNCH_SEARCH
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : 1000 + static_cast<int>(error);
}

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

extern "C" PF_EXPORT int potatoforge_int7_convrot_mse_squared_error(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream) {
    return launch_squared_error(weights, scales, output, rows, columns, stream, -63, 63, 0.0f);
}

extern "C" PF_EXPORT int potatoforge_int8_convrot_mse_squared_error(
    const float* weights,
    const float* scales,
    float* output,
    int rows,
    int columns,
    void* stream) {
    return launch_squared_error(weights, scales, output, rows, columns, stream, -128, 127, 1.0e-30f);
}
