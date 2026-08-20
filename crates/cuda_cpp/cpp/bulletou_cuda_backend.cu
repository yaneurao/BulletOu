#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

thread_local std::string g_last_error;

struct BulletOuCudaCppContext {
    int device = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    int* sfnn_inverse_counts = nullptr;
    size_t sfnn_inverse_counts_len = 0;
    int* sfnn_inverse_offsets = nullptr;
    size_t sfnn_inverse_offsets_len = 0;
    int* sfnn_inverse_block_sums = nullptr;
    size_t sfnn_inverse_block_sums_len = 0;
    int* sfnn_inverse_block_offsets = nullptr;
    size_t sfnn_inverse_block_offsets_len = 0;
    int* sfnn_inverse_write_counters = nullptr;
    size_t sfnn_inverse_write_counters_len = 0;
    int* sfnn_inverse_positions = nullptr;
    size_t sfnn_inverse_positions_len = 0;
};

struct BulletOuCudaCppF32Buffer {
    int device = 0;
    size_t len = 0;
    float* ptr = nullptr;
};

struct BulletOuCudaCppI32Buffer {
    int device = 0;
    size_t len = 0;
    int* ptr = nullptr;
};

struct BulletOuCudaCppPinnedF32Buffer {
    int device = 0;
    size_t len = 0;
    float* ptr = nullptr;
};

struct BulletOuCudaCppPinnedI32Buffer {
    int device = 0;
    size_t len = 0;
    int* ptr = nullptr;
};

struct BulletOuCudaCppEvent {
    int device = 0;
    cudaEvent_t event = nullptr;
};

struct BulletOuCudaCppGraphExec {
    int device = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};

void set_error(const char* context, cudaError_t status) {
    char buffer[512];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%s: %s (%d)",
        context,
        cudaGetErrorString(status),
        static_cast<int>(status));
    g_last_error = buffer;
}

void set_error_message(const char* message) {
    g_last_error = message;
}

int ok() {
    g_last_error.clear();
    return 0;
}

int fail(const char* context, cudaError_t status) {
    set_error(context, status);
    return static_cast<int>(status) == 0 ? -1 : static_cast<int>(status);
}

const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "alloc failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "arch mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "internal error";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "not supported";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "license error";
        default:
            return "unknown";
    }
}

int fail_blas(const char* context, cublasStatus_t status) {
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), "%s: cuBLAS %s (%d)", context, cublas_status_string(status), static_cast<int>(status));
    set_error_message(buffer);
    return static_cast<int>(status) == 0 ? -1 : static_cast<int>(status);
}

int fail_message(const char* message) {
    set_error_message(message);
    return -1;
}

int check_kernel_launch(const char* launch_label);
int block_count_1d(size_t len, int threads, int* blocks, const char* label);

template <typename T>
int checked_malloc(T** ptr, size_t len, const char* label) {
    if (len == 0) {
        *ptr = nullptr;
        return 0;
    }
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(ptr), len * sizeof(T));
    if (status != cudaSuccess) {
        return fail(label, status);
    }
    return 0;
}

int ensure_i32_scratch(int** ptr, size_t* capacity, size_t len, const char* label) {
    if (len <= *capacity) {
        return 0;
    }
    if (*ptr != nullptr) {
        cudaError_t free_status = cudaFree(*ptr);
        if (free_status != cudaSuccess) {
            return fail(label, free_status);
        }
        *ptr = nullptr;
        *capacity = 0;
    }
    if (checked_malloc(ptr, len, label) != 0) {
        return -1;
    }
    *capacity = len;
    return 0;
}

int ensure_f32_scratch(float** ptr, size_t* capacity, size_t len, const char* label) {
    if (len <= *capacity) {
        return 0;
    }
    if (*ptr != nullptr) {
        cudaError_t free_status = cudaFree(*ptr);
        if (free_status != cudaSuccess) {
            return fail(label, free_status);
        }
        *ptr = nullptr;
        *capacity = 0;
    }
    if (checked_malloc(ptr, len, label) != 0) {
        return -1;
    }
    *capacity = len;
    return 0;
}

int free_i32_scratch(int*& ptr, size_t& capacity, const char* label) {
    if (ptr == nullptr) {
        capacity = 0;
        return 0;
    }
    cudaError_t status = cudaFree(ptr);
    ptr = nullptr;
    capacity = 0;
    if (status != cudaSuccess) {
        return fail(label, status);
    }
    return 0;
}

int free_f32_scratch(float*& ptr, size_t& capacity, const char* label) {
    if (ptr == nullptr) {
        capacity = 0;
        return 0;
    }
    cudaError_t status = cudaFree(ptr);
    ptr = nullptr;
    capacity = 0;
    if (status != cudaSuccess) {
        return fail(label, status);
    }
    return 0;
}

template <typename T>
int checked_host_malloc(T** ptr, size_t len, const char* label) {
    if (len == 0) {
        *ptr = nullptr;
        return 0;
    }
    cudaError_t status = cudaMallocHost(reinterpret_cast<void**>(ptr), len * sizeof(T));
    if (status != cudaSuccess) {
        return fail(label, status);
    }
    return 0;
}

template <typename T>
int copy_h2d(T* dst, const T* src, size_t len, const char* label) {
    if (len == 0) {
        return 0;
    }
    cudaError_t status = cudaMemcpy(dst, src, len * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return fail(label, status);
    }
    return 0;
}

template <typename T>
int copy_d2h(T* dst, const T* src, size_t len, const char* label) {
    if (len == 0) {
        return 0;
    }
    cudaError_t status = cudaMemcpy(dst, src, len * sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        return fail(label, status);
    }
    return 0;
}

struct DeviceFloats {
    float* ptr = nullptr;

    ~DeviceFloats() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    int allocate(size_t len, const char* label) {
        return checked_malloc(&ptr, len, label);
    }
};

int warmup_context(BulletOuCudaCppContext* ctx) {
    DeviceFloats a;
    DeviceFloats b;
    DeviceFloats c;
    if (a.allocate(1, "cudaMalloc warmup a") != 0 || b.allocate(1, "cudaMalloc warmup b") != 0 ||
        c.allocate(1, "cudaMalloc warmup c") != 0) {
        return -1;
    }

    cudaError_t status = cudaMemsetAsync(a.ptr, 0, sizeof(float), ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemsetAsync warmup a", status);
    }
    status = cudaMemsetAsync(b.ptr, 0, sizeof(float), ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemsetAsync warmup b", status);
    }
    status = cudaMemsetAsync(c.ptr, 0, sizeof(float), ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemsetAsync warmup c", status);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t blas_status =
        cublasSgemm(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha, a.ptr, 1, b.ptr, 1, &beta, c.ptr, 1);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return fail_blas("cublasSgemm warmup", blas_status);
    }

    status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaStreamSynchronize warmup", status);
    }
    return 0;
}

__global__ void axpy_kernel(size_t len, float a, const float* x, const float* y, float* out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    out[idx] = a * x[idx] + y[idx];
}

__global__ void fill_f32_kernel(size_t len, float value, float* out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    out[idx] = value;
}

__device__ __forceinline__ void radam_update_one(
    float grad,
    float learning_rate,
    float step_size,
    int use_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    float* weight,
    float* momentum,
    float* velocity) {
    float rate = learning_rate * step_size;
    float w = *weight * (1.0f - decay * rate);
    float m = beta1 * *momentum + (1.0f - beta1) * grad;
    float v = beta2 * *velocity + (1.0f - beta2) * grad * grad;

    float update = m;
    if (use_denom != 0) {
        update /= sqrtf(v) + epsilon;
    }
    w -= rate * update;
    w = fminf(fmaxf(w, min_weight), max_weight);

    *weight = w;
    *momentum = m;
    *velocity = v;
}

__global__ void radam_update_reset_gradients_kernel(
    float* gradients,
    float* weights,
    float* momentum,
    float* velocity,
    size_t len,
    float gradient_factor,
    float learning_rate,
    float step_size,
    int use_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    float grad = gradient_factor * gradients[idx];
    float weight = weights[idx];
    float m = momentum[idx];
    float v = velocity[idx];
    radam_update_one(
        grad, learning_rate, step_size, use_denom, decay, beta1, beta2, epsilon, min_weight, max_weight, &weight, &m, &v);

    gradients[idx] = 0.0f;
    weights[idx] = weight;
    momentum[idx] = m;
    velocity[idx] = v;
}

__global__ void radam_update_reset_gradients_vec4_kernel(
    float* gradients,
    float* weights,
    float* momentum,
    float* velocity,
    size_t len4,
    float gradient_factor,
    float learning_rate,
    float step_size,
    int use_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len4) {
        return;
    }

    const float4 gradient4 = reinterpret_cast<const float4*>(gradients)[idx];
    float4 weight4 = reinterpret_cast<const float4*>(weights)[idx];
    float4 momentum4 = reinterpret_cast<const float4*>(momentum)[idx];
    float4 velocity4 = reinterpret_cast<const float4*>(velocity)[idx];

    radam_update_one(
        gradient_factor * gradient4.x,
        learning_rate,
        step_size,
        use_denom,
        decay,
        beta1,
        beta2,
        epsilon,
        min_weight,
        max_weight,
        &weight4.x,
        &momentum4.x,
        &velocity4.x);
    radam_update_one(
        gradient_factor * gradient4.y,
        learning_rate,
        step_size,
        use_denom,
        decay,
        beta1,
        beta2,
        epsilon,
        min_weight,
        max_weight,
        &weight4.y,
        &momentum4.y,
        &velocity4.y);
    radam_update_one(
        gradient_factor * gradient4.z,
        learning_rate,
        step_size,
        use_denom,
        decay,
        beta1,
        beta2,
        epsilon,
        min_weight,
        max_weight,
        &weight4.z,
        &momentum4.z,
        &velocity4.z);
    radam_update_one(
        gradient_factor * gradient4.w,
        learning_rate,
        step_size,
        use_denom,
        decay,
        beta1,
        beta2,
        epsilon,
        min_weight,
        max_weight,
        &weight4.w,
        &momentum4.w,
        &velocity4.w);

    reinterpret_cast<float4*>(gradients)[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    reinterpret_cast<float4*>(weights)[idx] = weight4;
    reinterpret_cast<float4*>(momentum)[idx] = momentum4;
    reinterpret_cast<float4*>(velocity)[idx] = velocity4;
}

__global__ void ranger_lookahead_kernel(float* weights, float* slow_params, size_t len, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    float next = alpha * weights[idx] + (1.0f - alpha) * slow_params[idx];
    weights[idx] = next;
    slow_params[idx] = next;
}

__global__ void ranger_lookahead_vec4_kernel(float* weights, float* slow_params, size_t len4, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len4) {
        return;
    }

    float4 weight4 = reinterpret_cast<const float4*>(weights)[idx];
    const float4 slow4 = reinterpret_cast<const float4*>(slow_params)[idx];
    weight4.x = alpha * weight4.x + (1.0f - alpha) * slow4.x;
    weight4.y = alpha * weight4.y + (1.0f - alpha) * slow4.y;
    weight4.z = alpha * weight4.z + (1.0f - alpha) * slow4.z;
    weight4.w = alpha * weight4.w + (1.0f - alpha) * slow4.w;
    reinterpret_cast<float4*>(weights)[idx] = weight4;
    reinterpret_cast<float4*>(slow_params)[idx] = weight4;
}

__global__ void radam_update_reset_gradients_stacked_dirty_kernel(
    const int* dirty_buckets,
    size_t dirty_count,
    size_t stride,
    size_t len,
    float* gradients,
    float* weights,
    float* momentum,
    float* velocity,
    float gradient_factor,
    float learning_rate,
    float step_size,
    int use_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dirty_count * stride;
    if (tid >= total) {
        return;
    }

    size_t bucket_offset = tid / stride;
    size_t in_bucket = tid - bucket_offset * stride;
    int bucket = dirty_buckets[bucket_offset];
    if (bucket < 0) {
        return;
    }
    size_t idx = static_cast<size_t>(bucket) * stride + in_bucket;
    if (idx >= len) {
        return;
    }

    float grad = gradient_factor * gradients[idx];
    float weight = weights[idx];
    float m = momentum[idx];
    float v = velocity[idx];
    radam_update_one(
        grad, learning_rate, step_size, use_denom, decay, beta1, beta2, epsilon, min_weight, max_weight, &weight, &m, &v);

    gradients[idx] = 0.0f;
    weights[idx] = weight;
    momentum[idx] = m;
    velocity[idx] = v;
}

__global__ void radam_update_reset_gradients_stacked_dirty_vec4_kernel(
    const int* dirty_buckets,
    size_t dirty_count,
    size_t stride4,
    size_t len4,
    float* gradients,
    float* weights,
    float* momentum,
    float* velocity,
    float gradient_factor,
    float learning_rate,
    float step_size,
    int use_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dirty_count * stride4;
    if (tid >= total) {
        return;
    }

    size_t bucket_offset = tid / stride4;
    size_t in_bucket = tid - bucket_offset * stride4;
    int bucket = dirty_buckets[bucket_offset];
    if (bucket < 0) {
        return;
    }
    size_t idx = static_cast<size_t>(bucket) * stride4 + in_bucket;
    if (idx >= len4) {
        return;
    }

    const float4 gradient4 = reinterpret_cast<const float4*>(gradients)[idx];
    float4 weight4 = reinterpret_cast<const float4*>(weights)[idx];
    float4 momentum4 = reinterpret_cast<const float4*>(momentum)[idx];
    float4 velocity4 = reinterpret_cast<const float4*>(velocity)[idx];

    radam_update_one(
        gradient_factor * gradient4.x,
        learning_rate,
        step_size,
        use_denom,
        decay,
        beta1,
        beta2,
        epsilon,
        min_weight,
        max_weight,
        &weight4.x,
        &momentum4.x,
        &velocity4.x);
    radam_update_one(
        gradient_factor * gradient4.y,
        learning_rate,
        step_size,
        use_denom,
        decay,
        beta1,
        beta2,
        epsilon,
        min_weight,
        max_weight,
        &weight4.y,
        &momentum4.y,
        &velocity4.y);
    radam_update_one(
        gradient_factor * gradient4.z,
        learning_rate,
        step_size,
        use_denom,
        decay,
        beta1,
        beta2,
        epsilon,
        min_weight,
        max_weight,
        &weight4.z,
        &momentum4.z,
        &velocity4.z);
    radam_update_one(
        gradient_factor * gradient4.w,
        learning_rate,
        step_size,
        use_denom,
        decay,
        beta1,
        beta2,
        epsilon,
        min_weight,
        max_weight,
        &weight4.w,
        &momentum4.w,
        &velocity4.w);

    reinterpret_cast<float4*>(gradients)[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    reinterpret_cast<float4*>(weights)[idx] = weight4;
    reinterpret_cast<float4*>(momentum)[idx] = momentum4;
    reinterpret_cast<float4*>(velocity)[idx] = velocity4;
}

__global__ void clear_gradients_stacked_dirty_kernel(
    const int* dirty_buckets,
    size_t dirty_count,
    size_t stride,
    size_t len,
    float* gradients) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dirty_count * stride;
    if (tid >= total) {
        return;
    }

    size_t bucket_offset = tid / stride;
    size_t in_bucket = tid - bucket_offset * stride;
    int bucket = dirty_buckets[bucket_offset];
    if (bucket < 0) {
        return;
    }
    size_t idx = static_cast<size_t>(bucket) * stride + in_bucket;
    if (idx < len) {
        gradients[idx] = 0.0f;
    }
}

__global__ void clear_gradients_stacked_dirty_vec4_kernel(
    const int* dirty_buckets,
    size_t dirty_count,
    size_t stride4,
    size_t len4,
    float* gradients) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dirty_count * stride4;
    if (tid >= total) {
        return;
    }

    size_t bucket_offset = tid / stride4;
    size_t in_bucket = tid - bucket_offset * stride4;
    int bucket = dirty_buckets[bucket_offset];
    if (bucket < 0) {
        return;
    }
    size_t idx = static_cast<size_t>(bucket) * stride4 + in_bucket;
    if (idx < len4) {
        reinterpret_cast<float4*>(gradients)[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__global__ void ranger_lookahead_stacked_dirty_kernel(
    const int* dirty_buckets,
    size_t dirty_count,
    size_t stride,
    size_t len,
    float* weights,
    float* slow_params,
    float alpha) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dirty_count * stride;
    if (tid >= total) {
        return;
    }

    size_t bucket_offset = tid / stride;
    size_t in_bucket = tid - bucket_offset * stride;
    int bucket = dirty_buckets[bucket_offset];
    if (bucket < 0) {
        return;
    }
    size_t idx = static_cast<size_t>(bucket) * stride + in_bucket;
    if (idx >= len) {
        return;
    }

    float next = alpha * weights[idx] + (1.0f - alpha) * slow_params[idx];
    weights[idx] = next;
    slow_params[idx] = next;
}

__global__ void ranger_lookahead_stacked_dirty_vec4_kernel(
    const int* dirty_buckets,
    size_t dirty_count,
    size_t stride4,
    size_t len4,
    float* weights,
    float* slow_params,
    float alpha) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dirty_count * stride4;
    if (tid >= total) {
        return;
    }

    size_t bucket_offset = tid / stride4;
    size_t in_bucket = tid - bucket_offset * stride4;
    int bucket = dirty_buckets[bucket_offset];
    if (bucket < 0) {
        return;
    }
    size_t idx = static_cast<size_t>(bucket) * stride4 + in_bucket;
    if (idx >= len4) {
        return;
    }

    float4 weight4 = reinterpret_cast<const float4*>(weights)[idx];
    const float4 slow4 = reinterpret_cast<const float4*>(slow_params)[idx];
    weight4.x = alpha * weight4.x + (1.0f - alpha) * slow4.x;
    weight4.y = alpha * weight4.y + (1.0f - alpha) * slow4.y;
    weight4.z = alpha * weight4.z + (1.0f - alpha) * slow4.z;
    weight4.w = alpha * weight4.w + (1.0f - alpha) * slow4.w;
    reinterpret_cast<float4*>(weights)[idx] = weight4;
    reinterpret_cast<float4*>(slow_params)[idx] = weight4;
}

__device__ float crelu(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

constexpr size_t NNUE_HALFKP_BASE_INPUT_SIZE = 125388;
constexpr size_t NNUE_HALFKP_PIECE_INPUTS = 1548;
constexpr size_t NNUE_HALFKP_FACTORIZED_INPUT_SIZE = NNUE_HALFKP_BASE_INPUT_SIZE + NNUE_HALFKP_PIECE_INPUTS;

__host__ __device__ size_t nnue_l0w_len_for_shape(size_t input_size, size_t rows) {
    return input_size * rows;
}

__device__ bool nnue_halfkp_factorized_feature(size_t feature, size_t input_size, size_t* out_base_feature, size_t* out_virtual_feature) {
    if (input_size == NNUE_HALFKP_FACTORIZED_INPUT_SIZE && feature < NNUE_HALFKP_BASE_INPUT_SIZE) {
        *out_base_feature = NNUE_HALFKP_PIECE_INPUTS + feature;
        *out_virtual_feature = feature % NNUE_HALFKP_PIECE_INPUTS;
        return true;
    }
    return false;
}

__global__ void nnue_sparse_l0_crelu_kernel(
    const int* indices,
    const float* weights,
    const float* bias,
    float* output,
    size_t batch,
    size_t max_active,
    size_t input_size,
    size_t rows) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * rows;
    if (tid >= total) {
        return;
    }

    size_t row = tid % rows;
    size_t sample = tid / rows;
    float sum = bias[row];
    size_t sparse_base = sample * max_active;
    for (size_t slot = 0; slot < max_active; ++slot) {
        int feature = indices[sparse_base + slot];
        if (feature >= 0 && static_cast<size_t>(feature) < input_size) {
            size_t base_feature = 0;
            size_t virtual_feature = 0;
            if (nnue_halfkp_factorized_feature(static_cast<size_t>(feature), input_size, &base_feature, &virtual_feature)) {
                sum += weights[base_feature * rows + row] + weights[virtual_feature * rows + row];
            } else {
                size_t weight_base = static_cast<size_t>(feature) * rows;
                sum += weights[weight_base + row];
            }
        }
    }
    output[tid] = crelu(sum);
}

__global__ void crelu_inplace_kernel(float* values, size_t len) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    values[tid] = crelu(values[tid]);
}

__global__ void nnue_concat_l0_kernel(
    const float* stm_l0,
    const float* nstm_l0,
    float* combined,
    size_t batch,
    size_t rows) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t combined_rows = rows * 2;
    size_t total = batch * combined_rows;
    if (tid >= total) {
        return;
    }

    size_t col = tid % combined_rows;
    size_t sample = tid / combined_rows;
    size_t src = sample * rows + (col % rows);
    combined[tid] = col < rows ? stm_l0[src] : nstm_l0[src];
}

__global__ void nnue_dense_crelu_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * output_dim;
    if (tid >= total) {
        return;
    }

    size_t out_col = tid % output_dim;
    size_t sample = tid / output_dim;
    size_t input_base = sample * input_dim;
    float sum = bias[out_col];
    for (size_t in_col = 0; in_col < input_dim; ++in_col) {
        sum += input[input_base + in_col] * weights[in_col * output_dim + out_col];
    }
    output[tid] = crelu(sum);
}

__global__ void nnue_dense_output_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t batch,
    size_t input_dim) {
    size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) {
        return;
    }

    size_t input_base = sample * input_dim;
    float sum = bias[0];
    for (size_t idx = 0; idx < input_dim; ++idx) {
        sum += input[input_base + idx] * weights[idx];
    }
    output[sample] = sum;
}

__global__ void dense_add_bias_kernel(
    float* output,
    const float* bias,
    size_t batch,
    size_t output_dim,
    int apply_crelu) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * output_dim;
    if (tid >= total) {
        return;
    }
    size_t out_col = tid % output_dim;
    float value = output[tid] + bias[out_col];
    output[tid] = apply_crelu != 0 ? crelu(value) : value;
}

constexpr size_t SFNN_HALFKA2_BASE_INPUT_SIZE = 131949;
constexpr size_t SFNN_HALFKA2_PIECE_INPUTS = 1629;
constexpr size_t SFNN_HALFKA2_FACTORIZED_INPUT_SIZE = SFNN_HALFKA2_BASE_INPUT_SIZE + SFNN_HALFKA2_PIECE_INPUTS;
constexpr float SFNN_PAIRWISE_SCALE = 127.0f / 128.0f;

bool sfnn_is_grouped_l1_shape(size_t l1_group_count) {
    return l1_group_count > 1;
}

bool sfnn_is_common_shard_l1_shape(size_t l1_common_size, size_t l1_shard_size) {
    return l1_common_size != 0 || l1_shard_size != 0;
}

__host__ __device__ inline size_t sfnn_l1_out_for_shape(size_t l1_hidden, int l1_skip) {
    return l1_hidden + (l1_skip != 0 ? 1 : 0);
}

size_t sfnn_l1w_len_for_shape(
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size) {
    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    if (sfnn_is_common_shard_l1_shape(l1_common_size, l1_shard_size)) {
        (void)ft_size;
        return num_stacks * l1_out * (l1_common_size + l1_shard_size);
    }
    if (sfnn_is_grouped_l1_shape(l1_group_count)) {
        return num_stacks * l1_group_count * (l1_out / l1_group_count) * (ft_size / l1_group_count);
    }
    return num_stacks * l1_out * ft_size;
}

__device__ bool sfnn_factorized_virtual_feature(size_t feature, size_t input_size, size_t* out_feature) {
    if (input_size == SFNN_HALFKA2_FACTORIZED_INPUT_SIZE && feature < SFNN_HALFKA2_BASE_INPUT_SIZE) {
        *out_feature = SFNN_HALFKA2_BASE_INPUT_SIZE + (feature % SFNN_HALFKA2_PIECE_INPUTS);
        return true;
    }
    return false;
}

__global__ void sfnn_sparse_l0_pairwise_concat_kernel(
    const int* stm_indices,
    const int* nstm_indices,
    const float* weights,
    const float* bias,
    float* stm_l0,
    float* nstm_l0,
    float* combined,
    size_t batch,
    size_t max_active,
    size_t input_size,
    size_t ft_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pairwise = ft_size / 2;
    size_t total = batch * pairwise;
    if (tid >= total) {
        return;
    }

    size_t pair = tid % pairwise;
    size_t sample = tid / pairwise;
    size_t row0 = pair;
    size_t row1 = pairwise + pair;
    size_t l0_base = sample * ft_size;
    size_t sparse_base = sample * max_active;
    float stm_sum0 = bias[row0];
    float stm_sum1 = bias[row1];
    float nstm_sum0 = bias[row0];
    float nstm_sum1 = bias[row1];

    for (size_t slot = 0; slot < max_active; ++slot) {
        int stm_feature_i32 = stm_indices[sparse_base + slot];
        if (stm_feature_i32 >= 0 && static_cast<size_t>(stm_feature_i32) < input_size) {
            size_t feature = static_cast<size_t>(stm_feature_i32);
            size_t weight_base = feature * ft_size;
            stm_sum0 += weights[weight_base + row0];
            stm_sum1 += weights[weight_base + row1];
            size_t virtual_feature = 0;
            if (sfnn_factorized_virtual_feature(feature, input_size, &virtual_feature)) {
                size_t virtual_weight_base = virtual_feature * ft_size;
                stm_sum0 += weights[virtual_weight_base + row0];
                stm_sum1 += weights[virtual_weight_base + row1];
            }
        }

        int nstm_feature_i32 = nstm_indices[sparse_base + slot];
        if (nstm_feature_i32 >= 0 && static_cast<size_t>(nstm_feature_i32) < input_size) {
            size_t feature = static_cast<size_t>(nstm_feature_i32);
            size_t weight_base = feature * ft_size;
            nstm_sum0 += weights[weight_base + row0];
            nstm_sum1 += weights[weight_base + row1];
            size_t virtual_feature = 0;
            if (sfnn_factorized_virtual_feature(feature, input_size, &virtual_feature)) {
                size_t virtual_weight_base = virtual_feature * ft_size;
                nstm_sum0 += weights[virtual_weight_base + row0];
                nstm_sum1 += weights[virtual_weight_base + row1];
            }
        }
    }

    float stm0 = crelu(stm_sum0);
    float stm1 = crelu(stm_sum1);
    float nstm0 = crelu(nstm_sum0);
    float nstm1 = crelu(nstm_sum1);
    stm_l0[l0_base + row0] = stm0;
    stm_l0[l0_base + row1] = stm1;
    nstm_l0[l0_base + row0] = nstm0;
    nstm_l0[l0_base + row1] = nstm1;
    combined[l0_base + pair] = stm0 * stm1 * SFNN_PAIRWISE_SCALE;
    combined[l0_base + pairwise + pair] = nstm0 * nstm1 * SFNN_PAIRWISE_SCALE;
}

__global__ void sfnn_fold_halfka2_l0w_kernel(
    const float* weights,
    float* folded_weights,
    size_t ft_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = SFNN_HALFKA2_BASE_INPUT_SIZE * ft_size;
    if (tid >= total) {
        return;
    }

    const size_t feature = tid / ft_size;
    const size_t row = tid - feature * ft_size;
    const size_t piece = feature % SFNN_HALFKA2_PIECE_INPUTS;
    const size_t virtual_feature = SFNN_HALFKA2_BASE_INPUT_SIZE + piece;
    folded_weights[tid] = weights[tid] + weights[virtual_feature * ft_size + row];
}

__device__ __forceinline__ size_t sfnn_factorizer_axis_ids(
    size_t stack,
    size_t num_stacks,
    size_t king_axis_dim,
    size_t hand_axis_dim,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    size_t* out_ids) {
    size_t count = 0;
    size_t king_bucket_count = king_axis_dim == 0 ? 1 : king_axis_dim * king_axis_dim;
    size_t hand_bucket_count = hand_axis_dim == 0 ? 1 : hand_axis_dim * hand_axis_dim;
    size_t factorizer_stack_count = king_bucket_count * hand_bucket_count;
    size_t progress_bucket_count =
        (factorizer_stack_count != 0 && num_stacks % factorizer_stack_count == 0) ? num_stacks / factorizer_stack_count : 1;
    size_t axis_stack = stack / progress_bucket_count;
    size_t king_bucket = axis_stack % king_bucket_count;
    size_t hand_bucket = (axis_stack / king_bucket_count) % hand_bucket_count;
    size_t progress_bucket = stack % progress_bucket_count;

    if (use_king_axis != 0 && king_axis_dim != 0) {
        out_ids[count++] = king_bucket / king_axis_dim;
        out_ids[count++] = king_axis_dim + (king_bucket % king_axis_dim);
    }
    if (use_hand_axis != 0 && hand_axis_dim != 0) {
        size_t offset = 2 * king_axis_dim;
        out_ids[count++] = offset + hand_bucket / hand_axis_dim;
        out_ids[count++] = offset + hand_axis_dim + (hand_bucket % hand_axis_dim);
    }
    size_t offset = 2 * king_axis_dim + 2 * hand_axis_dim;
    if (use_king_hand_pair != 0 && king_axis_dim != 0 && hand_axis_dim != 0) {
        out_ids[count++] = offset + hand_bucket * king_bucket_count + king_bucket;
    }
    offset += (use_king_hand_pair != 0 && king_axis_dim != 0 && hand_axis_dim != 0)
        ? king_bucket_count * hand_bucket_count
        : 0;
    if (use_king_progress_pair != 0 && king_axis_dim != 0 && progress_bucket_count > 1) {
        out_ids[count++] = offset + progress_bucket * king_bucket_count + king_bucket;
    }
    offset += (use_king_progress_pair != 0 && king_axis_dim != 0 && progress_bucket_count > 1)
        ? king_bucket_count * progress_bucket_count
        : 0;
    if (use_hand_progress_pair != 0 && hand_axis_dim != 0 && progress_bucket_count > 1) {
        out_ids[count++] = offset + progress_bucket * hand_bucket_count + hand_bucket;
    }
    offset += (use_hand_progress_pair != 0 && hand_axis_dim != 0 && progress_bucket_count > 1)
        ? hand_bucket_count * progress_bucket_count
        : 0;
    if (use_progress_axis != 0 && progress_bucket_count > 1) {
        out_ids[count++] = offset + progress_bucket;
    }
    return count;
}

__device__ __forceinline__ float sfnn_factorizer_axis_alpha(
    size_t axis,
    size_t num_stacks,
    size_t king_axis_dim,
    size_t hand_axis_dim,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences) {
    const size_t king_bucket_count = king_axis_dim == 0 ? 1 : king_axis_dim * king_axis_dim;
    const size_t hand_bucket_count = hand_axis_dim == 0 ? 1 : hand_axis_dim * hand_axis_dim;
    const size_t factorizer_stack_count = king_bucket_count * hand_bucket_count;
    const size_t progress_bucket_count =
        (factorizer_stack_count != 0 && num_stacks % factorizer_stack_count == 0) ? num_stacks / factorizer_stack_count : 1;
    const size_t king_axis_count = 2 * king_axis_dim;
    const size_t base_axis_count = king_axis_count + 2 * hand_axis_dim;
    size_t pair_axis_count = 0;
    if (use_king_hand_pair != 0 && king_axis_dim != 0 && hand_axis_dim != 0) {
        pair_axis_count += king_bucket_count * hand_bucket_count;
    }
    if (use_king_progress_pair != 0 && king_axis_dim != 0 && progress_bucket_count > 1) {
        pair_axis_count += king_bucket_count * progress_bucket_count;
    }
    if (use_hand_progress_pair != 0 && hand_axis_dim != 0 && progress_bucket_count > 1) {
        pair_axis_count += hand_bucket_count * progress_bucket_count;
    }
    float alpha = progress_axis_alpha;
    if (axis < king_axis_count) {
        alpha = king_axis_alpha;
    } else if (axis < base_axis_count) {
        alpha = hand_axis_alpha;
    } else if (axis < base_axis_count + pair_axis_count) {
        alpha = pair_alpha;
    }
    if (factorizer_axis_confidences != nullptr) {
        alpha *= factorizer_axis_confidences[axis];
    }
    return alpha;
}

__device__ __forceinline__ void sfnn_factorizer_axis_alphas(
    const size_t* axis_ids,
    size_t axis_count,
    size_t num_stacks,
    size_t king_axis_dim,
    size_t hand_axis_dim,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    float* out_alphas) {
    for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
        out_alphas[axis_idx] = sfnn_factorizer_axis_alpha(
            axis_ids[axis_idx],
            num_stacks,
            king_axis_dim,
            hand_axis_dim,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences);
    }
}

__host__ __device__ __forceinline__ size_t sfnn_factorizer_axis_count(
    size_t num_stacks,
    size_t king_axis_dim,
    size_t hand_axis_dim,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair) {
    const size_t king_bucket_count = king_axis_dim == 0 ? 1 : king_axis_dim * king_axis_dim;
    const size_t hand_bucket_count = hand_axis_dim == 0 ? 1 : hand_axis_dim * hand_axis_dim;
    const size_t factorizer_stack_count = king_bucket_count * hand_bucket_count;
    const size_t progress_bucket_count =
        (factorizer_stack_count != 0 && num_stacks % factorizer_stack_count == 0) ? num_stacks / factorizer_stack_count : 1;
    size_t count = 2 * king_axis_dim + 2 * hand_axis_dim;
    if (use_king_hand_pair != 0 && king_axis_dim != 0 && hand_axis_dim != 0) {
        count += king_bucket_count * hand_bucket_count;
    }
    if (use_king_progress_pair != 0 && king_axis_dim != 0 && progress_bucket_count > 1) {
        count += king_bucket_count * progress_bucket_count;
    }
    if (use_hand_progress_pair != 0 && hand_axis_dim != 0 && progress_bucket_count > 1) {
        count += hand_bucket_count * progress_bucket_count;
    }
    if (use_progress_axis != 0 && progress_bucket_count > 1) {
        count += progress_bucket_count;
    }
    return count;
}

__device__ __forceinline__ float sfnn_quantize_dequant_clamped(
    float value,
    float scale,
    float qmin,
    float qmax) {
    float q = roundf(value * scale);
    q = fminf(fmaxf(q, qmin), qmax);
    return q / scale;
}

__global__ void sfnn_build_quantized_proxy_l0w_kernel(
    const float* src,
    float* dst,
    size_t src_input_size,
    size_t dst_input_size,
    size_t virtual_rows,
    size_t ft_size,
    float scale) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = dst_input_size * ft_size;
    if (tid >= total) {
        return;
    }

    const size_t feature = tid / ft_size;
    const size_t row = tid - feature * ft_size;
    float value = src[tid];
    if (src_input_size != dst_input_size && virtual_rows != 0) {
        const size_t virtual_feature = dst_input_size + (feature % virtual_rows);
        value += src[virtual_feature * ft_size + row];
    }
    dst[tid] = sfnn_quantize_dequant_clamped(value, scale, -32768.0f, 32767.0f);
}

__global__ void sfnn_build_quantized_proxy_vector_kernel(
    const float* src,
    float* dst,
    size_t len,
    float scale,
    float qmin,
    float qmax) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    dst[tid] = sfnn_quantize_dequant_clamped(src[tid], scale, qmin, qmax);
}

__global__ void sfnn_build_quantized_proxy_stacked_weights_kernel(
    const float* weights,
    const float* shared_weights,
    const float* axis_weights,
    float* dst,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int shared_weight_input_major,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    float scale) {
    const size_t weight_cell_count = input_dim * output_dim;
    const size_t total = num_stacks * weight_cell_count;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const size_t stack = tid / weight_cell_count;
    const size_t cell = tid - stack * weight_cell_count;
    const size_t out_col = cell / input_dim;
    const size_t in_col = cell - out_col * input_dim;
    const size_t factorizer_cell = shared_weight_input_major != 0
        ? in_col * output_dim + out_col
        : cell;

    float value = weights[tid];
    if (has_shared != 0) {
        value += shared_alpha * shared_weights[factorizer_cell];
    }

    size_t axis_ids[8];
    const size_t axis_count = sfnn_factorizer_axis_ids(
        stack,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
    if (has_axis != 0) {
        float axis_alphas[8];
        sfnn_factorizer_axis_alphas(
            axis_ids,
            axis_count,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences,
            axis_alphas);
        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
            value += axis_alphas[axis_idx] * axis_weights[axis_ids[axis_idx] * weight_cell_count + factorizer_cell];
        }
    }

    dst[tid] = sfnn_quantize_dequant_clamped(value, scale, -128.0f, 127.0f);
}

__global__ void sfnn_build_quantized_proxy_stacked_bias_kernel(
    const float* bias,
    const float* shared_bias,
    const float* axis_bias,
    float* dst,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    float scale) {
    const size_t total = num_stacks * output_dim;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const size_t stack = tid / output_dim;
    const size_t out_col = tid - stack * output_dim;
    float value = bias[tid];
    if (has_shared != 0) {
        value += shared_alpha * shared_bias[out_col];
    }

    size_t axis_ids[8];
    const size_t axis_count = sfnn_factorizer_axis_ids(
        stack,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
    if (has_axis != 0) {
        float axis_alphas[8];
        sfnn_factorizer_axis_alphas(
            axis_ids,
            axis_count,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences,
            axis_alphas);
        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
            value += axis_alphas[axis_idx] * axis_bias[axis_ids[axis_idx] * output_dim + out_col];
        }
    }

    dst[tid] = sfnn_quantize_dequant_clamped(value, scale, -2147483648.0f, 2147483647.0f);
}

__global__ void sfnn_stacked_l1_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    const float* shared_weights,
    const float* shared_bias,
    const float* axis_weights,
    const float* axis_bias,
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * output_dim;
    if (tid >= total) {
        return;
    }

    size_t out_col = tid % output_dim;
    size_t sample = tid / output_dim;
    int stack_i32 = buckets[sample];
    if (stack_i32 < 0 || static_cast<size_t>(stack_i32) >= num_stacks) {
        output[tid] = 0.0f;
        return;
    }

    size_t stack = static_cast<size_t>(stack_i32);
    size_t input_base = sample * input_dim;
    size_t stack_base = stack * output_dim * input_dim;
    float sum = bias[stack * output_dim + out_col];
    if (has_shared != 0) {
        sum += shared_alpha * shared_bias[out_col];
    }
    size_t axis_ids[8];
    size_t axis_count = sfnn_factorizer_axis_ids(
        stack,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
    float axis_alphas[8];
    if (has_axis != 0) {
        sfnn_factorizer_axis_alphas(
            axis_ids,
            axis_count,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences,
            axis_alphas);
        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
            const float axis_alpha = axis_alphas[axis_idx];
            sum += axis_alpha * axis_bias[axis_ids[axis_idx] * output_dim + out_col];
        }
    }
    for (size_t in_col = 0; in_col < input_dim; ++in_col) {
        float input_value = input[input_base + in_col];
        sum += input_value * weights[stack_base + out_col * input_dim + in_col];
        if (has_shared != 0) {
            sum += input_value * shared_alpha * shared_weights[in_col * output_dim + out_col];
        }
        if (has_axis != 0) {
            for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                const float axis_alpha = axis_alphas[axis_idx];
                sum += input_value * axis_alpha *
                    axis_weights[axis_ids[axis_idx] * input_dim * output_dim + in_col * output_dim + out_col];
            }
        }
    }
    output[tid] = sum;
}

__global__ void sfnn_stacked_l1_warp_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    const float* shared_weights,
    const float* shared_bias,
    const float* axis_weights,
    const float* axis_bias,
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences) {
    constexpr unsigned int WARP = 32;
    constexpr unsigned int WARPS_PER_BLOCK = 8;
    const unsigned int lane = threadIdx.x & (WARP - 1U);
    const unsigned int warp_in_block = threadIdx.x >> 5;
    const size_t warp_id = static_cast<size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    const size_t total = batch * output_dim;
    if (warp_id >= total) {
        return;
    }

    const size_t out_col = warp_id % output_dim;
    const size_t sample = warp_id / output_dim;
    const int stack_i32 = buckets[sample];
    if (stack_i32 < 0 || static_cast<size_t>(stack_i32) >= num_stacks) {
        if (lane == 0) {
            output[warp_id] = 0.0f;
        }
        return;
    }

    const size_t stack = static_cast<size_t>(stack_i32);
    const size_t input_base = sample * input_dim;
    const size_t stack_base = stack * output_dim * input_dim + out_col * input_dim;
    size_t axis_ids[8];
    const size_t axis_count = sfnn_factorizer_axis_ids(
        stack,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
    float axis_alphas[8];
    if (has_axis != 0) {
        sfnn_factorizer_axis_alphas(
            axis_ids,
            axis_count,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences,
            axis_alphas);
    }

    float sum = 0.0f;
    for (size_t in_col = lane; in_col < input_dim; in_col += WARP) {
        const float input_value = input[input_base + in_col];
        float weight = weights[stack_base + in_col];
        if (has_shared != 0) {
            weight += shared_alpha * shared_weights[in_col * output_dim + out_col];
        }
        if (has_axis != 0) {
            for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                const float axis_alpha = axis_alphas[axis_idx];
                weight += axis_alpha *
                    axis_weights[axis_ids[axis_idx] * input_dim * output_dim + in_col * output_dim + out_col];
            }
        }
        sum += input_value * weight;
    }

    unsigned int mask = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
        sum += bias[stack * output_dim + out_col];
        if (has_shared != 0) {
            sum += shared_alpha * shared_bias[out_col];
        }
        if (has_axis != 0) {
            for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                const float axis_alpha = axis_alphas[axis_idx];
                sum += axis_alpha * axis_bias[axis_ids[axis_idx] * output_dim + out_col];
            }
        }
        output[warp_id] = sum;
    }
}

__global__ void sfnn_grouped_l1_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t group_count,
    size_t group_input,
    size_t group_output) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * output_dim;
    if (tid >= total) {
        return;
    }

    size_t out_col = tid % output_dim;
    size_t sample = tid / output_dim;
    int stack_i32 = buckets[sample];
    if (stack_i32 < 0 || static_cast<size_t>(stack_i32) >= num_stacks ||
        out_col >= group_count * group_output ||
        input_dim != group_count * group_input) {
        output[tid] = 0.0f;
        return;
    }

    size_t stack = static_cast<size_t>(stack_i32);
    size_t group = out_col / group_output;
    size_t local_out = out_col - group * group_output;
    size_t input_base = sample * input_dim + group * group_input;
    size_t stack_stride = group_count * group_output * group_input;
    size_t weight_base = stack * stack_stride +
        group * group_output * group_input +
        local_out * group_input;
    float sum = bias[stack * output_dim + out_col];
    for (size_t local_in = 0; local_in < group_input; ++local_in) {
        sum += input[input_base + local_in] * weights[weight_base + local_in];
    }
    output[tid] = sum;
}

__global__ void sfnn_common_shard_l1_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t group_count,
    size_t common_input,
    size_t shard_input,
    size_t group_output) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * output_dim;
    if (tid >= total) {
        return;
    }

    size_t out_col = tid % output_dim;
    size_t sample = tid / output_dim;
    int stack_i32 = buckets[sample];
    if (stack_i32 < 0 || static_cast<size_t>(stack_i32) >= num_stacks ||
        out_col >= group_count * group_output ||
        input_dim != common_input + group_count * shard_input) {
        output[tid] = 0.0f;
        return;
    }

    size_t stack = static_cast<size_t>(stack_i32);
    size_t group = out_col / group_output;
    size_t local_out = out_col - group * group_output;
    size_t row_input = common_input + shard_input;
    size_t stack_stride = group_count * group_output * row_input;
    size_t weight_base = stack * stack_stride +
        group * group_output * row_input +
        local_out * row_input;
    size_t input_base = sample * input_dim;
    float sum = bias[stack * output_dim + out_col];
    for (size_t local_in = 0; local_in < common_input; ++local_in) {
        sum += input[input_base + local_in] * weights[weight_base + local_in];
    }
    size_t shard_base = common_input + group * shard_input;
    size_t shard_weight_base = weight_base + common_input;
    for (size_t local_in = 0; local_in < shard_input; ++local_in) {
        sum += input[input_base + shard_base + local_in] * weights[shard_weight_base + local_in];
    }
    output[tid] = sum;
}

__global__ void sfnn_l2_input_kernel(const float* l1, float* output, size_t batch, size_t l1_hidden, int l1_skip) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t l2_input_dim = l1_hidden * 2;
    size_t total = batch * l2_input_dim;
    if (tid >= total) {
        return;
    }

    size_t col = tid % l2_input_dim;
    size_t sample = tid / l2_input_dim;
    size_t source_col = col % l1_hidden;
    size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    float value = l1[sample * l1_out + source_col];
    if (col < l1_hidden) {
        float abs_value = fabsf(value);
        output[tid] = crelu(abs_value * abs_value * SFNN_PAIRWISE_SCALE);
    } else {
        output[tid] = crelu(value);
    }
}

__global__ void sfnn_stacked_l2_crelu_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    const float* shared_weights,
    const float* shared_bias,
    const float* axis_weights,
    const float* axis_bias,
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * output_dim;
    if (tid >= total) {
        return;
    }

    size_t out_col = tid % output_dim;
    size_t sample = tid / output_dim;
    int stack_i32 = buckets[sample];
    if (stack_i32 < 0 || static_cast<size_t>(stack_i32) >= num_stacks) {
        output[tid] = 0.0f;
        return;
    }

    size_t stack = static_cast<size_t>(stack_i32);
    size_t input_base = sample * input_dim;
    size_t stack_base = stack * output_dim * input_dim;
    float sum = bias[stack * output_dim + out_col];
    if (has_shared != 0) {
        sum += shared_alpha * shared_bias[out_col];
    }
    size_t axis_ids[8];
    size_t axis_count = sfnn_factorizer_axis_ids(
        stack,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
    float axis_alphas[8];
    if (has_axis != 0) {
        sfnn_factorizer_axis_alphas(
            axis_ids,
            axis_count,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences,
            axis_alphas);
        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
            const float axis_alpha = axis_alphas[axis_idx];
            sum += axis_alpha * axis_bias[axis_ids[axis_idx] * output_dim + out_col];
        }
    }
    for (size_t in_col = 0; in_col < input_dim; ++in_col) {
        float weight = weights[stack_base + out_col * input_dim + in_col];
        if (has_shared != 0) {
            weight += shared_alpha * shared_weights[out_col * input_dim + in_col];
        }
        if (has_axis != 0) {
            for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                const float axis_alpha = axis_alphas[axis_idx];
                weight += axis_alpha *
                    axis_weights[axis_ids[axis_idx] * output_dim * input_dim + out_col * input_dim + in_col];
            }
        }
        sum += input[input_base + in_col] * weight;
    }
    output[tid] = crelu(sum);
}

__global__ void sfnn_stacked_l3_output_kernel(
    const float* input,
    const float* l1,
    const float* weights,
    const float* bias,
    const float* shared_weights,
    const float* shared_bias,
    const float* axis_weights,
    const float* axis_bias,
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t l1_hidden,
    int l1_skip,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences) {
    size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) {
        return;
    }

    int stack_i32 = buckets[sample];
    if (stack_i32 < 0 || static_cast<size_t>(stack_i32) >= num_stacks) {
        output[sample] = 0.0f;
        return;
    }

    size_t stack = static_cast<size_t>(stack_i32);
    size_t input_base = sample * input_dim;
    float sum = bias[stack];
    if (has_shared != 0) {
        sum += shared_alpha * shared_bias[0];
    }
    size_t axis_ids[8];
    size_t axis_count = sfnn_factorizer_axis_ids(
        stack,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
    float axis_alphas[8];
    if (has_axis != 0) {
        sfnn_factorizer_axis_alphas(
            axis_ids,
            axis_count,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences,
            axis_alphas);
        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
            const float axis_alpha = axis_alphas[axis_idx];
            sum += axis_alpha * axis_bias[axis_ids[axis_idx]];
        }
    }
    for (size_t in_col = 0; in_col < input_dim; ++in_col) {
        float weight = weights[stack * input_dim + in_col];
        if (has_shared != 0) {
            weight += shared_alpha * shared_weights[in_col];
        }
        if (has_axis != 0) {
            for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                const float axis_alpha = axis_alphas[axis_idx];
                weight += axis_alpha * axis_weights[axis_ids[axis_idx] * input_dim + in_col];
            }
        }
        sum += input[input_base + in_col] * weight;
    }
    if (l1_skip != 0) {
        sum += l1[sample * sfnn_l1_out_for_shape(l1_hidden, l1_skip) + l1_hidden];
    }
    output[sample] = sum;
}

__device__ float loss_sigmoid(float value) {
    float exp_neg = expf(-value);
    return 1.0f / (1.0f + exp_neg);
}

__device__ float sign_f32(float value) {
    if (value > 0.0f) {
        return 1.0f;
    }
    if (value < 0.0f) {
        return -1.0f;
    }
    return 0.0f;
}

__global__ void loss_sigmoid_pow_reduce_kernel(
    const float* outputs,
    const float* targets,
    const float* entry_weights,
    float* per_sample,
    float* mean_output_gradients,
    float output_inv_scale,
    float loss_pow_exp,
    size_t batch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }

    float prediction = loss_sigmoid(outputs[idx] * output_inv_scale);
    float error = prediction - targets[idx];
    float abs_error = fabsf(error);
    float loss = powf(abs_error, loss_pow_exp);
    float loss_gradient = abs_error == 0.0f ? 0.0f : loss_pow_exp * sign_f32(error) * powf(abs_error, loss_pow_exp - 1.0f);
    float gradient = loss_gradient * prediction * (1.0f - prediction) * output_inv_scale;
    float weighted = entry_weights[idx] * loss;
    per_sample[idx] = weighted;
    mean_output_gradients[idx] = entry_weights[idx] * gradient / static_cast<float>(batch);
}

__global__ void loss_win_rate_model_reduce_kernel(
    const float* outputs,
    const float* targets,
    const float* entry_weights,
    float* per_sample,
    float* mean_output_gradients,
    float loss_pow_exp,
    float wrm_output_scale,
    float wrm_in_offset_over_scaling,
    size_t batch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }

    float score_scaled = outputs[idx] * wrm_output_scale;
    float q = loss_sigmoid(score_scaled - wrm_in_offset_over_scaling);
    float qm = loss_sigmoid(-score_scaled - wrm_in_offset_over_scaling);
    float prediction = (1.0f + q - qm) * 0.5f;
    float error = prediction - targets[idx];
    float abs_error = fabsf(error);
    float loss = powf(abs_error, loss_pow_exp);
    float q_prime = q * (1.0f - q);
    float qm_prime = qm * (1.0f - qm);
    float prediction_gradient = 0.5f * wrm_output_scale * (q_prime + qm_prime);
    float loss_gradient = abs_error == 0.0f ? 0.0f : loss_pow_exp * sign_f32(error) * powf(abs_error, loss_pow_exp - 1.0f);
    per_sample[idx] = entry_weights[idx] * loss;
    mean_output_gradients[idx] = entry_weights[idx] * loss_gradient * prediction_gradient / static_cast<float>(batch);
}

__global__ void loss_finalize_from_per_sample_kernel(
    const float* per_sample,
    float* weighted_sum,
    float* mean,
    size_t batch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float sum = 0.0f;
    for (size_t idx = 0; idx < batch; ++idx) {
        sum += per_sample[idx];
    }
    weighted_sum[0] = sum;
    mean[0] = sum / static_cast<float>(batch);
}

__device__ float crelu_pre_gradient_from_value(float activation, float output_gradient) {
    return activation > 0.0f && activation < 1.0f ? output_gradient : 0.0f;
}

constexpr unsigned int SFNN_DENSE_REDUCE_TILE = 16;
constexpr unsigned int SFNN_DENSE_REDUCE_THREADS = SFNN_DENSE_REDUCE_TILE * SFNN_DENSE_REDUCE_TILE;
constexpr size_t SFNN_DENSE_REDUCE_MAX_FAST_STACKS = 128;
constexpr size_t SFNN_DENSE_REDUCE_ACCUM_MAX_STACKS = 16;

__device__ __forceinline__ float sfnn_dense_param_reduce_output_gradient(
    const float* activations,
    const float* output_gradients,
    size_t index,
    int use_crelu_gradient) {
    float grad = output_gradients[index];
    if (use_crelu_gradient != 0) {
        grad = crelu_pre_gradient_from_value(activations[index], grad);
    }
    return grad;
}

__global__ void sfnn_dense_param_reduce_tiled_kernel(
    const float* inputs,
    const float* activations,
    const float* output_gradients,
    const int* buckets,
    float* weight_gradients,
    float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    int has_shared,
    int shared_weight_input_major,
    float shared_alpha,
    int use_crelu_gradient) {
    const unsigned int input_tiles =
        static_cast<unsigned int>((input_dim + SFNN_DENSE_REDUCE_TILE - 1U) / SFNN_DENSE_REDUCE_TILE);
    const unsigned int tile = blockIdx.x;
    const unsigned int input_tile = tile % input_tiles;
    const unsigned int output_tile = tile / input_tiles;
    const unsigned int input_lane = threadIdx.x >> 4;
    const unsigned int output_lane = threadIdx.x & 15U;
    const size_t in_col = static_cast<size_t>(input_tile) * SFNN_DENSE_REDUCE_TILE + input_lane;
    const size_t out_col = static_cast<size_t>(output_tile) * SFNN_DENSE_REDUCE_TILE + output_lane;
    const size_t stack = blockIdx.z;
    if (stack >= num_stacks) {
        return;
    }

    const size_t split_count = gridDim.y;
    const size_t rows_per_split = (batch + split_count - 1) / split_count;
    const size_t row_begin = blockIdx.y * rows_per_split;
    const size_t row_end = min(row_begin + rows_per_split, batch);
    float weight_sum = 0.0f;
    float bias_sum = 0.0f;

    const bool valid_weight_cell = in_col < input_dim && out_col < output_dim;
    const bool valid_bias_cell = input_lane == 0 && input_tile == 0 && out_col < output_dim;
    for (size_t sample = row_begin; sample < row_end; ++sample) {
        int stack_i32 = buckets[sample];
        if (stack_i32 < 0 || static_cast<size_t>(stack_i32) != stack) {
            continue;
        }
        const size_t out_idx = sample * output_dim + out_col;
        float grad = 0.0f;
        if (out_col < output_dim) {
            grad = sfnn_dense_param_reduce_output_gradient(activations, output_gradients, out_idx, use_crelu_gradient);
        }
        if (valid_weight_cell && grad != 0.0f) {
            float input_value = inputs[sample * input_dim + in_col];
            if (input_value != 0.0f) {
                weight_sum += grad * input_value;
            }
        }
        if (valid_bias_cell) {
            bias_sum += grad;
        }
    }

    if (valid_weight_cell && weight_sum != 0.0f) {
        const size_t weight_idx = stack * output_dim * input_dim + out_col * input_dim + in_col;
        atomicAdd(&weight_gradients[weight_idx], weight_sum);
        if (has_shared != 0) {
            const size_t shared_idx = shared_weight_input_major != 0
                ? in_col * output_dim + out_col
                : out_col * input_dim + in_col;
            atomicAdd(&shared_weight_gradients[shared_idx], shared_alpha * weight_sum);
        }
    }
    if (valid_bias_cell && bias_sum != 0.0f) {
        atomicAdd(&bias_gradients[stack * output_dim + out_col], bias_sum);
        if (has_shared != 0) {
            atomicAdd(&shared_bias_gradients[out_col], shared_alpha * bias_sum);
        }
    }
}

__global__ void sfnn_dense_param_reduce_accum_stacks_kernel(
    const float* inputs,
    const float* activations,
    const float* output_gradients,
    const int* buckets,
    float* weight_gradients,
    float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    float* axis_weight_gradients,
    float* axis_bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int shared_weight_input_major,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    int use_crelu_gradient) {
    const unsigned int input_tiles =
        static_cast<unsigned int>((input_dim + SFNN_DENSE_REDUCE_TILE - 1U) / SFNN_DENSE_REDUCE_TILE);
    const unsigned int tile = blockIdx.x;
    const unsigned int input_tile = tile % input_tiles;
    const unsigned int output_tile = tile / input_tiles;
    const unsigned int input_lane = threadIdx.x >> 4;
    const unsigned int output_lane = threadIdx.x & 15U;
    const size_t in_col = static_cast<size_t>(input_tile) * SFNN_DENSE_REDUCE_TILE + input_lane;
    const size_t out_col = static_cast<size_t>(output_tile) * SFNN_DENSE_REDUCE_TILE + output_lane;

    if (num_stacks > SFNN_DENSE_REDUCE_ACCUM_MAX_STACKS) {
        return;
    }

    float weight_sums[SFNN_DENSE_REDUCE_ACCUM_MAX_STACKS];
    float bias_sums[SFNN_DENSE_REDUCE_ACCUM_MAX_STACKS];
#pragma unroll
    for (size_t stack = 0; stack < SFNN_DENSE_REDUCE_ACCUM_MAX_STACKS; ++stack) {
        weight_sums[stack] = 0.0f;
        bias_sums[stack] = 0.0f;
    }

    const size_t split_count = gridDim.y;
    const size_t rows_per_split = (batch + split_count - 1) / split_count;
    const size_t row_begin = blockIdx.y * rows_per_split;
    const size_t row_end = min(row_begin + rows_per_split, batch);
    const bool valid_weight_cell = in_col < input_dim && out_col < output_dim;
    const bool valid_bias_cell = input_lane == 0 && input_tile == 0 && out_col < output_dim;

    for (size_t sample = row_begin; sample < row_end; ++sample) {
        int stack_i32 = buckets[sample];
        if (stack_i32 < 0 || static_cast<size_t>(stack_i32) >= num_stacks) {
            continue;
        }
        const size_t stack = static_cast<size_t>(stack_i32);
        const size_t out_idx = sample * output_dim + out_col;
        float grad = 0.0f;
        if (out_col < output_dim) {
            grad = sfnn_dense_param_reduce_output_gradient(activations, output_gradients, out_idx, use_crelu_gradient);
        }
        if (valid_weight_cell && grad != 0.0f) {
            float input_value = inputs[sample * input_dim + in_col];
            if (input_value != 0.0f) {
                weight_sums[stack] += grad * input_value;
            }
        }
        if (valid_bias_cell) {
            bias_sums[stack] += grad;
        }
    }

    float shared_weight_sum = 0.0f;
    float shared_bias_sum = 0.0f;
    const size_t shared_idx = shared_weight_input_major != 0
        ? in_col * output_dim + out_col
        : out_col * input_dim + in_col;
    const size_t axis_cell_count = input_dim * output_dim;
    const size_t axis_weight_cell = shared_weight_input_major != 0
        ? in_col * output_dim + out_col
        : out_col * input_dim + in_col;

    for (size_t stack = 0; stack < num_stacks; ++stack) {
        const float weight_sum = weight_sums[stack];
        if (valid_weight_cell && weight_sum != 0.0f) {
            const size_t weight_idx = stack * output_dim * input_dim + out_col * input_dim + in_col;
            atomicAdd(&weight_gradients[weight_idx], weight_sum);
            shared_weight_sum += weight_sum;
            if (has_axis != 0) {
                size_t axis_ids[8];
                size_t axis_count = sfnn_factorizer_axis_ids(
                    stack,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
                float axis_alphas[8];
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
                for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                    const float axis_alpha = axis_alphas[axis_idx];
                    atomicAdd(
                        &axis_weight_gradients[axis_ids[axis_idx] * axis_cell_count + axis_weight_cell],
                        axis_alpha * weight_sum);
                }
            }
        }

        const float bias_sum = bias_sums[stack];
        if (valid_bias_cell && bias_sum != 0.0f) {
            atomicAdd(&bias_gradients[stack * output_dim + out_col], bias_sum);
            shared_bias_sum += bias_sum;
            if (has_axis != 0) {
                size_t axis_ids[8];
                size_t axis_count = sfnn_factorizer_axis_ids(
                    stack,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
                float axis_alphas[8];
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
                for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                    const float axis_alpha = axis_alphas[axis_idx];
                    atomicAdd(&axis_bias_gradients[axis_ids[axis_idx] * output_dim + out_col], axis_alpha * bias_sum);
                }
            }
        }
    }

    if (has_shared != 0 && valid_weight_cell && shared_weight_sum != 0.0f) {
        atomicAdd(&shared_weight_gradients[shared_idx], shared_alpha * shared_weight_sum);
    }
    if (has_shared != 0 && valid_bias_cell && shared_bias_sum != 0.0f) {
        atomicAdd(&shared_bias_gradients[out_col], shared_alpha * shared_bias_sum);
    }
}

__global__ void sfnn_fold_stacked_param_gradients_to_factorizers_kernel(
    const float* weight_gradients,
    const float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    float* axis_weight_gradients,
    float* axis_bias_gradients,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int shared_weight_input_major,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences) {
    const size_t weight_cell_count = input_dim * output_dim;
    const size_t weight_total = num_stacks * weight_cell_count;
    const size_t bias_total = num_stacks * output_dim;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = weight_total > bias_total ? weight_total : bias_total;
    if (tid >= total) {
        return;
    }

    if (tid < weight_total) {
        const float grad = weight_gradients[tid];
        if (grad != 0.0f) {
            const size_t stack = tid / weight_cell_count;
            const size_t cell = tid - stack * weight_cell_count;
            const size_t out_col = cell / input_dim;
            const size_t in_col = cell - out_col * input_dim;
            const size_t factorizer_cell = shared_weight_input_major != 0
                ? in_col * output_dim + out_col
                : cell;
            if (has_shared != 0) {
                atomicAdd(&shared_weight_gradients[factorizer_cell], shared_alpha * grad);
            }
            if (has_axis != 0) {
                size_t axis_ids[8];
                const size_t axis_count = sfnn_factorizer_axis_ids(
                    stack,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_axis,
                    use_hand_axis,
                    use_progress_axis,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    axis_ids);
                float axis_alphas[8];
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
                for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                    atomicAdd(
                        &axis_weight_gradients[axis_ids[axis_idx] * weight_cell_count + factorizer_cell],
                        axis_alphas[axis_idx] * grad);
                }
            }
        }
    }

    if (tid < bias_total) {
        const float grad = bias_gradients[tid];
        if (grad != 0.0f) {
            const size_t stack = tid / output_dim;
            const size_t out_col = tid - stack * output_dim;
            if (has_shared != 0) {
                atomicAdd(&shared_bias_gradients[out_col], shared_alpha * grad);
            }
            if (has_axis != 0) {
                size_t axis_ids[8];
                const size_t axis_count = sfnn_factorizer_axis_ids(
                    stack,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_axis,
                    use_hand_axis,
                    use_progress_axis,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    axis_ids);
                float axis_alphas[8];
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
                for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                    atomicAdd(&axis_bias_gradients[axis_ids[axis_idx] * output_dim + out_col], axis_alphas[axis_idx] * grad);
                }
            }
        }
    }
}

__global__ void sfnn_add_saturation_penalty_gradients_kernel(
    const float* weights,
    const float* shared_weights,
    const float* axis_weights,
    float* weight_gradients,
    float* shared_weight_gradients,
    float* axis_weight_gradients,
    const int* dirty_buckets,
    size_t dirty_count,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int shared_weight_input_major,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    float weight_scale,
    float threshold,
    float penalty) {
    const size_t weight_cell_count = input_dim * output_dim;
    const size_t stack_count = dirty_count == 0 ? num_stacks : dirty_count;
    const size_t total = stack_count * weight_cell_count;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const size_t selected_stack = tid / weight_cell_count;
    const size_t stack = dirty_count == 0 ? selected_stack : static_cast<size_t>(dirty_buckets[selected_stack]);
    if (stack >= num_stacks) {
        return;
    }
    const size_t cell = tid - selected_stack * weight_cell_count;
    const size_t out_col = cell / input_dim;
    const size_t in_col = cell - out_col * input_dim;
    const size_t stacked_idx = stack * weight_cell_count + cell;
    const size_t factorizer_cell = shared_weight_input_major != 0
        ? in_col * output_dim + out_col
        : cell;

    float value = weights[stacked_idx];
    if (has_shared != 0) {
        value += shared_alpha * shared_weights[factorizer_cell];
    }

    size_t axis_ids[8];
    const size_t axis_count = sfnn_factorizer_axis_ids(
        stack,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
    float axis_alphas[8];
    if (has_axis != 0) {
        sfnn_factorizer_axis_alphas(
            axis_ids,
            axis_count,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences,
            axis_alphas);
        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
            value += axis_alphas[axis_idx] * axis_weights[axis_ids[axis_idx] * weight_cell_count + factorizer_cell];
        }
    }

    const float q = value * weight_scale;
    const float excess = fabsf(q) - threshold;
    if (excess <= 0.0f) {
        return;
    }

    const float sign = q < 0.0f ? -1.0f : 1.0f;
    const float grad = penalty * 2.0f * excess * sign * weight_scale;
    atomicAdd(&weight_gradients[stacked_idx], grad);
    if (has_shared != 0) {
        atomicAdd(&shared_weight_gradients[factorizer_cell], shared_alpha * grad);
    }
    if (has_axis != 0) {
        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
            atomicAdd(
                &axis_weight_gradients[axis_ids[axis_idx] * weight_cell_count + factorizer_cell],
                axis_alphas[axis_idx] * grad);
        }
    }
}

__global__ void sfnn_add_residual_count_decay_gradients_kernel(
    const float* weights,
    float* gradients,
    const float* decay_by_stack,
    const int* dirty_buckets,
    size_t dirty_count,
    size_t stride,
    size_t num_stacks) {
    const size_t stack_count = dirty_count == 0 ? num_stacks : dirty_count;
    const size_t total = stack_count * stride;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const size_t selected_stack = tid / stride;
    const size_t stack = dirty_count == 0 ? selected_stack : static_cast<size_t>(dirty_buckets[selected_stack]);
    if (stack >= num_stacks) {
        return;
    }
    const float decay = decay_by_stack[stack];
    if (decay == 0.0f) {
        return;
    }
    const size_t cell = tid - selected_stack * stride;
    const size_t idx = stack * stride + cell;
    atomicAdd(&gradients[idx], 2.0f * decay * weights[idx]);
}

__global__ void sfnn_stacked_l3_backward_kernel(
    const float* inputs,
    const float* output_gradients,
    const float* weights,
    const float* shared_weights,
    const float* axis_weights,
    const int* buckets,
    float* input_gradients,
    float* l1_gradients,
    float* weight_gradients,
    float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    float* axis_weight_gradients,
    float* axis_bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t l1_out,
    int l1_skip,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    int compute_parameter_gradients,
    int accumulate_factorizer_gradients) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t input_gradient_len = batch * input_dim;
    size_t l1_gradient_len = batch * l1_out;

    if (tid < input_gradient_len) {
        size_t sample = tid / input_dim;
        size_t row = tid - sample * input_dim;
        int stack_i32 = buckets[sample];
        float value = 0.0f;
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            float output_gradient = output_gradients[sample];
            float weight = weights[stack * input_dim + row];
            if (has_shared != 0) {
                weight += shared_alpha * shared_weights[row];
            }
            size_t axis_ids[8];
            size_t axis_count = sfnn_factorizer_axis_ids(
                stack,
                num_stacks,
                factorizer_king_axis_dim,
                factorizer_hand_axis_dim,
                use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
            float axis_alphas[8];
            if (has_axis != 0) {
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
                for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                    const float axis_alpha = axis_alphas[axis_idx];
                    weight += axis_alpha * axis_weights[axis_ids[axis_idx] * input_dim + row];
                }
            }
            value = output_gradient * weight;
            float input_value = inputs[tid];
            if (compute_parameter_gradients != 0 && output_gradient != 0.0f && input_value != 0.0f) {
                float weight_gradient = output_gradient * input_value;
                atomicAdd(&weight_gradients[stack * input_dim + row], weight_gradient);
                if (accumulate_factorizer_gradients != 0 && has_shared != 0) {
                    atomicAdd(&shared_weight_gradients[row], shared_alpha * weight_gradient);
                }
                if (accumulate_factorizer_gradients != 0 && has_axis != 0) {
                    for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                        const float axis_alpha = axis_alphas[axis_idx];
                        atomicAdd(&axis_weight_gradients[axis_ids[axis_idx] * input_dim + row], axis_alpha * weight_gradient);
                    }
                }
            }
        }
        input_gradients[tid] = value;
    }

    if (tid < l1_gradient_len) {
        size_t sample = tid / l1_out;
        size_t col = tid - sample * l1_out;
        l1_gradients[tid] = (l1_skip != 0 && col + 1 == l1_out) ? output_gradients[sample] : 0.0f;
    }

    if (compute_parameter_gradients != 0 && tid < batch) {
        int stack_i32 = buckets[tid];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            float grad = output_gradients[tid];
            if (grad != 0.0f) {
                atomicAdd(&bias_gradients[static_cast<size_t>(stack_i32)], grad);
                if (accumulate_factorizer_gradients != 0 && has_shared != 0) {
                    atomicAdd(&shared_bias_gradients[0], shared_alpha * grad);
                }
                if (accumulate_factorizer_gradients != 0 && has_axis != 0) {
                    size_t axis_ids[8];
                    size_t axis_count = sfnn_factorizer_axis_ids(
                        static_cast<size_t>(stack_i32),
                        num_stacks,
                        factorizer_king_axis_dim,
                        factorizer_hand_axis_dim,
                        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
                    float axis_alphas[8];
                    sfnn_factorizer_axis_alphas(
                        axis_ids,
                        axis_count,
                        num_stacks,
                        factorizer_king_axis_dim,
                        factorizer_hand_axis_dim,
                        use_king_hand_pair,
                        use_king_progress_pair,
                        use_hand_progress_pair,
                        king_axis_alpha,
                        hand_axis_alpha,
                        progress_axis_alpha,
                        pair_alpha,
                        factorizer_axis_confidences,
            axis_alphas);
                    for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                        const float axis_alpha = axis_alphas[axis_idx];
                        atomicAdd(&axis_bias_gradients[axis_ids[axis_idx]], axis_alpha * grad);
                    }
                }
            }
        }
    }
}

__global__ void sfnn_stacked_crelu_backward_kernel(
    const float* inputs,
    const float* activations,
    const float* output_gradients,
    const float* weights,
    const float* shared_weights,
    const float* axis_weights,
    const int* buckets,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    float* axis_weight_gradients,
    float* axis_bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    int compute_parameter_gradients,
    int accumulate_factorizer_gradients) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t input_gradient_len = batch * input_dim;
    size_t weight_scatter_len = batch * input_dim * output_dim;
    size_t bias_scatter_len = batch * output_dim;

    if (tid < input_gradient_len) {
        size_t sample = tid / input_dim;
        size_t in_col = tid - sample * input_dim;
        int stack_i32 = buckets[sample];
        float sum = 0.0f;
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t stack_base = stack * output_dim * input_dim;
            size_t axis_ids[8];
            size_t axis_count = sfnn_factorizer_axis_ids(
                stack,
                num_stacks,
                factorizer_king_axis_dim,
                factorizer_hand_axis_dim,
                use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
            float axis_alphas[8];
            if (has_axis != 0) {
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
            }
            for (size_t out_col = 0; out_col < output_dim; ++out_col) {
                size_t out_idx = sample * output_dim + out_col;
                float grad = crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
                if (grad != 0.0f) {
                    float weight = weights[stack_base + out_col * input_dim + in_col];
                    if (has_shared != 0) {
                        weight += shared_alpha * shared_weights[out_col * input_dim + in_col];
                    }
                    if (has_axis != 0) {
                        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                            const float axis_alpha = axis_alphas[axis_idx];
                            weight += axis_alpha *
                                axis_weights[axis_ids[axis_idx] * output_dim * input_dim + out_col * input_dim + in_col];
                        }
                    }
                    sum += grad * weight;
                }
            }
        }
        input_gradients[tid] = sum;
    }

    if (compute_parameter_gradients != 0 && tid < weight_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t input_entry = tid / output_dim;
        size_t in_col = input_entry % input_dim;
        size_t sample = input_entry / input_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t axis_ids[8];
            size_t axis_count = 0;
            float axis_alphas[8];
            if (accumulate_factorizer_gradients != 0 && has_axis != 0) {
                axis_count = sfnn_factorizer_axis_ids(
                    stack,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_axis,
                    use_hand_axis,
                    use_progress_axis,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    axis_ids);
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
            }
            size_t out_idx = sample * output_dim + out_col;
            float grad = crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
            float input_value = inputs[sample * input_dim + in_col];
            if (grad != 0.0f && input_value != 0.0f) {
                size_t weight_idx = stack * output_dim * input_dim + out_col * input_dim + in_col;
                float weight_gradient = grad * input_value;
                atomicAdd(&weight_gradients[weight_idx], weight_gradient);
                if (accumulate_factorizer_gradients != 0 && has_shared != 0) {
                    atomicAdd(&shared_weight_gradients[out_col * input_dim + in_col], shared_alpha * weight_gradient);
                }
                if (accumulate_factorizer_gradients != 0 && has_axis != 0) {
                    for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                        const float axis_alpha = axis_alphas[axis_idx];
                        atomicAdd(
                            &axis_weight_gradients[axis_ids[axis_idx] * output_dim * input_dim + out_col * input_dim + in_col],
                            axis_alpha * weight_gradient);
                    }
                }
            }
        }
    }

    if (compute_parameter_gradients != 0 && tid < bias_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t sample = tid / output_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t out_idx = sample * output_dim + out_col;
            float grad = crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
            if (grad != 0.0f) {
                atomicAdd(&bias_gradients[stack * output_dim + out_col], grad);
                if (accumulate_factorizer_gradients != 0 && has_shared != 0) {
                    atomicAdd(&shared_bias_gradients[out_col], shared_alpha * grad);
                }
                if (accumulate_factorizer_gradients != 0 && has_axis != 0) {
                    size_t axis_ids[8];
                    size_t axis_count = sfnn_factorizer_axis_ids(
                        stack,
                        num_stacks,
                        factorizer_king_axis_dim,
                        factorizer_hand_axis_dim,
                        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
                    float axis_alphas[8];
                    sfnn_factorizer_axis_alphas(
                        axis_ids,
                        axis_count,
                        num_stacks,
                        factorizer_king_axis_dim,
                        factorizer_hand_axis_dim,
                        use_king_hand_pair,
                        use_king_progress_pair,
                        use_hand_progress_pair,
                        king_axis_alpha,
                        hand_axis_alpha,
                        progress_axis_alpha,
                        pair_alpha,
                        factorizer_axis_confidences,
            axis_alphas);
                    for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                        const float axis_alpha = axis_alphas[axis_idx];
                        atomicAdd(&axis_bias_gradients[axis_ids[axis_idx] * output_dim + out_col], axis_alpha * grad);
                    }
                }
            }
        }
    }
}

__global__ void sfnn_l2_input_backward_kernel(
    const float* l1,
    const float* l2_input,
    const float* l2_input_gradients,
    float* l1_gradients,
    size_t batch,
    size_t l1_hidden,
    int l1_skip) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    size_t total = batch * l1_out;
    if (tid >= total) {
        return;
    }

    size_t sample = tid / l1_out;
    size_t col = tid - sample * l1_out;
    if (col >= l1_hidden) {
        return;
    }

    size_t l2_input_dim = l1_hidden * 2;
    size_t l2_base = sample * l2_input_dim;
    size_t square_idx = l2_base + col;
    size_t linear_idx = l2_base + l1_hidden + col;
    float value = l1[tid];
    float square_grad = crelu_pre_gradient_from_value(l2_input[square_idx], l2_input_gradients[square_idx]) *
        (2.0f * value * SFNN_PAIRWISE_SCALE);
    float linear_grad = crelu_pre_gradient_from_value(l2_input[linear_idx], l2_input_gradients[linear_idx]);
    l1_gradients[tid] += square_grad + linear_grad;
}

__global__ void sfnn_stacked_affine_backward_kernel(
    const float* inputs,
    const float* output_gradients,
    const float* weights,
    const int* buckets,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    int compute_parameter_gradients) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t input_gradient_len = batch * input_dim;
    size_t weight_scatter_len = batch * input_dim * output_dim;
    size_t bias_scatter_len = batch * output_dim;

    if (tid < input_gradient_len) {
        size_t sample = tid / input_dim;
        size_t in_col = tid - sample * input_dim;
        int stack_i32 = buckets[sample];
        float sum = 0.0f;
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t stack_base = stack * output_dim * input_dim;
            for (size_t out_col = 0; out_col < output_dim; ++out_col) {
                float grad = output_gradients[sample * output_dim + out_col];
                if (grad != 0.0f) {
                    sum += grad * weights[stack_base + out_col * input_dim + in_col];
                }
            }
        }
        input_gradients[tid] = sum;
    }

    if (compute_parameter_gradients != 0 && tid < weight_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t input_entry = tid / output_dim;
        size_t in_col = input_entry % input_dim;
        size_t sample = input_entry / input_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            float output_gradient = output_gradients[sample * output_dim + out_col];
            float input_value = inputs[sample * input_dim + in_col];
            if (output_gradient != 0.0f && input_value != 0.0f) {
                size_t weight_idx = stack * output_dim * input_dim + out_col * input_dim + in_col;
                atomicAdd(&weight_gradients[weight_idx], output_gradient * input_value);
            }
        }
    }

    if (compute_parameter_gradients != 0 && tid < bias_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t sample = tid / output_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            float grad = output_gradients[tid];
            if (grad != 0.0f) {
                atomicAdd(&bias_gradients[stack * output_dim + out_col], grad);
            }
        }
    }
}

__global__ void sfnn_grouped_l1_backward_kernel(
    const float* inputs,
    const float* output_gradients,
    const float* weights,
    const int* buckets,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t group_count,
    size_t group_input,
    size_t group_output) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t input_gradient_len = batch * input_dim;
    size_t weight_scatter_len = batch * input_dim * group_output;
    size_t bias_scatter_len = batch * output_dim;
    size_t stack_stride = group_count * group_output * group_input;

    if (tid < input_gradient_len) {
        size_t sample = tid / input_dim;
        size_t in_col = tid - sample * input_dim;
        int stack_i32 = buckets[sample];
        float sum = 0.0f;
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks &&
            in_col < group_count * group_input) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t group = in_col / group_input;
            size_t local_in = in_col - group * group_input;
            size_t group_weight_base = stack * stack_stride + group * group_output * group_input;
            for (size_t local_out = 0; local_out < group_output; ++local_out) {
                size_t out_col = group * group_output + local_out;
                float grad = output_gradients[sample * output_dim + out_col];
                if (grad != 0.0f) {
                    sum += grad * weights[group_weight_base + local_out * group_input + local_in];
                }
            }
        }
        input_gradients[tid] = sum;
    }

    if (tid < weight_scatter_len) {
        size_t local_out = tid % group_output;
        size_t input_entry = tid / group_output;
        size_t in_col = input_entry % input_dim;
        size_t sample = input_entry / input_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks &&
            in_col < group_count * group_input) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t group = in_col / group_input;
            size_t local_in = in_col - group * group_input;
            size_t out_col = group * group_output + local_out;
            float output_gradient = output_gradients[sample * output_dim + out_col];
            float input_value = inputs[sample * input_dim + in_col];
            if (output_gradient != 0.0f && input_value != 0.0f) {
                size_t weight_idx = stack * stack_stride +
                    group * group_output * group_input +
                    local_out * group_input +
                    local_in;
                atomicAdd(&weight_gradients[weight_idx], output_gradient * input_value);
            }
        }
    }

    if (tid < bias_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t sample = tid / output_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            float grad = output_gradients[tid];
            if (grad != 0.0f) {
                atomicAdd(&bias_gradients[stack * output_dim + out_col], grad);
            }
        }
    }
}

__global__ void sfnn_common_shard_l1_backward_kernel(
    const float* inputs,
    const float* output_gradients,
    const float* weights,
    const int* buckets,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t group_count,
    size_t common_input,
    size_t shard_input,
    size_t group_output) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_input = common_input + shard_input;
    size_t input_gradient_len = batch * input_dim;
    size_t weight_scatter_len = batch * output_dim * row_input;
    size_t bias_scatter_len = batch * output_dim;
    size_t stack_stride = group_count * group_output * row_input;

    if (tid < input_gradient_len) {
        size_t sample = tid / input_dim;
        size_t in_col = tid - sample * input_dim;
        int stack_i32 = buckets[sample];
        float sum = 0.0f;
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks &&
            input_dim == common_input + group_count * shard_input) {
            size_t stack = static_cast<size_t>(stack_i32);
            if (in_col < common_input) {
                for (size_t group = 0; group < group_count; ++group) {
                    size_t group_weight_base = stack * stack_stride + group * group_output * row_input;
                    for (size_t local_out = 0; local_out < group_output; ++local_out) {
                        size_t out_col = group * group_output + local_out;
                        float grad = output_gradients[sample * output_dim + out_col];
                        if (grad != 0.0f) {
                            sum += grad * weights[group_weight_base + local_out * row_input + in_col];
                        }
                    }
                }
            } else {
                size_t shard_relative = in_col - common_input;
                size_t group = shard_relative / shard_input;
                size_t local_in = shard_relative - group * shard_input;
                if (group < group_count) {
                    size_t group_weight_base = stack * stack_stride + group * group_output * row_input;
                    for (size_t local_out = 0; local_out < group_output; ++local_out) {
                        size_t out_col = group * group_output + local_out;
                        float grad = output_gradients[sample * output_dim + out_col];
                        if (grad != 0.0f) {
                            sum += grad * weights[group_weight_base + local_out * row_input + common_input + local_in];
                        }
                    }
                }
            }
        }
        input_gradients[tid] = sum;
    }

    if (tid < weight_scatter_len) {
        size_t local_in = tid % row_input;
        size_t output_entry = tid / row_input;
        size_t out_col = output_entry % output_dim;
        size_t sample = output_entry / output_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks &&
            out_col < group_count * group_output &&
            input_dim == common_input + group_count * shard_input) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t group = out_col / group_output;
            size_t local_out = out_col - group * group_output;
            size_t input_col = local_in < common_input
                ? local_in
                : common_input + group * shard_input + (local_in - common_input);
            float output_gradient = output_gradients[sample * output_dim + out_col];
            float input_value = inputs[sample * input_dim + input_col];
            if (output_gradient != 0.0f && input_value != 0.0f) {
                size_t weight_idx = stack * stack_stride +
                    group * group_output * row_input +
                    local_out * row_input +
                    local_in;
                atomicAdd(&weight_gradients[weight_idx], output_gradient * input_value);
            }
        }
    }

    if (tid < bias_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t sample = tid / output_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            float grad = output_gradients[tid];
            if (grad != 0.0f) {
                atomicAdd(&bias_gradients[stack * output_dim + out_col], grad);
            }
        }
    }
}

__global__ void sfnn_factorized_l1_backward_kernel(
    const float* inputs,
    const float* output_gradients,
    const float* weights,
    const float* shared_weights,
    const float* axis_weights,
    const int* buckets,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    float* axis_weight_gradients,
    float* axis_bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    int compute_parameter_gradients,
    int accumulate_factorizer_gradients) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t input_gradient_len = batch * input_dim;
    size_t weight_scatter_len = batch * input_dim * output_dim;
    size_t bias_scatter_len = batch * output_dim;

    if (tid < input_gradient_len) {
        size_t sample = tid / input_dim;
        size_t in_col = tid - sample * input_dim;
        int stack_i32 = buckets[sample];
        float sum = 0.0f;
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t stack_base = stack * output_dim * input_dim;
            size_t axis_ids[8];
            size_t axis_count = sfnn_factorizer_axis_ids(
                stack,
                num_stacks,
                factorizer_king_axis_dim,
                factorizer_hand_axis_dim,
                use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
            float axis_alphas[8];
            if (has_axis != 0) {
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
            }
            for (size_t out_col = 0; out_col < output_dim; ++out_col) {
                float grad = output_gradients[sample * output_dim + out_col];
                if (grad != 0.0f) {
                    float weight = weights[stack_base + out_col * input_dim + in_col];
                    if (has_shared != 0) {
                        weight += shared_alpha * shared_weights[in_col * output_dim + out_col];
                    }
                    if (has_axis != 0) {
                        for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                            const float axis_alpha = axis_alphas[axis_idx];
                            weight += axis_alpha *
                                axis_weights[axis_ids[axis_idx] * input_dim * output_dim + in_col * output_dim + out_col];
                        }
                    }
                    sum += grad * weight;
                }
            }
        }
        input_gradients[tid] = sum;
    }

    if (compute_parameter_gradients != 0 && tid < weight_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t input_entry = tid / output_dim;
        size_t in_col = input_entry % input_dim;
        size_t sample = input_entry / input_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t axis_ids[8];
            size_t axis_count = 0;
            float axis_alphas[8];
            if (accumulate_factorizer_gradients != 0 && has_axis != 0) {
                axis_count = sfnn_factorizer_axis_ids(
                    stack,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_axis,
                    use_hand_axis,
                    use_progress_axis,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    axis_ids);
                sfnn_factorizer_axis_alphas(
                    axis_ids,
                    axis_count,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    king_axis_alpha,
                    hand_axis_alpha,
                    progress_axis_alpha,
                    pair_alpha,
                    factorizer_axis_confidences,
            axis_alphas);
            }
            float output_gradient = output_gradients[sample * output_dim + out_col];
            float input_value = inputs[sample * input_dim + in_col];
            if (output_gradient != 0.0f && input_value != 0.0f) {
                float grad = output_gradient * input_value;
                size_t weight_idx = stack * output_dim * input_dim + out_col * input_dim + in_col;
                atomicAdd(&weight_gradients[weight_idx], grad);
                if (accumulate_factorizer_gradients != 0 && has_shared != 0) {
                    size_t shared_weight_idx = in_col * output_dim + out_col;
                    atomicAdd(&shared_weight_gradients[shared_weight_idx], shared_alpha * grad);
                }
                if (accumulate_factorizer_gradients != 0 && has_axis != 0) {
                    for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                        const float axis_alpha = axis_alphas[axis_idx];
                        atomicAdd(
                            &axis_weight_gradients[axis_ids[axis_idx] * input_dim * output_dim + in_col * output_dim + out_col],
                            axis_alpha * grad);
                    }
                }
            }
        }
    }
    if (compute_parameter_gradients != 0 && tid < bias_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t sample = tid / output_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            float grad = output_gradients[tid];
            if (grad != 0.0f) {
                atomicAdd(&bias_gradients[stack * output_dim + out_col], grad);
                if (accumulate_factorizer_gradients != 0 && has_shared != 0) {
                    atomicAdd(&shared_bias_gradients[out_col], shared_alpha * grad);
                }
                if (accumulate_factorizer_gradients != 0 && has_axis != 0) {
                    size_t axis_ids[8];
                    size_t axis_count = sfnn_factorizer_axis_ids(
                        stack,
                        num_stacks,
                        factorizer_king_axis_dim,
                        factorizer_hand_axis_dim,
                        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        axis_ids);
                    float axis_alphas[8];
                    sfnn_factorizer_axis_alphas(
                        axis_ids,
                        axis_count,
                        num_stacks,
                        factorizer_king_axis_dim,
                        factorizer_hand_axis_dim,
                        use_king_hand_pair,
                        use_king_progress_pair,
                        use_hand_progress_pair,
                        king_axis_alpha,
                        hand_axis_alpha,
                        progress_axis_alpha,
                        pair_alpha,
                        factorizer_axis_confidences,
            axis_alphas);
                    for (size_t axis_idx = 0; axis_idx < axis_count; ++axis_idx) {
                        const float axis_alpha = axis_alphas[axis_idx];
                        atomicAdd(&axis_bias_gradients[axis_ids[axis_idx] * output_dim + out_col], axis_alpha * grad);
                    }
                }
            }
        }
    }
}

__global__ void sfnn_pairwise_backward_kernel(
    const float* stm_l0,
    const float* nstm_l0,
    const float* combined_gradients,
    float* stm_gradients,
    float* nstm_gradients,
    size_t batch,
    size_t ft_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * ft_size;
    if (tid >= total) {
        return;
    }

    size_t pairwise = ft_size / 2;
    size_t sample = tid / ft_size;
    size_t col = tid - sample * ft_size;
    size_t pair = col % pairwise;
    size_t mate_col = col < pairwise ? pairwise + pair : pair;
    size_t l0_base = sample * ft_size;
    size_t combined_base = sample * ft_size;
    stm_gradients[tid] = combined_gradients[combined_base + pair] * stm_l0[l0_base + mate_col] * SFNN_PAIRWISE_SCALE;
    nstm_gradients[tid] =
        combined_gradients[combined_base + pairwise + pair] * nstm_l0[l0_base + mate_col] * SFNN_PAIRWISE_SCALE;
}

__device__ void sfnn_atomic_add_l0w_gradient(
    float* gradients,
    size_t feature,
    size_t input_size,
    size_t rows,
    size_t row,
    float value);

__global__ void sfnn_pairwise_l0_sparse_backward_kernel(
    const int* stm_indices,
    const int* nstm_indices,
    const float* stm_activations,
    const float* nstm_activations,
    const float* combined_gradients,
    float* l0w_gradients,
    float* l0b_gradients,
    size_t batch,
    size_t max_active,
    size_t input_size,
    size_t ft_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pairwise = ft_size / 2;
    size_t total = batch * pairwise;
    if (tid >= total) {
        return;
    }

    size_t pair = tid % pairwise;
    size_t sample = tid / pairwise;
    size_t row0 = pair;
    size_t row1 = pairwise + pair;
    size_t l0_base = sample * ft_size;
    size_t sparse_base = sample * max_active;

    float stm0 = stm_activations[l0_base + row0];
    float stm1 = stm_activations[l0_base + row1];
    float nstm0 = nstm_activations[l0_base + row0];
    float nstm1 = nstm_activations[l0_base + row1];
    float stm_pair_grad = combined_gradients[l0_base + pair] * SFNN_PAIRWISE_SCALE;
    float nstm_pair_grad = combined_gradients[l0_base + pairwise + pair] * SFNN_PAIRWISE_SCALE;
    float stm_grad0 = crelu_pre_gradient_from_value(stm0, stm_pair_grad * stm1);
    float stm_grad1 = crelu_pre_gradient_from_value(stm1, stm_pair_grad * stm0);
    float nstm_grad0 = crelu_pre_gradient_from_value(nstm0, nstm_pair_grad * nstm1);
    float nstm_grad1 = crelu_pre_gradient_from_value(nstm1, nstm_pair_grad * nstm0);
    float bias_grad0 = stm_grad0 + nstm_grad0;
    float bias_grad1 = stm_grad1 + nstm_grad1;

    if (bias_grad0 == 0.0f && bias_grad1 == 0.0f) {
        return;
    }
    if (bias_grad0 != 0.0f) {
        atomicAdd(&l0b_gradients[row0], bias_grad0);
    }
    if (bias_grad1 != 0.0f) {
        atomicAdd(&l0b_gradients[row1], bias_grad1);
    }

    for (size_t slot = 0; slot < max_active; ++slot) {
        int stm_feature = stm_indices[sparse_base + slot];
        if (stm_feature >= 0 && static_cast<size_t>(stm_feature) < input_size) {
            size_t feature = static_cast<size_t>(stm_feature);
            if (stm_grad0 != 0.0f) {
                sfnn_atomic_add_l0w_gradient(l0w_gradients, feature, input_size, ft_size, row0, stm_grad0);
            }
            if (stm_grad1 != 0.0f) {
                sfnn_atomic_add_l0w_gradient(l0w_gradients, feature, input_size, ft_size, row1, stm_grad1);
            }
        }

        int nstm_feature = nstm_indices[sparse_base + slot];
        if (nstm_feature >= 0 && static_cast<size_t>(nstm_feature) < input_size) {
            size_t feature = static_cast<size_t>(nstm_feature);
            if (nstm_grad0 != 0.0f) {
                sfnn_atomic_add_l0w_gradient(l0w_gradients, feature, input_size, ft_size, row0, nstm_grad0);
            }
            if (nstm_grad1 != 0.0f) {
                sfnn_atomic_add_l0w_gradient(l0w_gradients, feature, input_size, ft_size, row1, nstm_grad1);
            }
        }
    }
}

__global__ void sfnn_pairwise_l0_pregrad_kernel(
    const float* stm_activations,
    const float* nstm_activations,
    const float* combined_gradients,
    float* stm_pre_gradients,
    float* nstm_pre_gradients,
    float* l0b_gradients,
    size_t batch,
    size_t ft_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pairwise = ft_size / 2;
    size_t total = batch * pairwise;
    if (tid >= total) {
        return;
    }

    size_t pair = tid % pairwise;
    size_t sample = tid / pairwise;
    size_t row0 = pair;
    size_t row1 = pairwise + pair;
    size_t l0_base = sample * ft_size;

    float stm0 = stm_activations[l0_base + row0];
    float stm1 = stm_activations[l0_base + row1];
    float nstm0 = nstm_activations[l0_base + row0];
    float nstm1 = nstm_activations[l0_base + row1];
    float stm_pair_grad = combined_gradients[l0_base + pair] * SFNN_PAIRWISE_SCALE;
    float nstm_pair_grad = combined_gradients[l0_base + pairwise + pair] * SFNN_PAIRWISE_SCALE;
    float stm_grad0 = crelu_pre_gradient_from_value(stm0, stm_pair_grad * stm1);
    float stm_grad1 = crelu_pre_gradient_from_value(stm1, stm_pair_grad * stm0);
    float nstm_grad0 = crelu_pre_gradient_from_value(nstm0, nstm_pair_grad * nstm1);
    float nstm_grad1 = crelu_pre_gradient_from_value(nstm1, nstm_pair_grad * nstm0);

    stm_pre_gradients[l0_base + row0] = stm_grad0;
    stm_pre_gradients[l0_base + row1] = stm_grad1;
    nstm_pre_gradients[l0_base + row0] = nstm_grad0;
    nstm_pre_gradients[l0_base + row1] = nstm_grad1;

    float bias_grad0 = stm_grad0 + nstm_grad0;
    float bias_grad1 = stm_grad1 + nstm_grad1;
    if (bias_grad0 != 0.0f) {
        atomicAdd(&l0b_gradients[row0], bias_grad0);
    }
    if (bias_grad1 != 0.0f) {
        atomicAdd(&l0b_gradients[row1], bias_grad1);
    }
}

__global__ void sfnn_inverse_feature_counts_kernel(
    const int* indices,
    int* counts,
    size_t total_entries,
    size_t max_active,
    size_t n_features) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_entries) {
        return;
    }
    int feature = indices[tid];
    if (feature >= 0 && static_cast<size_t>(feature) < n_features) {
        atomicAdd(&counts[feature], 1);
    }
}

__global__ void sfnn_inverse_prefix_sum_block_local_kernel(
    const int* counts,
    int* offsets,
    int* block_sums,
    size_t n_features) {
    __shared__ int partials[1024];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    int value = idx < n_features ? counts[idx] : 0;
    partials[tid] = value;
    __syncthreads();

    for (size_t stride = 1; stride < blockDim.x; stride <<= 1) {
        int add = tid >= stride ? partials[tid - stride] : 0;
        __syncthreads();
        partials[tid] += add;
        __syncthreads();
    }

    if (idx < n_features) {
        offsets[idx] = tid == 0 ? 0 : partials[tid - 1];
    }
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = partials[tid];
    }
}

__global__ void sfnn_inverse_prefix_sum_small_kernel(
    const int* block_sums,
    int* block_offsets,
    size_t num_blocks) {
    __shared__ int partials[1024];
    size_t tid = threadIdx.x;
    int value = tid < num_blocks ? block_sums[tid] : 0;
    partials[tid] = value;
    __syncthreads();

    for (size_t stride = 1; stride < blockDim.x; stride <<= 1) {
        int add = tid >= stride ? partials[tid - stride] : 0;
        __syncthreads();
        partials[tid] += add;
        __syncthreads();
    }

    if (tid < num_blocks) {
        block_offsets[tid] = tid == 0 ? 0 : partials[tid - 1];
    }
    if (num_blocks > 0 && tid == num_blocks - 1) {
        block_offsets[num_blocks] = partials[tid];
    }
}

__global__ void sfnn_inverse_prefix_add_block_offsets_kernel(
    int* offsets,
    const int* block_offsets,
    size_t n_features,
    size_t num_blocks) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    if (idx < n_features) {
        offsets[idx] += block_offsets[blockIdx.x];
    }
    if (blockIdx.x == 0 && tid == 0) {
        offsets[n_features] = block_offsets[num_blocks];
    }
}

__global__ void sfnn_inverse_scatter_positions_kernel(
    const int* indices,
    const int* offsets,
    int* write_counters,
    int* positions,
    size_t total_entries,
    size_t max_active,
    size_t n_features) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_entries) {
        return;
    }
    size_t sample = tid / max_active;
    int feature = indices[tid];
    if (feature >= 0 && static_cast<size_t>(feature) < n_features) {
        int pos = atomicAdd(&write_counters[feature], 1);
        positions[offsets[feature] + pos] = static_cast<int>(sample);
    }
}

__global__ void sfnn_inverse_gather_l0w_gradients_kernel(
    const float* pre_gradients,
    const int* positions,
    const int* offsets,
    float* l0w_gradients,
    size_t n_features,
    size_t ft_size,
    int add_to_existing) {
    size_t feature = blockIdx.x;
    size_t row = blockIdx.y * blockDim.x + threadIdx.x;
    if (feature >= n_features || row >= ft_size) {
        return;
    }

    size_t off_start = static_cast<size_t>(offsets[feature]);
    size_t off_end = static_cast<size_t>(offsets[feature + 1]);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    size_t i = off_start;
    size_t unroll_end = off_end >= off_start + 3 ? off_end - 3 : off_start;
    while (i < unroll_end) {
        size_t sample0 = static_cast<size_t>(positions[i]);
        size_t sample1 = static_cast<size_t>(positions[i + 1]);
        size_t sample2 = static_cast<size_t>(positions[i + 2]);
        size_t sample3 = static_cast<size_t>(positions[i + 3]);
        sum0 += pre_gradients[sample0 * ft_size + row];
        sum1 += pre_gradients[sample1 * ft_size + row];
        sum2 += pre_gradients[sample2 * ft_size + row];
        sum3 += pre_gradients[sample3 * ft_size + row];
        i += 4;
    }
    while (i < off_end) {
        size_t sample = static_cast<size_t>(positions[i]);
        sum0 += pre_gradients[sample * ft_size + row];
        ++i;
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    size_t weight_idx = feature * ft_size + row;
    if (add_to_existing != 0) {
        l0w_gradients[weight_idx] += sum;
    } else {
        l0w_gradients[weight_idx] = sum;
    }
}

__global__ void sfnn_reduce_halfka2_virtual_l0w_gradients_kernel(
    float* l0w_gradients,
    size_t ft_size) {
    size_t piece = blockIdx.x;
    size_t row = blockIdx.y * blockDim.x + threadIdx.x;
    if (piece >= SFNN_HALFKA2_PIECE_INPUTS || row >= ft_size) {
        return;
    }

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    size_t feature = piece;
    const size_t stride = SFNN_HALFKA2_PIECE_INPUTS;
    const size_t unroll_end = SFNN_HALFKA2_BASE_INPUT_SIZE >= piece + 3 * stride
        ? SFNN_HALFKA2_BASE_INPUT_SIZE - 3 * stride
        : piece;
    while (feature < unroll_end) {
        sum0 += l0w_gradients[feature * ft_size + row];
        sum1 += l0w_gradients[(feature + stride) * ft_size + row];
        sum2 += l0w_gradients[(feature + 2 * stride) * ft_size + row];
        sum3 += l0w_gradients[(feature + 3 * stride) * ft_size + row];
        feature += 4 * stride;
    }
    while (feature < SFNN_HALFKA2_BASE_INPUT_SIZE) {
        sum0 += l0w_gradients[feature * ft_size + row];
        feature += stride;
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    size_t virtual_feature = SFNN_HALFKA2_BASE_INPUT_SIZE + piece;
    l0w_gradients[virtual_feature * ft_size + row] = sum;
}

__device__ void sfnn_atomic_add_l0w_gradient(float* gradients, size_t feature, size_t input_size, size_t rows, size_t row, float value) {
    size_t weight_idx = feature * rows + row;
    atomicAdd(&gradients[weight_idx], value);
    size_t virtual_feature = 0;
    if (sfnn_factorized_virtual_feature(feature, input_size, &virtual_feature)) {
        atomicAdd(&gradients[virtual_feature * rows + row], value);
    }
}

__global__ void sfnn_l0_sparse_backward_kernel(
    const int* stm_indices,
    const int* nstm_indices,
    const float* stm_activations,
    const float* nstm_activations,
    const float* stm_output_gradients,
    const float* nstm_output_gradients,
    float* stm_pre_gradients,
    float* nstm_pre_gradients,
    float* l0w_gradients,
    float* l0b_gradients,
    size_t batch,
    size_t max_active,
    size_t input_size,
    size_t ft_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t l0_len = batch * ft_size;
    if (tid >= l0_len) {
        return;
    }

    size_t row = tid % ft_size;
    size_t sample = tid / ft_size;
    size_t sparse_base = sample * max_active;
    float stm_grad = crelu_pre_gradient_from_value(stm_activations[tid], stm_output_gradients[tid]);
    float nstm_grad = crelu_pre_gradient_from_value(nstm_activations[tid], nstm_output_gradients[tid]);
    stm_pre_gradients[tid] = stm_grad;
    nstm_pre_gradients[tid] = nstm_grad;

    if (stm_grad != 0.0f || nstm_grad != 0.0f) {
        atomicAdd(&l0b_gradients[row], stm_grad + nstm_grad);
    } else {
        return;
    }

    for (size_t slot = 0; slot < max_active; ++slot) {
        int stm_feature = stm_indices[sparse_base + slot];
        if (stm_grad != 0.0f && stm_feature >= 0 && static_cast<size_t>(stm_feature) < input_size) {
            sfnn_atomic_add_l0w_gradient(l0w_gradients, static_cast<size_t>(stm_feature), input_size, ft_size, row, stm_grad);
        }

        int nstm_feature = nstm_indices[sparse_base + slot];
        if (nstm_grad != 0.0f && nstm_feature >= 0 && static_cast<size_t>(nstm_feature) < input_size) {
            sfnn_atomic_add_l0w_gradient(
                l0w_gradients, static_cast<size_t>(nstm_feature), input_size, ft_size, row, nstm_grad);
        }
    }
}

__global__ void dense_output_backward_kernel(
    const float* inputs,
    const float* output_gradients,
    const float* weights,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradient,
    size_t batch,
    size_t input_len) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t input_gradient_len = batch * input_len;

    if (tid < input_gradient_len) {
        size_t sample = tid / input_len;
        size_t row = tid - sample * input_len;
        input_gradients[tid] = output_gradients[sample] * weights[row];
    }

    if (tid < input_len) {
        float sum = 0.0f;
        for (size_t sample = 0; sample < batch; ++sample) {
            sum += output_gradients[sample] * inputs[sample * input_len + tid];
        }
        weight_gradients[tid] = sum;
    }

    if (tid == 0) {
        float sum = 0.0f;
        for (size_t sample = 0; sample < batch; ++sample) {
            sum += output_gradients[sample];
        }
        bias_gradient[0] = sum;
    }
}

__global__ void dense_crelu_backward_kernel(
    const float* inputs,
    const float* activations,
    const float* output_gradients,
    const float* weights,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t input_gradient_len = batch * input_dim;
    size_t weight_len = input_dim * output_dim;

    if (tid < input_gradient_len) {
        size_t sample = tid / input_dim;
        size_t in_col = tid - sample * input_dim;
        float sum = 0.0f;
        for (size_t out_col = 0; out_col < output_dim; ++out_col) {
            size_t out_idx = sample * output_dim + out_col;
            float grad = crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
            sum += grad * weights[in_col * output_dim + out_col];
        }
        input_gradients[tid] = sum;
    }

    if (tid < weight_len) {
        size_t in_col = tid / output_dim;
        size_t out_col = tid - in_col * output_dim;
        float sum = 0.0f;
        for (size_t sample = 0; sample < batch; ++sample) {
            size_t out_idx = sample * output_dim + out_col;
            float grad = crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
            sum += grad * inputs[sample * input_dim + in_col];
        }
        weight_gradients[tid] = sum;
    }

    if (tid < output_dim) {
        float sum = 0.0f;
        for (size_t sample = 0; sample < batch; ++sample) {
            size_t out_idx = sample * output_dim + tid;
            sum += crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
        }
        bias_gradients[tid] = sum;
    }
}

__global__ void dense_crelu_pre_gradient_kernel(
    const float* activations,
    float* gradients,
    size_t batch,
    size_t output_dim) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t len = batch * output_dim;
    if (tid >= len) {
        return;
    }
    gradients[tid] = crelu_pre_gradient_from_value(activations[tid], gradients[tid]);
}

__global__ void dense_bias_sum_kernel(
    const float* gradients,
    float* bias_gradients,
    size_t batch,
    size_t output_dim) {
    constexpr int threads = 256;
    __shared__ float partial[threads];

    size_t out_col = blockIdx.x;
    size_t tid = threadIdx.x;
    float sum = 0.0f;
    for (size_t sample = tid; sample < batch; sample += threads) {
        sum += gradients[sample * output_dim + out_col];
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < static_cast<size_t>(stride)) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        bias_gradients[out_col] = partial[0];
    }
}

__global__ void l0_crelu_backward_inplace_kernel(
    const float* activations,
    float* gradients,
    size_t len) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    gradients[tid] = crelu_pre_gradient_from_value(activations[tid], gradients[tid]);
}

int launch_dense_crelu_backward_gemm(
    BulletOuCudaCppContext* ctx,
    const char* label,
    const float* inputs,
    const float* activations,
    float* output_gradients,
    const float* weights,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim) {
    constexpr int threads = 256;
    int blocks = 0;

    if (block_count_1d(batch * output_dim, threads, &blocks, "dense_crelu_pre_gradient_kernel") != 0) {
        return -1;
    }
    dense_crelu_pre_gradient_kernel<<<blocks, threads, 0, ctx->stream>>>(activations, output_gradients, batch, output_dim);
    if (check_kernel_launch("dense_crelu_pre_gradient_kernel launch") != 0) {
        return -1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        ctx->blas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(input_dim),
        static_cast<int>(batch),
        static_cast<int>(output_dim),
        &alpha,
        weights,
        static_cast<int>(output_dim),
        output_gradients,
        static_cast<int>(output_dim),
        &beta,
        input_gradients,
        static_cast<int>(input_dim));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return fail_blas(label, status);
    }

    status = cublasSgemm(
        ctx->blas,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        static_cast<int>(output_dim),
        static_cast<int>(input_dim),
        static_cast<int>(batch),
        &alpha,
        output_gradients,
        static_cast<int>(output_dim),
        inputs,
        static_cast<int>(input_dim),
        &beta,
        weight_gradients,
        static_cast<int>(output_dim));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return fail_blas(label, status);
    }

    dense_bias_sum_kernel<<<static_cast<int>(output_dim), threads, 0, ctx->stream>>>(
        output_gradients, bias_gradients, batch, output_dim);
    if (check_kernel_launch("dense_bias_sum_kernel launch") != 0) {
        return -1;
    }

    return 0;
}

__global__ void nnue_l0_crelu_backward_kernel(
    const float* combined_gradients,
    const float* stm_activations,
    const float* nstm_activations,
    float* stm_gradients,
    float* nstm_gradients,
    size_t batch,
    size_t l1) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t combined_stride = l1 * 2;
    size_t combined_len = batch * combined_stride;
    if (tid >= combined_len) {
        return;
    }

    size_t sample = tid / combined_stride;
    size_t col = tid - sample * combined_stride;
    if (col < l1) {
        size_t perspective_idx = sample * l1 + col;
        stm_gradients[perspective_idx] =
            crelu_pre_gradient_from_value(stm_activations[perspective_idx], combined_gradients[tid]);
    } else {
        size_t row = col - l1;
        size_t perspective_idx = sample * l1 + row;
        nstm_gradients[perspective_idx] =
            crelu_pre_gradient_from_value(nstm_activations[perspective_idx], combined_gradients[tid]);
    }
}

__global__ void nnue_l0_sparse_zero_gradients_kernel(
    float* l0w_gradients,
    float* l0b_gradients,
    size_t input_size,
    size_t l1) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t weight_len = nnue_l0w_len_for_shape(input_size, l1);

    if (tid < weight_len) {
        l0w_gradients[tid] = 0.0f;
    }
    if (tid < l1) {
        l0b_gradients[tid] = 0.0f;
    }
}

__global__ void nnue_l0_sparse_backward_kernel(
    const int* stm_indices,
    const int* nstm_indices,
    const float* stm_gradients,
    const float* nstm_gradients,
    float* l0w_gradients,
    size_t batch,
    size_t max_active,
    size_t input_size,
    size_t l1) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t scatter_len = batch * max_active * l1;
    if (tid >= scatter_len) {
        return;
    }

    size_t row = tid % l1;
    size_t sparse_entry = tid / l1;
    size_t sample = sparse_entry / max_active;
    size_t slot = sparse_entry - sample * max_active;
    size_t sparse_base = sample * max_active + slot;

    int stm_feature = stm_indices[sparse_base];
    if (stm_feature >= 0 && static_cast<size_t>(stm_feature) < input_size) {
        size_t base_feature = 0;
        size_t virtual_feature = 0;
        float value = stm_gradients[sample * l1 + row];
        if (value != 0.0f) {
            if (nnue_halfkp_factorized_feature(
                    static_cast<size_t>(stm_feature), input_size, &base_feature, &virtual_feature)) {
                atomicAdd(&l0w_gradients[base_feature * l1 + row], value);
                atomicAdd(&l0w_gradients[virtual_feature * l1 + row], value);
            } else {
                atomicAdd(&l0w_gradients[static_cast<size_t>(stm_feature) * l1 + row], value);
            }
        }
    }

    int nstm_feature = nstm_indices[sparse_base];
    if (nstm_feature >= 0 && static_cast<size_t>(nstm_feature) < input_size) {
        size_t base_feature = 0;
        size_t virtual_feature = 0;
        float value = nstm_gradients[sample * l1 + row];
        if (value != 0.0f) {
            if (nnue_halfkp_factorized_feature(
                    static_cast<size_t>(nstm_feature), input_size, &base_feature, &virtual_feature)) {
                atomicAdd(&l0w_gradients[base_feature * l1 + row], value);
                atomicAdd(&l0w_gradients[virtual_feature * l1 + row], value);
            } else {
                atomicAdd(&l0w_gradients[static_cast<size_t>(nstm_feature) * l1 + row], value);
            }
        }
    }
}

__global__ void nnue_l0_bias_backward_kernel(
    const float* stm_gradients,
    const float* nstm_gradients,
    float* l0b_gradients,
    size_t batch,
    size_t l1) {
    constexpr int threads = 256;
    __shared__ float partial[threads];

    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    float sum = 0.0f;
    for (size_t sample = tid; sample < batch; sample += threads) {
        sum += stm_gradients[sample * l1 + row] + nstm_gradients[sample * l1 + row];
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < static_cast<size_t>(stride)) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        l0b_gradients[row] = partial[0];
    }
}

int sync_after_kernel(const char* launch_label, const char* sync_label) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return fail(launch_label, status);
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return fail(sync_label, status);
    }
    return 0;
}

int check_kernel_launch(const char* launch_label) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return fail(launch_label, status);
    }
    return 0;
}

int validate_host_ptr(const void* ptr, size_t len, const char* name) {
    if (len != 0 && ptr == nullptr) {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer), "%s must not be null when len > 0", name);
        return fail_message(buffer);
    }
    return 0;
}

int validate_context(BulletOuCudaCppContext* ctx) {
    if (ctx == nullptr) {
        return fail_message("context must not be null");
    }
    if (ctx->stream == nullptr) {
        return fail_message("context stream must not be null");
    }
    return 0;
}

int validate_buffer(BulletOuCudaCppContext* ctx, BulletOuCudaCppF32Buffer* buffer, size_t len, const char* name) {
    if (validate_context(ctx) != 0) {
        return -1;
    }
    if (buffer == nullptr) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s buffer must not be null", name);
        return fail_message(message);
    }
    if (buffer->ptr == nullptr && buffer->len != 0) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s device pointer must not be null", name);
        return fail_message(message);
    }
    if (buffer->device != ctx->device) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s buffer belongs to device %d, context is device %d", name, buffer->device, ctx->device);
        return fail_message(message);
    }
    if (buffer->len < len) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s buffer length %zu is smaller than requested length %zu", name, buffer->len, len);
        return fail_message(message);
    }
    return 0;
}

int validate_i32_buffer(BulletOuCudaCppContext* ctx, BulletOuCudaCppI32Buffer* buffer, size_t len, const char* name) {
    if (validate_context(ctx) != 0) {
        return -1;
    }
    if (buffer == nullptr) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s buffer must not be null", name);
        return fail_message(message);
    }
    if (buffer->ptr == nullptr && buffer->len != 0) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s device pointer must not be null", name);
        return fail_message(message);
    }
    if (buffer->device != ctx->device) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s buffer belongs to device %d, context is device %d", name, buffer->device, ctx->device);
        return fail_message(message);
    }
    if (buffer->len < len) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s buffer length %zu is smaller than requested length %zu", name, buffer->len, len);
        return fail_message(message);
    }
    return 0;
}

int validate_pinned_f32_buffer(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppPinnedF32Buffer* buffer,
    size_t len,
    const char* name) {
    if (validate_context(ctx) != 0) {
        return -1;
    }
    if (buffer == nullptr) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s pinned buffer must not be null", name);
        return fail_message(message);
    }
    if (buffer->ptr == nullptr && buffer->len != 0) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s pinned host pointer must not be null", name);
        return fail_message(message);
    }
    if (buffer->device != ctx->device) {
        char message[256];
        std::snprintf(
            message,
            sizeof(message),
            "%s pinned buffer belongs to device %d, context is device %d",
            name,
            buffer->device,
            ctx->device);
        return fail_message(message);
    }
    if (buffer->len < len) {
        char message[256];
        std::snprintf(
            message,
            sizeof(message),
            "%s pinned buffer length %zu is smaller than requested length %zu",
            name,
            buffer->len,
            len);
        return fail_message(message);
    }
    return 0;
}

int validate_pinned_i32_buffer(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppPinnedI32Buffer* buffer,
    size_t len,
    const char* name) {
    if (validate_context(ctx) != 0) {
        return -1;
    }
    if (buffer == nullptr) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s pinned buffer must not be null", name);
        return fail_message(message);
    }
    if (buffer->ptr == nullptr && buffer->len != 0) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s pinned host pointer must not be null", name);
        return fail_message(message);
    }
    if (buffer->device != ctx->device) {
        char message[256];
        std::snprintf(
            message,
            sizeof(message),
            "%s pinned buffer belongs to device %d, context is device %d",
            name,
            buffer->device,
            ctx->device);
        return fail_message(message);
    }
    if (buffer->len < len) {
        char message[256];
        std::snprintf(
            message,
            sizeof(message),
            "%s pinned buffer length %zu is smaller than requested length %zu",
            name,
            buffer->len,
            len);
        return fail_message(message);
    }
    return 0;
}

int set_context_device(BulletOuCudaCppContext* ctx) {
    if (validate_context(ctx) != 0) {
        return -1;
    }
    cudaError_t status = cudaSetDevice(ctx->device);
    if (status != cudaSuccess) {
        return fail("cudaSetDevice", status);
    }
    return 0;
}

int validate_event(BulletOuCudaCppContext* ctx, BulletOuCudaCppEvent* event, const char* name) {
    if (validate_context(ctx) != 0) {
        return -1;
    }
    if (event == nullptr) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s event must not be null", name);
        return fail_message(message);
    }
    if (event->event == nullptr) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s event handle must not be null", name);
        return fail_message(message);
    }
    if (event->device != ctx->device) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s event belongs to device %d, context is device %d", name, event->device, ctx->device);
        return fail_message(message);
    }
    return 0;
}

int validate_graph(BulletOuCudaCppContext* ctx, BulletOuCudaCppGraphExec* graph, const char* name) {
    if (validate_context(ctx) != 0) {
        return -1;
    }
    if (graph == nullptr) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s graph must not be null", name);
        return fail_message(message);
    }
    if (graph->exec == nullptr) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s graph exec handle must not be null", name);
        return fail_message(message);
    }
    if (graph->device != ctx->device) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s graph belongs to device %d, context is device %d", name, graph->device, ctx->device);
        return fail_message(message);
    }
    return 0;
}

int block_count_1d(size_t len, int threads, int* blocks, const char* label) {
    if (len == 0) {
        *blocks = 0;
        return 0;
    }
    size_t computed = (len + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
    if (computed > static_cast<size_t>(INT_MAX)) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s launch grid too large: %zu blocks", label, computed);
        return fail_message(message);
    }
    *blocks = static_cast<int>(computed);
    return 0;
}

constexpr size_t SFNN_BACKWARD_PROFILE_MS_LEN = 7;

struct SfnnBackwardProfileEvents {
    bool enabled = false;
    float* out_ms = nullptr;
    cudaEvent_t events[7] = {};

    ~SfnnBackwardProfileEvents() {
        for (cudaEvent_t event : events) {
            if (event != nullptr) {
                cudaEventDestroy(event);
            }
        }
    }

    int init(BulletOuCudaCppContext* ctx, float* out, size_t out_len) {
        if (out == nullptr) {
            return 0;
        }
        if (out_len < SFNN_BACKWARD_PROFILE_MS_LEN) {
            return fail_message("SFNN backward profile output must have at least 7 floats");
        }
        enabled = true;
        out_ms = out;
        for (size_t i = 0; i < SFNN_BACKWARD_PROFILE_MS_LEN; ++i) {
            out_ms[i] = 0.0f;
        }
        for (size_t i = 0; i < 7; ++i) {
            cudaError_t status = cudaEventCreate(&events[i]);
            if (status != cudaSuccess) {
                return fail("cudaEventCreate SFNN backward profile", status);
            }
        }
        return record(0, ctx, "SFNN backward profile start");
    }

    int record(size_t idx, BulletOuCudaCppContext* ctx, const char* label) {
        if (!enabled) {
            return 0;
        }
        cudaError_t status = cudaEventRecord(events[idx], ctx->stream);
        if (status != cudaSuccess) {
            return fail(label, status);
        }
        return 0;
    }

    int finish() {
        if (!enabled) {
            return 0;
        }
        cudaError_t status = cudaEventSynchronize(events[6]);
        if (status != cudaSuccess) {
            return fail("cudaEventSynchronize SFNN backward profile", status);
        }
        const size_t ranges[SFNN_BACKWARD_PROFILE_MS_LEN][2] = {
            {0, 1}, // zero parameter gradients
            {1, 2}, // L3 backward
            {2, 3}, // L2 backward
            {3, 4}, // L2-input backward
            {4, 5}, // L1 backward
            {5, 6}, // pairwise/L0 backward
            {0, 6}, // total backward
        };
        for (size_t i = 0; i < SFNN_BACKWARD_PROFILE_MS_LEN; ++i) {
            status = cudaEventElapsedTime(&out_ms[i], events[ranges[i][0]], events[ranges[i][1]]);
            if (status != cudaSuccess) {
                return fail("cudaEventElapsedTime SFNN backward profile", status);
            }
        }
        return 0;
    }
};

int validate_nnue_shape(size_t input_size, size_t l1, size_t l2, size_t l3, size_t batch, size_t max_active) {
    if (input_size == 0 || l1 == 0 || l2 == 0 || l3 == 0) {
        return fail_message("NNUE shape dimensions must be greater than zero");
    }
    if (batch == 0) {
        return fail_message("NNUE batch size must be greater than zero");
    }
    if (max_active == 0) {
        return fail_message("NNUE max_active must be greater than zero");
    }
    return 0;
}

int validate_sfnn_shape(
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    size_t batch,
    size_t max_active) {
    if (input_size == 0 || ft_size == 0 || l1_hidden == 0 || l2_size == 0 || num_stacks == 0 || l1_group_count == 0) {
        return fail_message("SFNN shape dimensions must be greater than zero");
    }
    if ((ft_size % 2) != 0) {
        return fail_message("SFNN ft_size must be even");
    }
    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    const bool common_shard_l1 = sfnn_is_common_shard_l1_shape(l1_common_size, l1_shard_size);
    if (common_shard_l1 &&
        (l1_shard_size == 0 || l1_group_count <= 1 ||
            l1_common_size + l1_shard_size * l1_group_count != ft_size ||
            (l1_out % l1_group_count) != 0 ||
            (l1_common_size % 64) != 0 || (l1_shard_size % 64) != 0)) {
        return fail_message("SFNN common+shard L1 requires common + shard * group == ft_size, fc0 output count divisible by group, shard > 0, and common/shard multiples of 64");
    }
    if (!common_shard_l1 && sfnn_is_grouped_l1_shape(l1_group_count) && (ft_size % l1_group_count != 0 || (l1_out % l1_group_count) != 0)) {
        return fail_message("SFNN grouped L1 requires ft_size and fc0 output count to be divisible by l1_group_count");
    }
    if (factorizer_king_axis_dim != 0 && factorizer_king_axis_dim > SIZE_MAX / factorizer_king_axis_dim) {
        return fail_message("SFNN king axis bucket count overflow");
    }
    if (factorizer_hand_axis_dim != 0 && factorizer_hand_axis_dim > SIZE_MAX / factorizer_hand_axis_dim) {
        return fail_message("SFNN hand axis bucket count overflow");
    }
    const size_t king_bucket_count =
        factorizer_king_axis_dim == 0 ? 1 : factorizer_king_axis_dim * factorizer_king_axis_dim;
    const size_t hand_bucket_count =
        factorizer_hand_axis_dim == 0 ? 1 : factorizer_hand_axis_dim * factorizer_hand_axis_dim;
    if (king_bucket_count != 0 && hand_bucket_count > SIZE_MAX / king_bucket_count) {
        return fail_message("SFNN factorizer bucket count overflow");
    }
    const size_t expected_stacks = king_bucket_count * hand_bucket_count;
    if ((factorizer_king_axis_dim != 0 || factorizer_hand_axis_dim != 0) && (num_stacks % expected_stacks) != 0) {
        return fail_message("SFNN factorizer axis dimensions do not divide num_stacks");
    }
    if (batch == 0) {
        return fail_message("SFNN batch size must be greater than zero");
    }
    if (max_active == 0) {
        return fail_message("SFNN max_active must be greater than zero");
    }
    return 0;
}

int launch_dense_forward_gemm_raw(
    BulletOuCudaCppContext* ctx,
    const char* label,
    const float* inputs,
    const float* weights,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    float beta) {
    const float alpha = 1.0f;
    cublasStatus_t status = cublasSgemm(
        ctx->blas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(output_dim),
        static_cast<int>(batch),
        static_cast<int>(input_dim),
        &alpha,
        weights,
        static_cast<int>(output_dim),
        inputs,
        static_cast<int>(input_dim),
        &beta,
        output,
        static_cast<int>(output_dim));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return fail_blas(label, status);
    }
    return 0;
}

int launch_dense_forward_gemm(
    BulletOuCudaCppContext* ctx,
    const char* label,
    const float* inputs,
    const float* weights,
    const float* bias,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    int apply_crelu) {
    if (launch_dense_forward_gemm_raw(
            ctx,
            label,
            inputs,
            weights,
            output,
            batch,
            input_dim,
            output_dim,
            0.0f) != 0) {
        return -1;
    }

    constexpr int threads = 256;
    int blocks = 0;
    if (block_count_1d(batch * output_dim, threads, &blocks, "dense_add_bias_kernel") != 0) {
        return -1;
    }
    dense_add_bias_kernel<<<blocks, threads, 0, ctx->stream>>>(output, bias, batch, output_dim, apply_crelu);
    if (check_kernel_launch("dense_add_bias_kernel launch") != 0) {
        return -1;
    }
    return 0;
}

int launch_nnue_forward_kernels(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t l1,
    size_t l2,
    size_t l3,
    size_t batch,
    size_t max_active,
    const int* stm_indices,
    const int* nstm_indices,
    const float* l0w,
    const float* l0b,
    const float* l1w,
    const float* l1b,
    const float* l2w,
    const float* l2b,
    const float* outw,
    const float* outb,
    float* stm_l0,
    float* nstm_l0,
    float* combined,
    float* hidden1,
    float* hidden2,
    float* output) {
    if (validate_nnue_shape(input_size, l1, l2, l3, batch, max_active) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }

    constexpr int threads = 256;
    int blocks = 0;

    if (block_count_1d(batch * l1, threads, &blocks, "nnue_sparse_l0_crelu_kernel") != 0) {
        return -1;
    }
    nnue_sparse_l0_crelu_kernel<<<blocks, threads, 0, ctx->stream>>>(
        stm_indices, l0w, l0b, stm_l0, batch, max_active, input_size, l1);
    if (check_kernel_launch("nnue_sparse_l0_crelu_kernel stm launch") != 0) {
        return -1;
    }
    nnue_sparse_l0_crelu_kernel<<<blocks, threads, 0, ctx->stream>>>(
        nstm_indices, l0w, l0b, nstm_l0, batch, max_active, input_size, l1);
    if (check_kernel_launch("nnue_sparse_l0_crelu_kernel nstm launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l1 * 2, threads, &blocks, "nnue_concat_l0_kernel") != 0) {
        return -1;
    }
    nnue_concat_l0_kernel<<<blocks, threads, 0, ctx->stream>>>(stm_l0, nstm_l0, combined, batch, l1);
    if (check_kernel_launch("nnue_concat_l0_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l2, threads, &blocks, "nnue_dense_l1_crelu_kernel") != 0) {
        return -1;
    }
    nnue_dense_crelu_kernel<<<blocks, threads, 0, ctx->stream>>>(combined, l1w, l1b, hidden1, batch, l1 * 2, l2);
    if (check_kernel_launch("nnue_dense_l1_crelu_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l3, threads, &blocks, "nnue_dense_l2_crelu_kernel") != 0) {
        return -1;
    }
    nnue_dense_crelu_kernel<<<blocks, threads, 0, ctx->stream>>>(hidden1, l2w, l2b, hidden2, batch, l2, l3);
    if (check_kernel_launch("nnue_dense_l2_crelu_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch, threads, &blocks, "nnue_dense_output_kernel") != 0) {
        return -1;
    }
    nnue_dense_output_kernel<<<blocks, threads, 0, ctx->stream>>>(hidden2, outw, outb, output, batch, l3);
    if (check_kernel_launch("nnue_dense_output_kernel launch") != 0) {
        return -1;
    }

    return 0;
}

int launch_sfnn_forward_kernels(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    size_t batch,
    size_t max_active,
    const int* stm_indices,
    const int* nstm_indices,
    const int* buckets,
    const float* l0w,
    float* folded_l0w,
    const float* l0b,
    const float* l1w,
    const float* l1b,
    const float* l1fw,
    const float* l1fb,
    int has_l1f,
    const float* l1axw,
    const float* l1axb,
    int has_l1ax,
    const float* l2w,
    const float* l2b,
    const float* l2fw,
    const float* l2fb,
    int has_l2f,
    const float* l2axw,
    const float* l2axb,
    int has_l2ax,
    const float* l3w,
    const float* l3b,
    const float* l3fw,
    const float* l3fb,
    int has_l3f,
    const float* l3axw,
    const float* l3axb,
    int has_l3ax,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const float* factorizer_axis_confidences,
    float* stm_l0,
    float* nstm_l0,
    float* combined,
    float* l1,
    float* l2_input,
    float* l2,
    float* output) {
    if (validate_sfnn_shape(
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            batch,
            max_active) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }

    constexpr int threads = 256;
    int blocks = 0;
    const size_t pairwise = ft_size / 2;
    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    const size_t l2_in = l1_hidden * 2;
    const bool common_shard_l1 = sfnn_is_common_shard_l1_shape(l1_common_size, l1_shard_size);
    const bool grouped_l1 = !common_shard_l1 && sfnn_is_grouped_l1_shape(l1_group_count);
    if ((grouped_l1 || common_shard_l1) && (has_l1f != 0 || has_l1ax != 0)) {
        return fail_message("SFNN compact L1 does not support factorized L1");
    }

    const float* effective_l0w = l0w;
    size_t effective_input_size = input_size;
    if (folded_l0w != nullptr && input_size == SFNN_HALFKA2_FACTORIZED_INPUT_SIZE) {
        const size_t fold_len = SFNN_HALFKA2_BASE_INPUT_SIZE * ft_size;
        if (block_count_1d(fold_len, threads, &blocks, "sfnn_fold_halfka2_l0w_kernel") != 0) {
            return -1;
        }
        sfnn_fold_halfka2_l0w_kernel<<<blocks, threads, 0, ctx->stream>>>(l0w, folded_l0w, ft_size);
        if (check_kernel_launch("sfnn_fold_halfka2_l0w_kernel launch") != 0) {
            return -1;
        }
        effective_l0w = folded_l0w;
        effective_input_size = SFNN_HALFKA2_BASE_INPUT_SIZE;
    }

    if (block_count_1d(batch * pairwise, threads, &blocks, "sfnn_sparse_l0_pairwise_concat_kernel") != 0) {
        return -1;
    }
    sfnn_sparse_l0_pairwise_concat_kernel<<<blocks, threads, 0, ctx->stream>>>(
        stm_indices,
        nstm_indices,
        effective_l0w,
        l0b,
        stm_l0,
        nstm_l0,
        combined,
        batch,
        max_active,
        effective_input_size,
        ft_size);
    if (check_kernel_launch("sfnn_sparse_l0_pairwise_concat_kernel launch") != 0) {
        return -1;
    }

    if (common_shard_l1) {
        const size_t group_output = l1_out / l1_group_count;
        if (block_count_1d(batch * l1_out, threads, &blocks, "sfnn_common_shard_l1_kernel") != 0) {
            return -1;
        }
        sfnn_common_shard_l1_kernel<<<blocks, threads, 0, ctx->stream>>>(
            combined,
            l1w,
            l1b,
            buckets,
            l1,
            batch,
            ft_size,
            l1_out,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            group_output);
        if (check_kernel_launch("sfnn_common_shard_l1_kernel launch") != 0) {
            return -1;
        }
    } else if (grouped_l1) {
        const size_t group_input = ft_size / l1_group_count;
        const size_t group_output = l1_out / l1_group_count;
        if (block_count_1d(batch * l1_out, threads, &blocks, "sfnn_grouped_l1_kernel") != 0) {
            return -1;
        }
        sfnn_grouped_l1_kernel<<<blocks, threads, 0, ctx->stream>>>(
            combined,
            l1w,
            l1b,
            buckets,
            l1,
            batch,
            ft_size,
            l1_out,
            num_stacks,
            l1_group_count,
            group_input,
            group_output);
        if (check_kernel_launch("sfnn_grouped_l1_kernel launch") != 0) {
            return -1;
        }
    } else {
        constexpr int l1_warp_threads = 256;
        constexpr int l1_warps_per_block = l1_warp_threads / 32;
        size_t l1_warps = batch * l1_out;
        if (l1_warps > static_cast<size_t>(INT_MAX) * l1_warps_per_block) {
            return fail_message("sfnn_stacked_l1_warp_kernel grid overflow");
        }
        blocks = static_cast<int>((l1_warps + l1_warps_per_block - 1) / l1_warps_per_block);
        sfnn_stacked_l1_warp_kernel<<<blocks, l1_warp_threads, 0, ctx->stream>>>(
            combined,
            l1w,
            l1b,
            l1fw,
            l1fb,
            l1axw,
            l1axb,
            buckets,
            l1,
            batch,
            ft_size,
            l1_out,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            has_l1f,
            has_l1ax,
            use_king_axis,
            use_hand_axis,
            use_progress_axis,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            factorizer_shared_alpha,
            factorizer_king_axis_alpha,
            factorizer_hand_axis_alpha,
            factorizer_progress_axis_alpha,
            factorizer_pair_alpha,
            factorizer_axis_confidences);
        if (check_kernel_launch("sfnn_stacked_l1_warp_kernel launch") != 0) {
            return -1;
        }
    }

    if (block_count_1d(batch * l2_in, threads, &blocks, "sfnn_l2_input_kernel") != 0) {
        return -1;
    }
    sfnn_l2_input_kernel<<<blocks, threads, 0, ctx->stream>>>(l1, l2_input, batch, l1_hidden, l1_skip);
    if (check_kernel_launch("sfnn_l2_input_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l2_size, threads, &blocks, "sfnn_stacked_l2_crelu_kernel") != 0) {
        return -1;
    }
    sfnn_stacked_l2_crelu_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l2_input,
        l2w,
        l2b,
        l2fw,
        l2fb,
        l2axw,
        l2axb,
        buckets,
        l2,
        batch,
        l2_in,
        l2_size,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l2f,
        has_l2ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
            factorizer_shared_alpha,
            factorizer_king_axis_alpha,
            factorizer_hand_axis_alpha,
            factorizer_progress_axis_alpha,
            factorizer_pair_alpha,
            factorizer_axis_confidences);
    if (check_kernel_launch("sfnn_stacked_l2_crelu_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch, threads, &blocks, "sfnn_stacked_l3_output_kernel") != 0) {
        return -1;
    }
    sfnn_stacked_l3_output_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l2,
        l1,
        l3w,
        l3b,
        l3fw,
        l3fb,
        l3axw,
        l3axb,
        buckets,
        output,
        batch,
        l2_size,
        l1_hidden,
        l1_skip,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l3f,
        has_l3ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences);
    if (check_kernel_launch("sfnn_stacked_l3_output_kernel launch") != 0) {
        return -1;
    }

    return 0;
}

int launch_fill_f32_raw(BulletOuCudaCppContext* ctx, float* ptr, size_t len, float value, const char* label) {
    if (len == 0) {
        return 0;
    }
    constexpr int threads = 256;
    int blocks = 0;
    if (block_count_1d(len, threads, &blocks, label) != 0) {
        return -1;
    }
    fill_f32_kernel<<<blocks, threads, 0, ctx->stream>>>(len, value, ptr);
    if (check_kernel_launch(label) != 0) {
        return -1;
    }
    return 0;
}

int launch_zero_sfnn_backward_parameter_gradients(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float* l0w_gradients,
    float* l0b_gradients,
    float* l1w_gradients,
    float* l1b_gradients,
    float* l1fw_gradients,
    float* l1fb_gradients,
    float* l1axw_gradients,
    float* l1axb_gradients,
    float* l2w_gradients,
    float* l2b_gradients,
    float* l2fw_gradients,
    float* l2fb_gradients,
    float* l2axw_gradients,
    float* l2axb_gradients,
    float* l3w_gradients,
    float* l3b_gradients,
    float* l3fw_gradients,
    float* l3fb_gradients,
    float* l3axw_gradients,
    float* l3axb_gradients,
    int zero_l0w_gradients) {
    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    const size_t l2_in = l1_hidden * 2;
    const size_t axis_count = sfnn_factorizer_axis_count(
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair);
    const size_t l1w_len =
        sfnn_l1w_len_for_shape(ft_size, l1_hidden, l1_skip, num_stacks, l1_group_count, l1_common_size, l1_shard_size);
    if (zero_l0w_gradients != 0 &&
        launch_fill_f32_raw(ctx, l0w_gradients, input_size * ft_size, 0.0f, "sfnn zero l0w_gradients") != 0) {
        return -1;
    }
    if (launch_fill_f32_raw(ctx, l0b_gradients, ft_size, 0.0f, "sfnn zero l0b_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1w_gradients, l1w_len, 0.0f, "sfnn zero l1w_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1b_gradients, num_stacks * l1_out, 0.0f, "sfnn zero l1b_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1fw_gradients, ft_size * l1_out, 0.0f, "sfnn zero l1fw_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1fb_gradients, l1_out, 0.0f, "sfnn zero l1fb_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1axw_gradients, axis_count * ft_size * l1_out, 0.0f, "sfnn zero l1axw_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1axb_gradients, axis_count * l1_out, 0.0f, "sfnn zero l1axb_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l2w_gradients, num_stacks * l2_size * l2_in, 0.0f, "sfnn zero l2w_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l2b_gradients, num_stacks * l2_size, 0.0f, "sfnn zero l2b_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l2fw_gradients, l2_size * l2_in, 0.0f, "sfnn zero l2fw_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l2fb_gradients, l2_size, 0.0f, "sfnn zero l2fb_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l2axw_gradients, axis_count * l2_size * l2_in, 0.0f, "sfnn zero l2axw_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l2axb_gradients, axis_count * l2_size, 0.0f, "sfnn zero l2axb_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l3w_gradients, num_stacks * l2_size, 0.0f, "sfnn zero l3w_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l3b_gradients, num_stacks, 0.0f, "sfnn zero l3b_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l3fw_gradients, l2_size, 0.0f, "sfnn zero l3fw_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l3fb_gradients, 1, 0.0f, "sfnn zero l3fb_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l3axw_gradients, axis_count * l2_size, 0.0f, "sfnn zero l3axw_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l3axb_gradients, axis_count, 0.0f, "sfnn zero l3axb_gradients") != 0) {
        return -1;
    }
    return 0;
}

unsigned int sfnn_dense_reduce_split_count(size_t batch) {
    if (batch <= 2048) {
        return 1;
    }
    size_t splits = (batch + 2047) / 2048;
    splits = std::max<size_t>(1, std::min<size_t>(32, splits));
    return static_cast<unsigned int>(splits);
}

int launch_sfnn_dense_param_reduce_tiled(
    BulletOuCudaCppContext* ctx,
    const float* inputs,
    const float* activations,
    const float* output_gradients,
    const int* buckets,
    float* weight_gradients,
    float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    float* axis_weight_gradients,
    float* axis_bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int shared_weight_input_major,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    int use_crelu_gradient,
    const char* label) {
    if (batch == 0 || input_dim == 0 || output_dim == 0 || num_stacks == 0) {
        return 0;
    }
    if (num_stacks <= SFNN_DENSE_REDUCE_ACCUM_MAX_STACKS) {
        const size_t input_tiles = (input_dim + SFNN_DENSE_REDUCE_TILE - 1) / SFNN_DENSE_REDUCE_TILE;
        const size_t output_tiles = (output_dim + SFNN_DENSE_REDUCE_TILE - 1) / SFNN_DENSE_REDUCE_TILE;
        if (input_tiles == 0 || output_tiles == 0 || input_tiles > UINT_MAX / output_tiles) {
            return fail_message("sfnn dense reduce accum tile grid overflow");
        }
        const size_t tile_count = input_tiles * output_tiles;
        if (tile_count > static_cast<size_t>(INT_MAX)) {
            return fail_message("sfnn dense reduce accum x grid too large");
        }
        dim3 grid(
            static_cast<unsigned int>(tile_count),
            sfnn_dense_reduce_split_count(batch),
            1U);
        sfnn_dense_param_reduce_accum_stacks_kernel<<<grid, SFNN_DENSE_REDUCE_THREADS, 0, ctx->stream>>>(
            inputs,
            activations,
            output_gradients,
            buckets,
            weight_gradients,
            bias_gradients,
            shared_weight_gradients,
            shared_bias_gradients,
            axis_weight_gradients,
            axis_bias_gradients,
            batch,
            input_dim,
            output_dim,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            has_shared,
            shared_weight_input_major,
            has_axis,
            use_king_axis,
            use_hand_axis,
            use_progress_axis,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            shared_alpha,
            king_axis_alpha,
            hand_axis_alpha,
            progress_axis_alpha,
            pair_alpha,
            factorizer_axis_confidences,
            use_crelu_gradient);
        if (check_kernel_launch(label) != 0) {
            return -1;
        }
        return 0;
    }
    if (has_axis != 0 || use_king_axis != 0 || use_hand_axis != 0 || use_progress_axis != 0) {
        return fail_message("sfnn dense reduce axis fast path was requested for too many stacks");
    }
    if (num_stacks > SFNN_DENSE_REDUCE_MAX_FAST_STACKS) {
        return fail_message("sfnn dense reduce fast path was requested for too many stacks");
    }
    const size_t input_tiles = (input_dim + SFNN_DENSE_REDUCE_TILE - 1) / SFNN_DENSE_REDUCE_TILE;
    const size_t output_tiles = (output_dim + SFNN_DENSE_REDUCE_TILE - 1) / SFNN_DENSE_REDUCE_TILE;
    if (input_tiles == 0 || output_tiles == 0 || input_tiles > UINT_MAX / output_tiles) {
        return fail_message("sfnn dense reduce tile grid overflow");
    }
    const size_t tile_count = input_tiles * output_tiles;
    if (tile_count > static_cast<size_t>(INT_MAX)) {
        return fail_message("sfnn dense reduce x grid too large");
    }
    if (num_stacks > 65535) {
        return fail_message("sfnn dense reduce z grid too large");
    }
    dim3 grid(
        static_cast<unsigned int>(tile_count),
        sfnn_dense_reduce_split_count(batch),
        static_cast<unsigned int>(num_stacks));
    sfnn_dense_param_reduce_tiled_kernel<<<grid, SFNN_DENSE_REDUCE_THREADS, 0, ctx->stream>>>(
        inputs,
        activations,
        output_gradients,
        buckets,
        weight_gradients,
        bias_gradients,
        shared_weight_gradients,
        shared_bias_gradients,
        batch,
        input_dim,
        output_dim,
        num_stacks,
        has_shared,
        shared_weight_input_major,
        shared_alpha,
        use_crelu_gradient);
    if (check_kernel_launch(label) != 0) {
        return -1;
    }
    return 0;
}

int launch_sfnn_fold_stacked_param_gradients_to_factorizers(
    BulletOuCudaCppContext* ctx,
    const float* weight_gradients,
    const float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    float* axis_weight_gradients,
    float* axis_bias_gradients,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int shared_weight_input_major,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    const char* label) {
    if ((has_shared == 0 && has_axis == 0) || input_dim == 0 || output_dim == 0 || num_stacks == 0) {
        return 0;
    }
    if (input_dim > SIZE_MAX / output_dim) {
        return fail_message("sfnn factorizer gradient fold weight cell overflow");
    }
    const size_t weight_cell_count = input_dim * output_dim;
    if (num_stacks > SIZE_MAX / weight_cell_count || num_stacks > SIZE_MAX / output_dim) {
        return fail_message("sfnn factorizer gradient fold total size overflow");
    }
    const size_t weight_total = num_stacks * weight_cell_count;
    const size_t bias_total = num_stacks * output_dim;
    const size_t total = std::max(weight_total, bias_total);
    int blocks = 0;
    constexpr int threads = 256;
    if (block_count_1d(total, threads, &blocks, label) != 0) {
        return -1;
    }
    sfnn_fold_stacked_param_gradients_to_factorizers_kernel<<<blocks, threads, 0, ctx->stream>>>(
        weight_gradients,
        bias_gradients,
        shared_weight_gradients,
        shared_bias_gradients,
        axis_weight_gradients,
        axis_bias_gradients,
        input_dim,
        output_dim,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_shared,
        shared_weight_input_major,
        has_axis,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        shared_alpha,
        king_axis_alpha,
        hand_axis_alpha,
        progress_axis_alpha,
        pair_alpha,
        factorizer_axis_confidences);
    if (check_kernel_launch(label) != 0) {
        return -1;
    }
    return 0;
}

int launch_sfnn_add_saturation_penalty_gradients(
    BulletOuCudaCppContext* ctx,
    const float* weights,
    const float* shared_weights,
    const float* axis_weights,
    float* weight_gradients,
    float* shared_weight_gradients,
    float* axis_weight_gradients,
    const int* dirty_buckets,
    size_t dirty_count,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int has_shared,
    int shared_weight_input_major,
    int has_axis,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float shared_alpha,
    float king_axis_alpha,
    float hand_axis_alpha,
    float progress_axis_alpha,
    float pair_alpha,
    const float* factorizer_axis_confidences,
    float weight_scale,
    float threshold,
    float penalty,
    const char* label) {
    if (penalty == 0.0f) {
        return 0;
    }
    if (input_dim == 0 || output_dim == 0 || num_stacks == 0) {
        return fail_message("sfnn saturation penalty dimensions must be non-zero");
    }
    if (!(std::isfinite(weight_scale) && weight_scale > 0.0f)) {
        return fail_message("sfnn saturation penalty weight_scale must be finite and positive");
    }
    if (!(std::isfinite(threshold) && threshold > 0.0f && threshold <= 128.0f)) {
        return fail_message("sfnn saturation penalty threshold must be finite and in (0, 128]");
    }
    if (!(std::isfinite(penalty) && penalty >= 0.0f)) {
        return fail_message("sfnn saturation penalty must be finite and non-negative");
    }
    if (input_dim > SIZE_MAX / output_dim) {
        return fail_message("sfnn saturation penalty weight cell overflow");
    }
    const size_t weight_cell_count = input_dim * output_dim;
    const size_t stack_count = dirty_count == 0 ? num_stacks : dirty_count;
    if (stack_count > SIZE_MAX / weight_cell_count) {
        return fail_message("sfnn saturation penalty total size overflow");
    }
    const size_t total = stack_count * weight_cell_count;
    int blocks = 0;
    constexpr int threads = 256;
    if (block_count_1d(total, threads, &blocks, label) != 0) {
        return -1;
    }
    sfnn_add_saturation_penalty_gradients_kernel<<<blocks, threads, 0, ctx->stream>>>(
        weights,
        shared_weights,
        axis_weights,
        weight_gradients,
        shared_weight_gradients,
        axis_weight_gradients,
        dirty_buckets,
        dirty_count,
        input_dim,
        output_dim,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_shared,
        shared_weight_input_major,
        has_axis,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        shared_alpha,
        king_axis_alpha,
        hand_axis_alpha,
        progress_axis_alpha,
        pair_alpha,
        factorizer_axis_confidences,
        weight_scale,
        threshold,
        penalty);
    if (check_kernel_launch(label) != 0) {
        return -1;
    }
    return 0;
}

int launch_sfnn_add_residual_count_decay_gradients(
    BulletOuCudaCppContext* ctx,
    const float* weights,
    float* gradients,
    const float* decay_by_stack,
    const int* dirty_buckets,
    size_t dirty_count,
    size_t stride,
    size_t num_stacks,
    const char* label) {
    if (stride == 0 || num_stacks == 0) {
        return fail_message("sfnn residual count decay dimensions must be non-zero");
    }
    const size_t stack_count = dirty_count == 0 ? num_stacks : dirty_count;
    if (stack_count > SIZE_MAX / stride) {
        return fail_message("sfnn residual count decay total size overflow");
    }
    const size_t total = stack_count * stride;
    int blocks = 0;
    constexpr int threads = 256;
    if (block_count_1d(total, threads, &blocks, label) != 0) {
        return -1;
    }
    sfnn_add_residual_count_decay_gradients_kernel<<<blocks, threads, 0, ctx->stream>>>(
        weights,
        gradients,
        decay_by_stack,
        dirty_buckets,
        dirty_count,
        stride,
        num_stacks);
    if (check_kernel_launch(label) != 0) {
        return -1;
    }
    return 0;
}

int memset_i32_async(BulletOuCudaCppContext* ctx, int* ptr, size_t len, int value, const char* label) {
    if (len == 0) {
        return 0;
    }
    if (value != 0) {
        return fail_message("memset_i32_async currently supports only zero fills");
    }
    cudaError_t status = cudaMemsetAsync(ptr, 0, len * sizeof(int), ctx->stream);
    if (status != cudaSuccess) {
        return fail(label, status);
    }
    return 0;
}

int ensure_sfnn_inverse_index_scratch(
    BulletOuCudaCppContext* ctx,
    size_t n_features,
    size_t total_entries,
    size_t prefix_blocks) {
    if (prefix_blocks == 0 || prefix_blocks > 1024) {
        return fail_message("SFNN inverse-index prefix block count must be in 1..=1024");
    }
    if (ensure_i32_scratch(
            &ctx->sfnn_inverse_counts,
            &ctx->sfnn_inverse_counts_len,
            n_features,
            "cudaMalloc sfnn inverse counts") != 0 ||
        ensure_i32_scratch(
            &ctx->sfnn_inverse_offsets,
            &ctx->sfnn_inverse_offsets_len,
            n_features + 1,
            "cudaMalloc sfnn inverse offsets") != 0 ||
        ensure_i32_scratch(
            &ctx->sfnn_inverse_block_sums,
            &ctx->sfnn_inverse_block_sums_len,
            prefix_blocks,
            "cudaMalloc sfnn inverse block sums") != 0 ||
        ensure_i32_scratch(
            &ctx->sfnn_inverse_block_offsets,
            &ctx->sfnn_inverse_block_offsets_len,
            prefix_blocks + 1,
            "cudaMalloc sfnn inverse block offsets") != 0 ||
        ensure_i32_scratch(
            &ctx->sfnn_inverse_write_counters,
            &ctx->sfnn_inverse_write_counters_len,
            n_features,
            "cudaMalloc sfnn inverse write counters") != 0 ||
        ensure_i32_scratch(
            &ctx->sfnn_inverse_positions,
            &ctx->sfnn_inverse_positions_len,
            total_entries,
            "cudaMalloc sfnn inverse positions") != 0) {
        return -1;
    }
    return 0;
}

int launch_sfnn_inverse_index_for_perspective(
    BulletOuCudaCppContext* ctx,
    const int* indices,
    const float* pre_gradients,
    float* l0w_gradients,
    size_t batch,
    size_t max_active,
    size_t n_features,
    size_t ft_size,
    int add_to_existing) {
    constexpr int count_threads = 256;
    constexpr int scan_threads = 1024;
    constexpr int gather_threads = 128;
    const size_t total_entries = batch * max_active;
    const size_t prefix_blocks = (n_features + scan_threads - 1) / scan_threads;
    if (ensure_sfnn_inverse_index_scratch(ctx, n_features, total_entries, prefix_blocks) != 0) {
        return -1;
    }

    if (memset_i32_async(ctx, ctx->sfnn_inverse_counts, n_features, 0, "cudaMemsetAsync sfnn inverse counts") != 0 ||
        memset_i32_async(
            ctx,
            ctx->sfnn_inverse_write_counters,
            n_features,
            0,
            "cudaMemsetAsync sfnn inverse write counters") != 0) {
        return -1;
    }

    int blocks = 0;
    if (block_count_1d(total_entries, count_threads, &blocks, "sfnn_inverse_feature_counts_kernel") != 0) {
        return -1;
    }
    sfnn_inverse_feature_counts_kernel<<<blocks, count_threads, 0, ctx->stream>>>(
        indices,
        ctx->sfnn_inverse_counts,
        total_entries,
        max_active,
        n_features);
    if (check_kernel_launch("sfnn_inverse_feature_counts_kernel launch") != 0) {
        return -1;
    }

    sfnn_inverse_prefix_sum_block_local_kernel<<<static_cast<int>(prefix_blocks), scan_threads, 0, ctx->stream>>>(
        ctx->sfnn_inverse_counts,
        ctx->sfnn_inverse_offsets,
        ctx->sfnn_inverse_block_sums,
        n_features);
    if (check_kernel_launch("sfnn_inverse_prefix_sum_block_local_kernel launch") != 0) {
        return -1;
    }
    sfnn_inverse_prefix_sum_small_kernel<<<1, scan_threads, 0, ctx->stream>>>(
        ctx->sfnn_inverse_block_sums,
        ctx->sfnn_inverse_block_offsets,
        prefix_blocks);
    if (check_kernel_launch("sfnn_inverse_prefix_sum_small_kernel launch") != 0) {
        return -1;
    }
    sfnn_inverse_prefix_add_block_offsets_kernel<<<static_cast<int>(prefix_blocks), scan_threads, 0, ctx->stream>>>(
        ctx->sfnn_inverse_offsets,
        ctx->sfnn_inverse_block_offsets,
        n_features,
        prefix_blocks);
    if (check_kernel_launch("sfnn_inverse_prefix_add_block_offsets_kernel launch") != 0) {
        return -1;
    }

    sfnn_inverse_scatter_positions_kernel<<<blocks, count_threads, 0, ctx->stream>>>(
        indices,
        ctx->sfnn_inverse_offsets,
        ctx->sfnn_inverse_write_counters,
        ctx->sfnn_inverse_positions,
        total_entries,
        max_active,
        n_features);
    if (check_kernel_launch("sfnn_inverse_scatter_positions_kernel launch") != 0) {
        return -1;
    }

    dim3 gather_grid(
        static_cast<unsigned int>(n_features),
        static_cast<unsigned int>((ft_size + gather_threads - 1) / gather_threads),
        1);
    sfnn_inverse_gather_l0w_gradients_kernel<<<gather_grid, gather_threads, 0, ctx->stream>>>(
        pre_gradients,
        ctx->sfnn_inverse_positions,
        ctx->sfnn_inverse_offsets,
        l0w_gradients,
        n_features,
        ft_size,
        add_to_existing);
    if (check_kernel_launch("sfnn_inverse_gather_l0w_gradients_kernel launch") != 0) {
        return -1;
    }
    return 0;
}

int launch_sfnn_inverse_index_l0_backward(
    BulletOuCudaCppContext* ctx,
    const int* stm_indices,
    const int* nstm_indices,
    const float* stm_l0,
    const float* nstm_l0,
    const float* combined_gradients,
    float* stm_l0_pre_gradients,
    float* nstm_l0_pre_gradients,
    float* l0w_gradients,
    float* l0b_gradients,
    size_t batch,
    size_t max_active,
    size_t input_size,
    size_t ft_size) {
    constexpr int threads = 256;
    int blocks = 0;
    if (block_count_1d(batch * (ft_size / 2), threads, &blocks, "sfnn_pairwise_l0_pregrad_kernel") != 0) {
        return -1;
    }
    sfnn_pairwise_l0_pregrad_kernel<<<blocks, threads, 0, ctx->stream>>>(
        stm_l0,
        nstm_l0,
        combined_gradients,
        stm_l0_pre_gradients,
        nstm_l0_pre_gradients,
        l0b_gradients,
        batch,
        ft_size);
    if (check_kernel_launch("sfnn_pairwise_l0_pregrad_kernel launch") != 0) {
        return -1;
    }

    size_t n_features = input_size;
    const bool halfka2_factorized = input_size == SFNN_HALFKA2_FACTORIZED_INPUT_SIZE;
    if (halfka2_factorized) {
        n_features = SFNN_HALFKA2_BASE_INPUT_SIZE;
    }

    if (launch_sfnn_inverse_index_for_perspective(
            ctx,
            stm_indices,
            stm_l0_pre_gradients,
            l0w_gradients,
            batch,
            max_active,
            n_features,
            ft_size,
            0) != 0 ||
        launch_sfnn_inverse_index_for_perspective(
            ctx,
            nstm_indices,
            nstm_l0_pre_gradients,
            l0w_gradients,
            batch,
            max_active,
            n_features,
            ft_size,
            1) != 0) {
        return -1;
    }

    if (halfka2_factorized) {
        constexpr int gather_threads = 128;
        dim3 reduce_grid(
            static_cast<unsigned int>(SFNN_HALFKA2_PIECE_INPUTS),
            static_cast<unsigned int>((ft_size + gather_threads - 1) / gather_threads),
            1);
        sfnn_reduce_halfka2_virtual_l0w_gradients_kernel<<<reduce_grid, gather_threads, 0, ctx->stream>>>(
            l0w_gradients,
            ft_size);
        if (check_kernel_launch("sfnn_reduce_halfka2_virtual_l0w_gradients_kernel launch") != 0) {
            return -1;
        }
    }

    return 0;
}

int launch_sfnn_backward_kernels(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    size_t batch,
    size_t max_active,
    const int* stm_indices,
    const int* nstm_indices,
    const int* buckets,
    const float* stm_l0,
    const float* nstm_l0,
    const float* combined,
    const float* l1,
    const float* l2_input,
    const float* l2,
    const float* l1w,
    const float* l1fw,
    int has_l1f,
    const float* l1axw,
    int has_l1ax,
    const float* l2w,
    const float* l2fw,
    int has_l2f,
    const float* l2axw,
    int has_l2ax,
    const float* l3w,
    const float* l3fw,
    int has_l3f,
    const float* l3axw,
    int has_l3ax,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const float* factorizer_axis_confidences,
    const float* mean_output_gradients,
    float* l2_gradients,
    float* l1_gradients,
    float* l2_input_gradients,
    float* combined_gradients,
    float* stm_l0_gradients,
    float* nstm_l0_gradients,
    float* stm_l0_pre_gradients,
    float* nstm_l0_pre_gradients,
    float* l0w_gradients,
    float* l0b_gradients,
    float* l1w_gradients,
    float* l1b_gradients,
    float* l1fw_gradients,
    float* l1fb_gradients,
    float* l1axw_gradients,
    float* l1axb_gradients,
    float* l2w_gradients,
    float* l2b_gradients,
    float* l2fw_gradients,
    float* l2fb_gradients,
    float* l2axw_gradients,
    float* l2axb_gradients,
    float* l3w_gradients,
    float* l3b_gradients,
    float* l3fw_gradients,
    float* l3fb_gradients,
    float* l3axw_gradients,
    float* l3axb_gradients,
    int zero_parameter_gradients,
    int fuse_pairwise_l0,
    float* profile_ms,
    size_t profile_ms_len) {
    if (validate_sfnn_shape(
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            batch,
            max_active) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }

    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    const size_t l2_in = l1_hidden * 2;
    const bool common_shard_l1 = sfnn_is_common_shard_l1_shape(l1_common_size, l1_shard_size);
    const bool grouped_l1 = !common_shard_l1 && sfnn_is_grouped_l1_shape(l1_group_count);
    const bool dense_param_reduce_accum_path = num_stacks <= SFNN_DENSE_REDUCE_ACCUM_MAX_STACKS;
    const bool dense_param_reduce_fast_path =
        dense_param_reduce_accum_path ||
        (num_stacks <= SFNN_DENSE_REDUCE_MAX_FAST_STACKS &&
         has_l1ax == 0 && has_l2ax == 0 && has_l3ax == 0 &&
         use_king_axis == 0 && use_hand_axis == 0);
    if ((grouped_l1 || common_shard_l1) && (has_l1f != 0 || has_l1ax != 0)) {
        return fail_message("SFNN compact L1 does not support factorized L1");
    }
    constexpr int threads = 256;
    int blocks = 0;
    SfnnBackwardProfileEvents profile;
    if (profile.init(ctx, profile_ms, profile_ms_len) != 0) {
        return -1;
    }

    if (zero_parameter_gradients != 0) {
        if (launch_zero_sfnn_backward_parameter_gradients(
                ctx,
                input_size,
                ft_size,
                l1_hidden,
                l1_skip,
                l2_size,
                num_stacks,
                l1_group_count,
                l1_common_size,
                l1_shard_size,
                factorizer_king_axis_dim,
                factorizer_hand_axis_dim,
                use_progress_axis,
                use_king_hand_pair,
                use_king_progress_pair,
                use_hand_progress_pair,
                l0w_gradients,
                l0b_gradients,
                l1w_gradients,
                l1b_gradients,
                l1fw_gradients,
                l1fb_gradients,
                l1axw_gradients,
                l1axb_gradients,
                l2w_gradients,
                l2b_gradients,
                l2fw_gradients,
                l2fb_gradients,
                l2axw_gradients,
                l2axb_gradients,
                l3w_gradients,
                l3b_gradients,
                l3fw_gradients,
                l3fb_gradients,
                l3axw_gradients,
                l3axb_gradients,
                fuse_pairwise_l0 == 0 ? 1 : 0) != 0) {
            return -1;
        }
    }
    if (profile.record(1, ctx, "SFNN backward profile after zero") != 0) {
        return -1;
    }

    size_t l3_threads = std::max(batch * l2_size, batch * l1_out);
    l3_threads = std::max(l3_threads, batch);
    if (block_count_1d(l3_threads, threads, &blocks, "sfnn_stacked_l3_backward_kernel") != 0) {
        return -1;
    }
    sfnn_stacked_l3_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l2,
        mean_output_gradients,
        l3w,
        l3fw,
        l3axw,
        buckets,
        l2_gradients,
        l1_gradients,
        l3w_gradients,
        l3b_gradients,
        l3fw_gradients,
        l3fb_gradients,
        l3axw_gradients,
        l3axb_gradients,
        batch,
        l2_size,
        l1_out,
        l1_skip,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l3f,
        has_l3ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences,
        dense_param_reduce_fast_path ? 0 : 1,
        0);
    if (check_kernel_launch("sfnn_stacked_l3_backward_kernel launch") != 0) {
        return -1;
    }
    if (!dense_param_reduce_fast_path) {
        if (launch_sfnn_fold_stacked_param_gradients_to_factorizers(
                ctx,
                l3w_gradients,
                l3b_gradients,
                has_l3f != 0 ? l3fw_gradients : nullptr,
                has_l3f != 0 ? l3fb_gradients : nullptr,
                has_l3ax != 0 ? l3axw_gradients : nullptr,
                has_l3ax != 0 ? l3axb_gradients : nullptr,
                l2_size,
                1,
                num_stacks,
                factorizer_king_axis_dim,
                factorizer_hand_axis_dim,
                has_l3f,
                0,
                has_l3ax,
                use_king_axis,
                use_hand_axis,
                use_progress_axis,
                use_king_hand_pair,
                use_king_progress_pair,
                use_hand_progress_pair,
                factorizer_shared_alpha,
                factorizer_king_axis_alpha,
                factorizer_hand_axis_alpha,
                factorizer_progress_axis_alpha,
                factorizer_pair_alpha,
                factorizer_axis_confidences,
                "sfnn fold L3 factorizer gradients") != 0) {
            return -1;
        }
    }
    if (dense_param_reduce_fast_path) {
        if (launch_sfnn_dense_param_reduce_tiled(
                ctx,
                l2,
                nullptr,
                mean_output_gradients,
                buckets,
                l3w_gradients,
                l3b_gradients,
                has_l3f != 0 ? l3fw_gradients : nullptr,
                has_l3f != 0 ? l3fb_gradients : nullptr,
                has_l3ax != 0 ? l3axw_gradients : nullptr,
                has_l3ax != 0 ? l3axb_gradients : nullptr,
                batch,
                l2_size,
                1,
                num_stacks,
                factorizer_king_axis_dim,
                factorizer_hand_axis_dim,
                has_l3f,
                0,
                has_l3ax,
                use_king_axis,
                use_hand_axis,
                use_progress_axis,
                use_king_hand_pair,
                use_king_progress_pair,
                use_hand_progress_pair,
                factorizer_shared_alpha,
                factorizer_king_axis_alpha,
                factorizer_hand_axis_alpha,
                factorizer_progress_axis_alpha,
                factorizer_pair_alpha,
                factorizer_axis_confidences,
                0,
                "sfnn dense reduce L3 params") != 0) {
            return -1;
        }
    }
    if (profile.record(2, ctx, "SFNN backward profile after L3") != 0) {
        return -1;
    }

    size_t l2_threads = dense_param_reduce_fast_path
        ? batch * l2_in
        : std::max(batch * l2_in, batch * l2_in * l2_size);
    if (!dense_param_reduce_fast_path) {
        l2_threads = std::max(l2_threads, batch * l2_size);
    }
    if (block_count_1d(l2_threads, threads, &blocks, "sfnn_stacked_crelu_backward_kernel") != 0) {
        return -1;
    }
    sfnn_stacked_crelu_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l2_input,
        l2,
        l2_gradients,
        l2w,
        l2fw,
        l2axw,
        buckets,
        l2_input_gradients,
        l2w_gradients,
        l2b_gradients,
        l2fw_gradients,
        l2fb_gradients,
        l2axw_gradients,
        l2axb_gradients,
        batch,
        l2_in,
        l2_size,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l2f,
        has_l2ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences,
        dense_param_reduce_fast_path ? 0 : 1,
        0);
    if (check_kernel_launch("sfnn_stacked_crelu_backward_kernel launch") != 0) {
        return -1;
    }
    if (!dense_param_reduce_fast_path) {
        if (launch_sfnn_fold_stacked_param_gradients_to_factorizers(
                ctx,
                l2w_gradients,
                l2b_gradients,
                has_l2f != 0 ? l2fw_gradients : nullptr,
                has_l2f != 0 ? l2fb_gradients : nullptr,
                has_l2ax != 0 ? l2axw_gradients : nullptr,
                has_l2ax != 0 ? l2axb_gradients : nullptr,
                l2_in,
                l2_size,
                num_stacks,
                factorizer_king_axis_dim,
                factorizer_hand_axis_dim,
                has_l2f,
                0,
                has_l2ax,
                use_king_axis,
                use_hand_axis,
                use_progress_axis,
                use_king_hand_pair,
                use_king_progress_pair,
                use_hand_progress_pair,
                factorizer_shared_alpha,
                factorizer_king_axis_alpha,
                factorizer_hand_axis_alpha,
                factorizer_progress_axis_alpha,
                factorizer_pair_alpha,
                factorizer_axis_confidences,
                "sfnn fold L2 factorizer gradients") != 0) {
            return -1;
        }
    }
    if (dense_param_reduce_fast_path) {
        if (launch_sfnn_dense_param_reduce_tiled(
                ctx,
                l2_input,
                l2,
                l2_gradients,
                buckets,
                l2w_gradients,
                l2b_gradients,
                has_l2f != 0 ? l2fw_gradients : nullptr,
                has_l2f != 0 ? l2fb_gradients : nullptr,
                has_l2ax != 0 ? l2axw_gradients : nullptr,
                has_l2ax != 0 ? l2axb_gradients : nullptr,
                batch,
                l2_in,
                l2_size,
                num_stacks,
                factorizer_king_axis_dim,
                factorizer_hand_axis_dim,
                has_l2f,
                0,
                has_l2ax,
                use_king_axis,
                use_hand_axis,
                use_progress_axis,
                use_king_hand_pair,
                use_king_progress_pair,
                use_hand_progress_pair,
                factorizer_shared_alpha,
                factorizer_king_axis_alpha,
                factorizer_hand_axis_alpha,
                factorizer_progress_axis_alpha,
                factorizer_pair_alpha,
                factorizer_axis_confidences,
                1,
                "sfnn dense reduce L2 params") != 0) {
            return -1;
        }
    }
    if (profile.record(3, ctx, "SFNN backward profile after L2") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l1_out, threads, &blocks, "sfnn_l2_input_backward_kernel") != 0) {
        return -1;
    }
    sfnn_l2_input_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l1, l2_input, l2_input_gradients, l1_gradients, batch, l1_hidden, l1_skip);
    if (check_kernel_launch("sfnn_l2_input_backward_kernel launch") != 0) {
        return -1;
    }
    if (profile.record(4, ctx, "SFNN backward profile after L2 input") != 0) {
        return -1;
    }

    if (common_shard_l1) {
        const size_t group_output = l1_out / l1_group_count;
        const size_t row_input = l1_common_size + l1_shard_size;
        size_t l1_threads = std::max(batch * ft_size, batch * l1_out * row_input);
        l1_threads = std::max(l1_threads, batch * l1_out);
        if (block_count_1d(l1_threads, threads, &blocks, "sfnn_common_shard_l1_backward_kernel") != 0) {
            return -1;
        }
        sfnn_common_shard_l1_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
            combined,
            l1_gradients,
            l1w,
            buckets,
            combined_gradients,
            l1w_gradients,
            l1b_gradients,
            batch,
            ft_size,
            l1_out,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            group_output);
        if (check_kernel_launch("sfnn_common_shard_l1_backward_kernel launch") != 0) {
            return -1;
        }
    } else if (grouped_l1) {
        const size_t group_input = ft_size / l1_group_count;
        const size_t group_output = l1_out / l1_group_count;
        size_t l1_threads = std::max(batch * ft_size, batch * ft_size * group_output);
        l1_threads = std::max(l1_threads, batch * l1_out);
        if (block_count_1d(l1_threads, threads, &blocks, "sfnn_grouped_l1_backward_kernel") != 0) {
            return -1;
        }
        sfnn_grouped_l1_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
            combined,
            l1_gradients,
            l1w,
            buckets,
            combined_gradients,
            l1w_gradients,
            l1b_gradients,
            batch,
            ft_size,
            l1_out,
            num_stacks,
            l1_group_count,
            group_input,
            group_output);
        if (check_kernel_launch("sfnn_grouped_l1_backward_kernel launch") != 0) {
            return -1;
        }
    } else if (has_l1f != 0 || has_l1ax != 0) {
        const bool reduce_l1_params = dense_param_reduce_fast_path;
        size_t l1_threads = reduce_l1_params ? batch * ft_size : std::max(batch * ft_size, batch * ft_size * l1_out);
        if (!reduce_l1_params) {
            l1_threads = std::max(l1_threads, batch * l1_out);
        }
        if (block_count_1d(l1_threads, threads, &blocks, "sfnn_factorized_l1_backward_kernel") != 0) {
            return -1;
        }
        sfnn_factorized_l1_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
            combined,
            l1_gradients,
            l1w,
            l1fw,
            l1axw,
            buckets,
            combined_gradients,
            l1w_gradients,
            l1b_gradients,
            l1fw_gradients,
            l1fb_gradients,
            l1axw_gradients,
            l1axb_gradients,
            batch,
            ft_size,
            l1_out,
            num_stacks,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            has_l1f,
            has_l1ax,
            use_king_axis,
            use_hand_axis,
            use_progress_axis,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            factorizer_shared_alpha,
            factorizer_king_axis_alpha,
            factorizer_hand_axis_alpha,
            factorizer_progress_axis_alpha,
            factorizer_pair_alpha,
            factorizer_axis_confidences,
            reduce_l1_params ? 0 : 1,
            0);
        if (check_kernel_launch("sfnn_factorized_l1_backward_kernel launch") != 0) {
            return -1;
        }
        if (!reduce_l1_params) {
            if (launch_sfnn_fold_stacked_param_gradients_to_factorizers(
                    ctx,
                    l1w_gradients,
                    l1b_gradients,
                    has_l1f != 0 ? l1fw_gradients : nullptr,
                    has_l1f != 0 ? l1fb_gradients : nullptr,
                    has_l1ax != 0 ? l1axw_gradients : nullptr,
                    has_l1ax != 0 ? l1axb_gradients : nullptr,
                    ft_size,
                    l1_out,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    has_l1f,
                    1,
                    has_l1ax,
                    use_king_axis,
                    use_hand_axis,
                    use_progress_axis,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    factorizer_shared_alpha,
                    factorizer_king_axis_alpha,
                    factorizer_hand_axis_alpha,
                    factorizer_progress_axis_alpha,
                    factorizer_pair_alpha,
                    factorizer_axis_confidences,
                    "sfnn fold L1 factorizer gradients") != 0) {
                return -1;
            }
        }
        if (reduce_l1_params) {
            if (launch_sfnn_dense_param_reduce_tiled(
                    ctx,
                    combined,
                    nullptr,
                    l1_gradients,
                    buckets,
                    l1w_gradients,
                    l1b_gradients,
                    has_l1f != 0 ? l1fw_gradients : nullptr,
                    has_l1f != 0 ? l1fb_gradients : nullptr,
                    has_l1ax != 0 ? l1axw_gradients : nullptr,
                    has_l1ax != 0 ? l1axb_gradients : nullptr,
                    batch,
                    ft_size,
                    l1_out,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    has_l1f,
                    1,
                    has_l1ax,
                    use_king_axis,
                    use_hand_axis,
                    use_progress_axis,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    factorizer_shared_alpha,
                    factorizer_king_axis_alpha,
                    factorizer_hand_axis_alpha,
                    factorizer_progress_axis_alpha,
                    factorizer_pair_alpha,
                    factorizer_axis_confidences,
                    0,
                    "sfnn dense reduce L1 params") != 0) {
                return -1;
            }
        }
    } else {
        const bool reduce_l1_params = dense_param_reduce_fast_path;
        size_t l1_threads = reduce_l1_params ? batch * ft_size : std::max(batch * ft_size, batch * ft_size * l1_out);
        if (!reduce_l1_params) {
            l1_threads = std::max(l1_threads, batch * l1_out);
        }
        if (block_count_1d(l1_threads, threads, &blocks, "sfnn_stacked_affine_backward_kernel") != 0) {
            return -1;
        }
        sfnn_stacked_affine_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
            combined,
            l1_gradients,
            l1w,
            buckets,
            combined_gradients,
            l1w_gradients,
            l1b_gradients,
            batch,
            ft_size,
            l1_out,
            num_stacks,
            reduce_l1_params ? 0 : 1);
        if (check_kernel_launch("sfnn_stacked_affine_backward_kernel launch") != 0) {
            return -1;
        }
        if (reduce_l1_params) {
            if (launch_sfnn_dense_param_reduce_tiled(
                    ctx,
                    combined,
                    nullptr,
                    l1_gradients,
                    buckets,
                    l1w_gradients,
                    l1b_gradients,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    batch,
                    ft_size,
                    l1_out,
                    num_stacks,
                    factorizer_king_axis_dim,
                    factorizer_hand_axis_dim,
                    0,
                    0,
                    0,
                    use_king_axis,
                    use_hand_axis,
                    use_progress_axis,
                    use_king_hand_pair,
                    use_king_progress_pair,
                    use_hand_progress_pair,
                    factorizer_shared_alpha,
                    factorizer_king_axis_alpha,
                    factorizer_hand_axis_alpha,
                    factorizer_progress_axis_alpha,
                    factorizer_pair_alpha,
                    factorizer_axis_confidences,
                    0,
                    "sfnn dense reduce L1 params") != 0) {
                return -1;
            }
        }
    }
    if (profile.record(5, ctx, "SFNN backward profile after L1") != 0) {
        return -1;
    }

    if (fuse_pairwise_l0 != 0) {
        if (launch_sfnn_inverse_index_l0_backward(
                ctx,
                stm_indices,
                nstm_indices,
                stm_l0,
                nstm_l0,
                combined_gradients,
                stm_l0_pre_gradients,
                nstm_l0_pre_gradients,
                l0w_gradients,
                l0b_gradients,
                batch,
                max_active,
                input_size,
                ft_size) != 0) {
            return -1;
        }
    } else {
        if (block_count_1d(batch * ft_size, threads, &blocks, "sfnn_pairwise_backward_kernel") != 0) {
            return -1;
        }
        sfnn_pairwise_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
            stm_l0, nstm_l0, combined_gradients, stm_l0_gradients, nstm_l0_gradients, batch, ft_size);
        if (check_kernel_launch("sfnn_pairwise_backward_kernel launch") != 0) {
            return -1;
        }

        if (block_count_1d(batch * ft_size, threads, &blocks, "sfnn_l0_sparse_backward_kernel") != 0) {
            return -1;
        }
        sfnn_l0_sparse_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
            stm_indices,
            nstm_indices,
            stm_l0,
            nstm_l0,
            stm_l0_gradients,
            nstm_l0_gradients,
            stm_l0_pre_gradients,
            nstm_l0_pre_gradients,
            l0w_gradients,
            l0b_gradients,
            batch,
            max_active,
            input_size,
            ft_size);
        if (check_kernel_launch("sfnn_l0_sparse_backward_kernel launch") != 0) {
            return -1;
        }
    }
    if (profile.record(6, ctx, "SFNN backward profile after L0") != 0) {
        return -1;
    }
    if (profile.finish() != 0) {
        return -1;
    }

    return 0;
}

int validate_scalar_loss(size_t batch, int kind, float loss_pow_exp, float loss_param) {
    if (batch == 0) {
        return fail_message("scalar loss batch size must be greater than zero");
    }
    if (kind != 0 && kind != 1) {
        return fail_message("scalar loss kind must be 0 (sigmoid-pow) or 1 (win-rate-model)");
    }
    if (!(std::isfinite(loss_pow_exp) && loss_pow_exp >= 1.0f)) {
        return fail_message("loss_pow_exp must be finite and >= 1");
    }
    if (kind == 1 && !std::isfinite(loss_param)) {
        return fail_message("WRM loss parameter must be finite");
    }
    return 0;
}

int launch_scalar_loss_kernels(
    BulletOuCudaCppContext* ctx,
    int kind,
    float output_inv_scale,
    float loss_pow_exp,
    float loss_param,
    size_t batch,
    const float* outputs,
    const float* targets,
    const float* entry_weights,
    float* per_sample,
    float* mean_output_gradients,
    float* weighted_sum,
    float* mean,
    int finalize_loss) {
    if (validate_scalar_loss(batch, kind, loss_pow_exp, loss_param) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }

    constexpr int threads = 256;
    int blocks = 0;
    if (block_count_1d(batch, threads, &blocks, "scalar loss reduce kernel") != 0) {
        return -1;
    }

    if (kind == 0) {
        loss_sigmoid_pow_reduce_kernel<<<blocks, threads, 0, ctx->stream>>>(
            outputs,
            targets,
            entry_weights,
            per_sample,
            mean_output_gradients,
            output_inv_scale,
            loss_pow_exp,
            batch);
        if (check_kernel_launch("loss_sigmoid_pow_reduce_kernel launch") != 0) {
            return -1;
        }
    } else {
        loss_win_rate_model_reduce_kernel<<<blocks, threads, 0, ctx->stream>>>(
            outputs,
            targets,
            entry_weights,
            per_sample,
            mean_output_gradients,
            loss_pow_exp,
            output_inv_scale,
            loss_param,
            batch);
        if (check_kernel_launch("loss_win_rate_model_reduce_kernel launch") != 0) {
            return -1;
        }
    }

    if (finalize_loss != 0) {
        loss_finalize_from_per_sample_kernel<<<1, 1, 0, ctx->stream>>>(per_sample, weighted_sum, mean, batch);
        if (check_kernel_launch("loss_finalize_from_per_sample_kernel launch") != 0) {
            return -1;
        }
    }
    return 0;
}

__global__ void kppt_table_forward_kernel(
    const int* stm_indices,
    const int* nstm_indices,
    const float* table_w,
    const float* table_b,
    const float* outw,
    const float* outb,
    float* stm_eval,
    float* nstm_eval,
    float* outputs,
    size_t batch,
    size_t max_active,
    size_t input_size) {
    size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) {
        return;
    }

    size_t sparse_base = sample * max_active;
    float stm = table_b[0];
    float nstm = table_b[0];
    for (size_t slot = 0; slot < max_active; ++slot) {
        int stm_feature = stm_indices[sparse_base + slot];
        if (stm_feature >= 0 && static_cast<size_t>(stm_feature) < input_size) {
            stm += table_w[static_cast<size_t>(stm_feature)];
        }
        int nstm_feature = nstm_indices[sparse_base + slot];
        if (nstm_feature >= 0 && static_cast<size_t>(nstm_feature) < input_size) {
            nstm += table_w[static_cast<size_t>(nstm_feature)];
        }
    }
    stm_eval[sample] = stm;
    nstm_eval[sample] = nstm;
    outputs[sample] = outw[0] * stm + outw[1] * nstm + outb[0];
}

__global__ void kppt_table_backward_kernel(
    const int* stm_indices,
    const int* nstm_indices,
    const float* stm_eval,
    const float* nstm_eval,
    const float* outw,
    const float* mean_output_gradients,
    float* table_w_gradients,
    float* table_b_gradients,
    float* outw_gradients,
    float* outb_gradients,
    size_t batch,
    size_t max_active,
    size_t input_size) {
    size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) {
        return;
    }

    float grad = mean_output_gradients[sample];
    float stm = stm_eval[sample];
    float nstm = nstm_eval[sample];
    atomicAdd(&outw_gradients[0], grad * stm);
    atomicAdd(&outw_gradients[1], grad * nstm);
    atomicAdd(&outb_gradients[0], grad);

    float stm_grad = grad * outw[0];
    float nstm_grad = grad * outw[1];
    atomicAdd(&table_b_gradients[0], stm_grad + nstm_grad);

    size_t sparse_base = sample * max_active;
    for (size_t slot = 0; slot < max_active; ++slot) {
        int stm_feature = stm_indices[sparse_base + slot];
        if (stm_feature >= 0 && static_cast<size_t>(stm_feature) < input_size) {
            atomicAdd(&table_w_gradients[static_cast<size_t>(stm_feature)], stm_grad);
        }
        int nstm_feature = nstm_indices[sparse_base + slot];
        if (nstm_feature >= 0 && static_cast<size_t>(nstm_feature) < input_size) {
            atomicAdd(&table_w_gradients[static_cast<size_t>(nstm_feature)], nstm_grad);
        }
    }
}

int validate_kppt_table_shape(size_t input_size, size_t batch, size_t max_active) {
    if (input_size == 0) {
        return fail_message("KPPT table input_size must be greater than zero");
    }
    if (batch == 0) {
        return fail_message("KPPT table batch size must be greater than zero");
    }
    if (max_active == 0) {
        return fail_message("KPPT table max_active must be greater than zero");
    }
    return 0;
}

int launch_kppt_table_forward(
    BulletOuCudaCppContext* ctx,
    const int* stm_indices,
    const int* nstm_indices,
    const float* table_w,
    const float* table_b,
    const float* outw,
    const float* outb,
    float* stm_eval,
    float* nstm_eval,
    float* outputs,
    size_t input_size,
    size_t batch,
    size_t max_active) {
    if (validate_kppt_table_shape(input_size, batch, max_active) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    constexpr int threads = 256;
    int blocks = 0;
    if (block_count_1d(batch, threads, &blocks, "kppt_table_forward_kernel") != 0) {
        return -1;
    }
    kppt_table_forward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        stm_indices, nstm_indices, table_w, table_b, outw, outb, stm_eval, nstm_eval, outputs, batch, max_active, input_size);
    return check_kernel_launch("kppt_table_forward_kernel launch");
}

int launch_kppt_table_backward(
    BulletOuCudaCppContext* ctx,
    const int* stm_indices,
    const int* nstm_indices,
    const float* stm_eval,
    const float* nstm_eval,
    const float* outw,
    const float* mean_output_gradients,
    float* table_w_gradients,
    float* table_b_gradients,
    float* outw_gradients,
    float* outb_gradients,
    size_t input_size,
    size_t batch,
    size_t max_active) {
    if (validate_kppt_table_shape(input_size, batch, max_active) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (launch_fill_f32_raw(ctx, table_w_gradients, input_size, 0.0f, "kppt zero table_w_gradients") != 0 ||
        launch_fill_f32_raw(ctx, table_b_gradients, 1, 0.0f, "kppt zero table_b_gradients") != 0 ||
        launch_fill_f32_raw(ctx, outw_gradients, 2, 0.0f, "kppt zero outw_gradients") != 0 ||
        launch_fill_f32_raw(ctx, outb_gradients, 1, 0.0f, "kppt zero outb_gradients") != 0) {
        return -1;
    }

    constexpr int threads = 256;
    int blocks = 0;
    if (block_count_1d(batch, threads, &blocks, "kppt_table_backward_kernel") != 0) {
        return -1;
    }
    kppt_table_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        stm_indices,
        nstm_indices,
        stm_eval,
        nstm_eval,
        outw,
        mean_output_gradients,
        table_w_gradients,
        table_b_gradients,
        outw_gradients,
        outb_gradients,
        batch,
        max_active,
        input_size);
    return check_kernel_launch("kppt_table_backward_kernel launch");
}

int launch_nnue_backward_kernels(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t l1,
    size_t l2,
    size_t l3,
    size_t batch,
    size_t max_active,
    const int* stm_indices,
    const int* nstm_indices,
    const float* combined,
    const float* hidden1,
    const float* hidden2,
    const float* stm_l0,
    const float* nstm_l0,
    const float* l1w,
    const float* l2w,
    const float* outw,
    const float* mean_output_gradients,
    float* hidden2_gradients,
    float* hidden1_gradients,
    float* combined_gradients,
    float* stm_l0_gradients,
    float* nstm_l0_gradients,
    float* l0w_gradients,
    float* l0b_gradients,
    float* l1w_gradients,
    float* l1b_gradients,
    float* l2w_gradients,
    float* l2b_gradients,
    float* outw_gradients,
    float* outb_gradients,
    int zero_l0_gradients) {
    if (validate_nnue_shape(input_size, l1, l2, l3, batch, max_active) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }

    constexpr int threads = 256;
    int blocks = 0;

    size_t out_threads = std::max(batch * l3, std::max(l3, static_cast<size_t>(1)));
    if (block_count_1d(out_threads, threads, &blocks, "dense_output_backward_kernel") != 0) {
        return -1;
    }
    dense_output_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        hidden2,
        mean_output_gradients,
        outw,
        hidden2_gradients,
        outw_gradients,
        outb_gradients,
        batch,
        l3);
    if (check_kernel_launch("dense_output_backward_kernel launch") != 0) {
        return -1;
    }

    if (launch_dense_crelu_backward_gemm(
            ctx,
            "dense_l2_crelu_backward_gemm",
        hidden1,
        hidden2,
        hidden2_gradients,
        l2w,
        hidden1_gradients,
        l2w_gradients,
        l2b_gradients,
        batch,
        l2,
            l3) != 0) {
        return -1;
    }

    size_t l1_input_dim = l1 * 2;
    if (launch_dense_crelu_backward_gemm(
            ctx,
            "dense_l1_crelu_backward_gemm",
            combined,
            hidden1,
            hidden1_gradients,
            l1w,
            combined_gradients,
            l1w_gradients,
            l1b_gradients,
            batch,
            l1_input_dim,
            l2) != 0) {
        return -1;
    }

    if (block_count_1d(batch * l1 * 2, threads, &blocks, "nnue_l0_crelu_backward_kernel") != 0) {
        return -1;
    }
    nnue_l0_crelu_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        combined_gradients,
        stm_l0,
        nstm_l0,
        stm_l0_gradients,
        nstm_l0_gradients,
        batch,
        l1);
    if (check_kernel_launch("nnue_l0_crelu_backward_kernel launch") != 0) {
        return -1;
    }

    if (zero_l0_gradients != 0) {
        size_t l0_zero_threads = std::max(nnue_l0w_len_for_shape(input_size, l1), l1);
        if (block_count_1d(l0_zero_threads, threads, &blocks, "nnue_l0_sparse_zero_gradients_kernel") != 0) {
            return -1;
        }
        nnue_l0_sparse_zero_gradients_kernel<<<blocks, threads, 0, ctx->stream>>>(
            l0w_gradients, l0b_gradients, input_size, l1);
        if (check_kernel_launch("nnue_l0_sparse_zero_gradients_kernel launch") != 0) {
            return -1;
        }
    }

    size_t l0_scatter_threads = batch * max_active * l1;
    if (block_count_1d(l0_scatter_threads, threads, &blocks, "nnue_l0_sparse_backward_kernel") != 0) {
        return -1;
    }
    nnue_l0_sparse_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        stm_indices,
        nstm_indices,
        stm_l0_gradients,
        nstm_l0_gradients,
        l0w_gradients,
        batch,
        max_active,
        input_size,
        l1);
    if (check_kernel_launch("nnue_l0_sparse_backward_kernel launch") != 0) {
        return -1;
    }

    nnue_l0_bias_backward_kernel<<<static_cast<int>(l1), threads, 0, ctx->stream>>>(
        stm_l0_gradients, nstm_l0_gradients, l0b_gradients, batch, l1);
    if (check_kernel_launch("nnue_l0_bias_backward_kernel launch") != 0) {
        return -1;
    }

    return 0;
}

} // namespace

extern "C" int bulletou_cuda_cpp_last_error(char* out, size_t out_len) {
    if (out == nullptr || out_len == 0) {
        return fail_message("last_error output buffer is null or empty");
    }
    size_t n = std::min(out_len - 1, g_last_error.size());
    std::memcpy(out, g_last_error.data(), n);
    out[n] = '\0';
    return 0;
}

extern "C" int bulletou_cuda_cpp_device_name(int device, char* out, size_t out_len) {
    if (out == nullptr || out_len == 0) {
        return fail_message("device_name output buffer is null or empty");
    }

    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        return fail("cudaSetDevice", status);
    }

    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) {
        return fail("cudaGetDeviceProperties", status);
    }

    size_t n = std::min(out_len - 1, std::strlen(prop.name));
    std::memcpy(out, prop.name, n);
    out[n] = '\0';
    return ok();
}

extern "C" int bulletou_cuda_cpp_context_create(int device, BulletOuCudaCppContext** out) {
    if (out == nullptr) {
        return fail_message("context_create output pointer must not be null");
    }
    *out = nullptr;

    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        return fail("cudaSetDevice", status);
    }

    BulletOuCudaCppContext* ctx = new BulletOuCudaCppContext();
    ctx->device = device;
    status = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        delete ctx;
        return fail("cudaStreamCreateWithFlags", status);
    }
    cublasStatus_t blas_status = cublasCreate(&ctx->blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        return fail_blas("cublasCreate", blas_status);
    }
    blas_status = cublasSetStream(ctx->blas, ctx->stream);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(ctx->blas);
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        return fail_blas("cublasSetStream", blas_status);
    }
    if (warmup_context(ctx) != 0) {
        cublasDestroy(ctx->blas);
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        return -1;
    }

    *out = ctx;
    return ok();
}

extern "C" int bulletou_cuda_cpp_context_destroy(BulletOuCudaCppContext* ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    cudaError_t status = cudaSetDevice(ctx->device);
    if (status != cudaSuccess) {
        delete ctx;
        return fail("cudaSetDevice", status);
    }
    if (free_i32_scratch(ctx->sfnn_inverse_counts, ctx->sfnn_inverse_counts_len, "cudaFree sfnn inverse counts") != 0 ||
        free_i32_scratch(ctx->sfnn_inverse_offsets, ctx->sfnn_inverse_offsets_len, "cudaFree sfnn inverse offsets") != 0 ||
        free_i32_scratch(ctx->sfnn_inverse_block_sums, ctx->sfnn_inverse_block_sums_len, "cudaFree sfnn inverse block sums") != 0 ||
        free_i32_scratch(ctx->sfnn_inverse_block_offsets, ctx->sfnn_inverse_block_offsets_len, "cudaFree sfnn inverse block offsets") != 0 ||
        free_i32_scratch(ctx->sfnn_inverse_write_counters, ctx->sfnn_inverse_write_counters_len, "cudaFree sfnn inverse write counters") != 0 ||
        free_i32_scratch(ctx->sfnn_inverse_positions, ctx->sfnn_inverse_positions_len, "cudaFree sfnn inverse positions") != 0) {
        delete ctx;
        return -1;
    }
    if (ctx->blas != nullptr) {
        cublasStatus_t blas_status = cublasDestroy(ctx->blas);
        if (blas_status != CUBLAS_STATUS_SUCCESS) {
            delete ctx;
            return fail_blas("cublasDestroy", blas_status);
        }
    }
    if (ctx->stream != nullptr) {
        status = cudaStreamDestroy(ctx->stream);
        if (status != cudaSuccess) {
            delete ctx;
            return fail("cudaStreamDestroy", status);
        }
    }
    delete ctx;
    return ok();
}

extern "C" int bulletou_cuda_cpp_context_synchronize(BulletOuCudaCppContext* ctx) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    cudaError_t status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaStreamSynchronize", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_event_create(BulletOuCudaCppContext* ctx, BulletOuCudaCppEvent** out) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (out == nullptr) {
        return fail_message("event_create output pointer must not be null");
    }
    *out = nullptr;

    BulletOuCudaCppEvent* event = new BulletOuCudaCppEvent();
    event->device = ctx->device;
    cudaError_t status = cudaEventCreateWithFlags(&event->event, cudaEventDefault);
    if (status != cudaSuccess) {
        delete event;
        return fail("cudaEventCreateWithFlags", status);
    }

    *out = event;
    return ok();
}

extern "C" int bulletou_cuda_cpp_event_destroy(BulletOuCudaCppEvent* event) {
    if (event == nullptr) {
        return 0;
    }
    cudaError_t status = cudaSetDevice(event->device);
    if (status != cudaSuccess) {
        delete event;
        return fail("cudaSetDevice", status);
    }
    if (event->event != nullptr) {
        status = cudaEventDestroy(event->event);
        if (status != cudaSuccess) {
            delete event;
            return fail("cudaEventDestroy", status);
        }
    }
    delete event;
    return ok();
}

extern "C" int bulletou_cuda_cpp_event_record(BulletOuCudaCppContext* ctx, BulletOuCudaCppEvent* event) {
    if (validate_event(ctx, event, "record") != 0 || set_context_device(ctx) != 0) {
        return -1;
    }
    cudaError_t status = cudaEventRecord(event->event, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaEventRecord", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_event_wait(BulletOuCudaCppContext* ctx, BulletOuCudaCppEvent* event) {
    if (validate_event(ctx, event, "wait") != 0 || set_context_device(ctx) != 0) {
        return -1;
    }
    cudaError_t status = cudaStreamWaitEvent(ctx->stream, event->event, 0);
    if (status != cudaSuccess) {
        return fail("cudaStreamWaitEvent", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_event_synchronize(BulletOuCudaCppEvent* event) {
    if (event == nullptr || event->event == nullptr) {
        return fail_message("event_synchronize event must not be null");
    }
    cudaError_t status = cudaSetDevice(event->device);
    if (status != cudaSuccess) {
        return fail("cudaSetDevice", status);
    }
    status = cudaEventSynchronize(event->event);
    if (status != cudaSuccess) {
        return fail("cudaEventSynchronize", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_event_elapsed_ms(
    BulletOuCudaCppEvent* start,
    BulletOuCudaCppEvent* stop,
    float* out_ms) {
    if (start == nullptr || start->event == nullptr || stop == nullptr || stop->event == nullptr || out_ms == nullptr) {
        return fail_message("event_elapsed_ms arguments must not be null");
    }
    if (start->device != stop->device) {
        return fail_message("event_elapsed_ms events belong to different devices");
    }
    cudaError_t status = cudaSetDevice(start->device);
    if (status != cudaSuccess) {
        return fail("cudaSetDevice", status);
    }
    status = cudaEventElapsedTime(out_ms, start->event, stop->event);
    if (status != cudaSuccess) {
        return fail("cudaEventElapsedTime", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_graph_begin_capture(BulletOuCudaCppContext* ctx) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    cudaError_t status = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeGlobal);
    if (status != cudaSuccess) {
        return fail("cudaStreamBeginCapture", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_graph_end_capture(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppGraphExec** out) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (out == nullptr) {
        return fail_message("graph_end_capture output pointer must not be null");
    }
    *out = nullptr;

    cudaGraph_t graph = nullptr;
    cudaError_t status = cudaStreamEndCapture(ctx->stream, &graph);
    if (status != cudaSuccess) {
        return fail("cudaStreamEndCapture", status);
    }

    cudaGraphExec_t exec = nullptr;
    status = cudaGraphInstantiate(&exec, graph, 0);
    if (status != cudaSuccess) {
        cudaGraphDestroy(graph);
        return fail("cudaGraphInstantiate", status);
    }

    BulletOuCudaCppGraphExec* graph_exec = new BulletOuCudaCppGraphExec();
    graph_exec->device = ctx->device;
    graph_exec->graph = graph;
    graph_exec->exec = exec;
    *out = graph_exec;
    return ok();
}

extern "C" int bulletou_cuda_cpp_graph_destroy(BulletOuCudaCppGraphExec* graph) {
    if (graph == nullptr) {
        return 0;
    }
    cudaError_t status = cudaSetDevice(graph->device);
    if (status != cudaSuccess) {
        delete graph;
        return fail("cudaSetDevice", status);
    }
    if (graph->exec != nullptr) {
        status = cudaGraphExecDestroy(graph->exec);
        if (status != cudaSuccess) {
            delete graph;
            return fail("cudaGraphExecDestroy", status);
        }
    }
    if (graph->graph != nullptr) {
        status = cudaGraphDestroy(graph->graph);
        if (status != cudaSuccess) {
            delete graph;
            return fail("cudaGraphDestroy", status);
        }
    }
    delete graph;
    return ok();
}

extern "C" int bulletou_cuda_cpp_graph_launch(BulletOuCudaCppContext* ctx, BulletOuCudaCppGraphExec* graph) {
    if (validate_graph(ctx, graph, "launch") != 0 || set_context_device(ctx) != 0) {
        return -1;
    }
    cudaError_t status = cudaGraphLaunch(graph->exec, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaGraphLaunch", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_f32_buffer_create(
    BulletOuCudaCppContext* ctx,
    size_t len,
    BulletOuCudaCppF32Buffer** out) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (out == nullptr) {
        return fail_message("f32_buffer_create output pointer must not be null");
    }
    *out = nullptr;

    BulletOuCudaCppF32Buffer* buffer = new BulletOuCudaCppF32Buffer();
    buffer->device = ctx->device;
    buffer->len = len;
    if (checked_malloc(&buffer->ptr, len, "cudaMalloc f32 buffer") != 0) {
        delete buffer;
        return -1;
    }

    *out = buffer;
    return ok();
}

extern "C" int bulletou_cuda_cpp_f32_buffer_destroy(BulletOuCudaCppF32Buffer* buffer) {
    if (buffer == nullptr) {
        return 0;
    }
    cudaError_t status = cudaSetDevice(buffer->device);
    if (status != cudaSuccess) {
        delete buffer;
        return fail("cudaSetDevice", status);
    }
    if (buffer->ptr != nullptr) {
        status = cudaFree(buffer->ptr);
        if (status != cudaSuccess) {
            delete buffer;
            return fail("cudaFree f32 buffer", status);
        }
    }
    delete buffer;
    return ok();
}

extern "C" int bulletou_cuda_cpp_i32_buffer_create(
    BulletOuCudaCppContext* ctx,
    size_t len,
    BulletOuCudaCppI32Buffer** out) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (out == nullptr) {
        return fail_message("i32_buffer_create output pointer must not be null");
    }
    *out = nullptr;

    BulletOuCudaCppI32Buffer* buffer = new BulletOuCudaCppI32Buffer();
    buffer->device = ctx->device;
    buffer->len = len;
    if (checked_malloc(&buffer->ptr, len, "cudaMalloc i32 buffer") != 0) {
        delete buffer;
        return -1;
    }

    *out = buffer;
    return ok();
}

extern "C" int bulletou_cuda_cpp_i32_buffer_destroy(BulletOuCudaCppI32Buffer* buffer) {
    if (buffer == nullptr) {
        return 0;
    }
    cudaError_t status = cudaSetDevice(buffer->device);
    if (status != cudaSuccess) {
        delete buffer;
        return fail("cudaSetDevice", status);
    }
    if (buffer->ptr != nullptr) {
        status = cudaFree(buffer->ptr);
        if (status != cudaSuccess) {
            delete buffer;
            return fail("cudaFree i32 buffer", status);
        }
    }
    delete buffer;
    return ok();
}

extern "C" int bulletou_cuda_cpp_pinned_f32_buffer_create(
    BulletOuCudaCppContext* ctx,
    size_t len,
    BulletOuCudaCppPinnedF32Buffer** out) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (out == nullptr) {
        return fail_message("pinned_f32_buffer_create output pointer must not be null");
    }
    *out = nullptr;

    BulletOuCudaCppPinnedF32Buffer* buffer = new BulletOuCudaCppPinnedF32Buffer();
    buffer->device = ctx->device;
    buffer->len = len;
    if (checked_host_malloc(&buffer->ptr, len, "cudaMallocHost pinned f32 buffer") != 0) {
        delete buffer;
        return -1;
    }

    *out = buffer;
    return ok();
}

extern "C" int bulletou_cuda_cpp_pinned_f32_buffer_destroy(BulletOuCudaCppPinnedF32Buffer* buffer) {
    if (buffer == nullptr) {
        return 0;
    }
    cudaError_t status = cudaSetDevice(buffer->device);
    if (status != cudaSuccess) {
        delete buffer;
        return fail("cudaSetDevice", status);
    }
    if (buffer->ptr != nullptr) {
        status = cudaFreeHost(buffer->ptr);
        if (status != cudaSuccess) {
            delete buffer;
            return fail("cudaFreeHost pinned f32 buffer", status);
        }
    }
    delete buffer;
    return ok();
}

extern "C" int bulletou_cuda_cpp_pinned_i32_buffer_create(
    BulletOuCudaCppContext* ctx,
    size_t len,
    BulletOuCudaCppPinnedI32Buffer** out) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (out == nullptr) {
        return fail_message("pinned_i32_buffer_create output pointer must not be null");
    }
    *out = nullptr;

    BulletOuCudaCppPinnedI32Buffer* buffer = new BulletOuCudaCppPinnedI32Buffer();
    buffer->device = ctx->device;
    buffer->len = len;
    if (checked_host_malloc(&buffer->ptr, len, "cudaMallocHost pinned i32 buffer") != 0) {
        delete buffer;
        return -1;
    }

    *out = buffer;
    return ok();
}

extern "C" int bulletou_cuda_cpp_pinned_i32_buffer_destroy(BulletOuCudaCppPinnedI32Buffer* buffer) {
    if (buffer == nullptr) {
        return 0;
    }
    cudaError_t status = cudaSetDevice(buffer->device);
    if (status != cudaSuccess) {
        delete buffer;
        return fail("cudaSetDevice", status);
    }
    if (buffer->ptr != nullptr) {
        status = cudaFreeHost(buffer->ptr);
        if (status != cudaSuccess) {
            delete buffer;
            return fail("cudaFreeHost pinned i32 buffer", status);
        }
    }
    delete buffer;
    return ok();
}

extern "C" int bulletou_cuda_cpp_f32_upload_staged(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* dst,
    BulletOuCudaCppPinnedF32Buffer* staging,
    const float* src,
    size_t len) {
    if (validate_buffer(ctx, dst, len, "dst") != 0 ||
        validate_pinned_f32_buffer(ctx, staging, len, "staging") != 0 ||
        validate_host_ptr(src, len, "src") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    std::memcpy(staging->ptr, src, len * sizeof(float));
    cudaError_t status = cudaMemcpyAsync(dst->ptr, staging->ptr, len * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemcpyAsync staged f32 upload", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_i32_upload_staged(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppI32Buffer* dst,
    BulletOuCudaCppPinnedI32Buffer* staging,
    const int* src,
    size_t len) {
    if (validate_i32_buffer(ctx, dst, len, "dst") != 0 ||
        validate_pinned_i32_buffer(ctx, staging, len, "staging") != 0 ||
        validate_host_ptr(src, len, "src") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    std::memcpy(staging->ptr, src, len * sizeof(int));
    cudaError_t status = cudaMemcpyAsync(dst->ptr, staging->ptr, len * sizeof(int), cudaMemcpyHostToDevice, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemcpyAsync staged i32 upload", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_i32_upload(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppI32Buffer* dst,
    const int* src,
    size_t len) {
    if (validate_i32_buffer(ctx, dst, len, "dst") != 0 || validate_host_ptr(src, len, "src") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    cudaError_t status = cudaMemcpyAsync(dst->ptr, src, len * sizeof(int), cudaMemcpyHostToDevice, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemcpyAsync i32 upload", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_i32_download(
    BulletOuCudaCppContext* ctx,
    const BulletOuCudaCppI32Buffer* src,
    int* dst,
    size_t len) {
    if (validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(src), len, "src") != 0 ||
        validate_host_ptr(dst, len, "dst") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    cudaError_t status = cudaMemcpyAsync(dst, src->ptr, len * sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemcpyAsync i32 download", status);
    }
    status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaStreamSynchronize i32 download", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_i32_copy_device(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppI32Buffer* dst,
    const BulletOuCudaCppI32Buffer* src,
    size_t len) {
    if (validate_i32_buffer(ctx, dst, len, "dst") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(src), len, "src") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    cudaError_t status = cudaMemcpyAsync(dst->ptr, src->ptr, len * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemcpyAsync i32 device copy", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_f32_upload(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* dst,
    const float* src,
    size_t len) {
    if (validate_buffer(ctx, dst, len, "dst") != 0 || validate_host_ptr(src, len, "src") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    cudaError_t status = cudaMemcpyAsync(dst->ptr, src, len * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemcpyAsync f32 upload", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_f32_download(
    BulletOuCudaCppContext* ctx,
    const BulletOuCudaCppF32Buffer* src,
    float* dst,
    size_t len) {
    if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src), len, "src") != 0 ||
        validate_host_ptr(dst, len, "dst") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    cudaError_t status = cudaMemcpyAsync(dst, src->ptr, len * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemcpyAsync f32 download", status);
    }
    status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaStreamSynchronize f32 download", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_f32_copy_device(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* dst,
    const BulletOuCudaCppF32Buffer* src,
    size_t len) {
    if (validate_buffer(ctx, dst, len, "dst") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src), len, "src") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    cudaError_t status = cudaMemcpyAsync(dst->ptr, src->ptr, len * sizeof(float), cudaMemcpyDeviceToDevice, ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaMemcpyAsync f32 device copy", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_f32_fill(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* dst,
    float value,
    size_t len) {
    if (validate_buffer(ctx, dst, len, "dst") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    int threads = 256;
    int blocks = static_cast<int>((len + threads - 1) / threads);
    fill_f32_kernel<<<blocks, threads, 0, ctx->stream>>>(len, value, dst->ptr);
    if (check_kernel_launch("fill_f32_kernel launch") != 0) {
        return -1;
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_axpy_device(
    BulletOuCudaCppContext* ctx,
    size_t len,
    float a,
    const BulletOuCudaCppF32Buffer* x,
    const BulletOuCudaCppF32Buffer* y,
    BulletOuCudaCppF32Buffer* out) {
    if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(x), len, "x") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(y), len, "y") != 0 ||
        validate_buffer(ctx, out, len, "out") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }
    int threads = 256;
    int blocks = static_cast<int>((len + threads - 1) / threads);
    axpy_kernel<<<blocks, threads, 0, ctx->stream>>>(len, a, x->ptr, y->ptr, out->ptr);
    if (check_kernel_launch("axpy_kernel launch") != 0) {
        return -1;
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_ranger_update_device(
    BulletOuCudaCppContext* ctx,
    size_t len,
    float gradient_factor,
    float learning_rate,
    float step_size,
    int use_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    int do_lookahead,
    float lookahead_alpha,
    BulletOuCudaCppF32Buffer* gradients,
    BulletOuCudaCppF32Buffer* weights,
    BulletOuCudaCppF32Buffer* momentum,
    BulletOuCudaCppF32Buffer* velocity,
    BulletOuCudaCppF32Buffer* slow_params) {
    if (validate_buffer(ctx, gradients, len, "gradients") != 0 || validate_buffer(ctx, weights, len, "weights") != 0 ||
        validate_buffer(ctx, momentum, len, "momentum") != 0 || validate_buffer(ctx, velocity, len, "velocity") != 0 ||
        validate_buffer(ctx, slow_params, len, "slow_params") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0) {
        return ok();
    }

    constexpr int threads = 256;
    const bool use_vec4 = (len % 4) == 0;
    const size_t update_threads = use_vec4 ? len / 4 : len;
    int blocks = 0;
    if (block_count_1d(update_threads, threads, &blocks, "radam_update_reset_gradients_kernel") != 0) {
        return -1;
    }
    if (use_vec4) {
        radam_update_reset_gradients_vec4_kernel<<<blocks, threads, 0, ctx->stream>>>(
            gradients->ptr,
            weights->ptr,
            momentum->ptr,
            velocity->ptr,
            len / 4,
            gradient_factor,
            learning_rate,
            step_size,
            use_denom,
            decay,
            beta1,
            beta2,
            epsilon,
            min_weight,
            max_weight);
        if (check_kernel_launch("radam_update_reset_gradients_vec4_kernel launch") != 0) {
            return -1;
        }
    } else {
        radam_update_reset_gradients_kernel<<<blocks, threads, 0, ctx->stream>>>(
            gradients->ptr,
            weights->ptr,
            momentum->ptr,
            velocity->ptr,
            len,
            gradient_factor,
            learning_rate,
            step_size,
            use_denom,
            decay,
            beta1,
            beta2,
            epsilon,
            min_weight,
            max_weight);
        if (check_kernel_launch("radam_update_reset_gradients_kernel launch") != 0) {
            return -1;
        }
    }

    if (do_lookahead != 0) {
        if (use_vec4) {
            ranger_lookahead_vec4_kernel<<<blocks, threads, 0, ctx->stream>>>(
                weights->ptr, slow_params->ptr, len / 4, lookahead_alpha);
            if (check_kernel_launch("ranger_lookahead_vec4_kernel launch") != 0) {
                return -1;
            }
        } else {
            ranger_lookahead_kernel<<<blocks, threads, 0, ctx->stream>>>(weights->ptr, slow_params->ptr, len, lookahead_alpha);
            if (check_kernel_launch("ranger_lookahead_kernel launch") != 0) {
                return -1;
            }
        }
    }

    return ok();
}

extern "C" int bulletou_cuda_cpp_ranger_update_stacked_dirty_device(
    BulletOuCudaCppContext* ctx,
    size_t len,
    size_t stride,
    size_t dirty_count,
    float gradient_factor,
    float learning_rate,
    float step_size,
    int use_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    int do_lookahead,
    float lookahead_alpha,
    const BulletOuCudaCppI32Buffer* dirty_buckets,
    BulletOuCudaCppF32Buffer* gradients,
    BulletOuCudaCppF32Buffer* weights,
    BulletOuCudaCppF32Buffer* momentum,
    BulletOuCudaCppF32Buffer* velocity,
    BulletOuCudaCppF32Buffer* slow_params) {
    if (validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(dirty_buckets), dirty_count, "dirty_buckets") != 0 ||
        validate_buffer(ctx, gradients, len, "gradients") != 0 || validate_buffer(ctx, weights, len, "weights") != 0 ||
        validate_buffer(ctx, momentum, len, "momentum") != 0 || validate_buffer(ctx, velocity, len, "velocity") != 0 ||
        validate_buffer(ctx, slow_params, len, "slow_params") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (len == 0 || stride == 0 || dirty_count == 0) {
        return ok();
    }
    if (dirty_count > SIZE_MAX / stride) {
        return fail_message("ranger dirty stacked update length overflow");
    }

    constexpr int threads = 256;
    const bool use_vec4 = (stride % 4) == 0 && (len % 4) == 0;
    const size_t effective_stride = use_vec4 ? stride / 4 : stride;
    const size_t effective_len = use_vec4 ? len / 4 : len;
    const size_t update_threads = dirty_count * effective_stride;
    int blocks = 0;
    if (block_count_1d(update_threads, threads, &blocks, "radam_update_reset_gradients_stacked_dirty_kernel") != 0) {
        return -1;
    }

    if (learning_rate == 0.0f) {
        if (use_vec4) {
            clear_gradients_stacked_dirty_vec4_kernel<<<blocks, threads, 0, ctx->stream>>>(
                dirty_buckets->ptr,
                dirty_count,
                effective_stride,
                effective_len,
                gradients->ptr);
            if (check_kernel_launch("clear_gradients_stacked_dirty_vec4_kernel launch") != 0) {
                return -1;
            }
        } else {
            clear_gradients_stacked_dirty_kernel<<<blocks, threads, 0, ctx->stream>>>(
                dirty_buckets->ptr,
                dirty_count,
                stride,
                len,
                gradients->ptr);
            if (check_kernel_launch("clear_gradients_stacked_dirty_kernel launch") != 0) {
                return -1;
            }
        }
        return ok();
    }

    if (use_vec4) {
        radam_update_reset_gradients_stacked_dirty_vec4_kernel<<<blocks, threads, 0, ctx->stream>>>(
            dirty_buckets->ptr,
            dirty_count,
            effective_stride,
            effective_len,
            gradients->ptr,
            weights->ptr,
            momentum->ptr,
            velocity->ptr,
            gradient_factor,
            learning_rate,
            step_size,
            use_denom,
            decay,
            beta1,
            beta2,
            epsilon,
            min_weight,
            max_weight);
        if (check_kernel_launch("radam_update_reset_gradients_stacked_dirty_vec4_kernel launch") != 0) {
            return -1;
        }
    } else {
        radam_update_reset_gradients_stacked_dirty_kernel<<<blocks, threads, 0, ctx->stream>>>(
            dirty_buckets->ptr,
            dirty_count,
            stride,
            len,
            gradients->ptr,
            weights->ptr,
            momentum->ptr,
            velocity->ptr,
            gradient_factor,
            learning_rate,
            step_size,
            use_denom,
            decay,
            beta1,
            beta2,
            epsilon,
            min_weight,
            max_weight);
        if (check_kernel_launch("radam_update_reset_gradients_stacked_dirty_kernel launch") != 0) {
            return -1;
        }
    }

    if (do_lookahead != 0) {
        if (use_vec4) {
            ranger_lookahead_stacked_dirty_vec4_kernel<<<blocks, threads, 0, ctx->stream>>>(
                dirty_buckets->ptr,
                dirty_count,
                effective_stride,
                effective_len,
                weights->ptr,
                slow_params->ptr,
                lookahead_alpha);
            if (check_kernel_launch("ranger_lookahead_stacked_dirty_vec4_kernel launch") != 0) {
                return -1;
            }
        } else {
            ranger_lookahead_stacked_dirty_kernel<<<blocks, threads, 0, ctx->stream>>>(
                dirty_buckets->ptr,
                dirty_count,
                stride,
                len,
                weights->ptr,
                slow_params->ptr,
                lookahead_alpha);
            if (check_kernel_launch("ranger_lookahead_stacked_dirty_kernel launch") != 0) {
                return -1;
            }
        }
    }

    return ok();
}

extern "C" int bulletou_cuda_cpp_nnue_forward_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t l1,
    size_t l2,
    size_t l3,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppF32Buffer* l0w,
    const BulletOuCudaCppF32Buffer* l0b,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l1b,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l2b,
    const BulletOuCudaCppF32Buffer* outw,
    const BulletOuCudaCppF32Buffer* outb,
    BulletOuCudaCppF32Buffer* stm_l0,
    BulletOuCudaCppF32Buffer* nstm_l0,
    BulletOuCudaCppF32Buffer* combined,
    BulletOuCudaCppF32Buffer* hidden1,
    BulletOuCudaCppF32Buffer* hidden2,
    BulletOuCudaCppF32Buffer* output) {
    if (validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), batch * max_active, "stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), batch * max_active, "nstm_indices") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l0w), nnue_l0w_len_for_shape(input_size, l1), "l0w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l0b), l1, "l0b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), l1 * 2 * l2, "l1w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1b), l2, "l1b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2w), l2 * l3, "l2w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2b), l3, "l2b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(outw), l3, "outw") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(outb), 1, "outb") != 0 ||
        validate_buffer(ctx, stm_l0, batch * l1, "stm_l0") != 0 ||
        validate_buffer(ctx, nstm_l0, batch * l1, "nstm_l0") != 0 ||
        validate_buffer(ctx, combined, batch * l1 * 2, "combined") != 0 ||
        validate_buffer(ctx, hidden1, batch * l2, "hidden1") != 0 ||
        validate_buffer(ctx, hidden2, batch * l3, "hidden2") != 0 ||
        validate_buffer(ctx, output, batch, "output") != 0) {
        return -1;
    }

    if (launch_nnue_forward_kernels(
            ctx,
            input_size,
            l1,
            l2,
            l3,
            batch,
            max_active,
            stm_indices->ptr,
            nstm_indices->ptr,
            l0w->ptr,
            l0b->ptr,
            l1w->ptr,
            l1b->ptr,
            l2w->ptr,
            l2b->ptr,
            outw->ptr,
            outb->ptr,
            stm_l0->ptr,
            nstm_l0->ptr,
            combined->ptr,
            hidden1->ptr,
            hidden2->ptr,
            output->ptr) != 0) {
        return -1;
    }

    return ok();
}

extern "C" int bulletou_cuda_cpp_sfnn_build_quantized_proxy_device(
    BulletOuCudaCppContext* ctx,
    size_t src_input_size,
    size_t dst_input_size,
    size_t virtual_rows,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    const BulletOuCudaCppF32Buffer* src_l0w,
    const BulletOuCudaCppF32Buffer* src_l0b,
    const BulletOuCudaCppF32Buffer* src_l1w,
    const BulletOuCudaCppF32Buffer* src_l1b,
    const BulletOuCudaCppF32Buffer* src_l1fw,
    const BulletOuCudaCppF32Buffer* src_l1fb,
    int has_l1f,
    const BulletOuCudaCppF32Buffer* src_l1axw,
    const BulletOuCudaCppF32Buffer* src_l1axb,
    int has_l1ax,
    const BulletOuCudaCppF32Buffer* src_l2w,
    const BulletOuCudaCppF32Buffer* src_l2b,
    const BulletOuCudaCppF32Buffer* src_l2fw,
    const BulletOuCudaCppF32Buffer* src_l2fb,
    int has_l2f,
    const BulletOuCudaCppF32Buffer* src_l2axw,
    const BulletOuCudaCppF32Buffer* src_l2axb,
    int has_l2ax,
    const BulletOuCudaCppF32Buffer* src_l3w,
    const BulletOuCudaCppF32Buffer* src_l3b,
    const BulletOuCudaCppF32Buffer* src_l3fw,
    const BulletOuCudaCppF32Buffer* src_l3fb,
    int has_l3f,
    const BulletOuCudaCppF32Buffer* src_l3axw,
    const BulletOuCudaCppF32Buffer* src_l3axb,
    int has_l3ax,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const BulletOuCudaCppF32Buffer* factorizer_axis_confidences,
    BulletOuCudaCppF32Buffer* dst_l0w,
    BulletOuCudaCppF32Buffer* dst_l0b,
    BulletOuCudaCppF32Buffer* dst_l1w,
    BulletOuCudaCppF32Buffer* dst_l1b,
    BulletOuCudaCppF32Buffer* dst_l2w,
    BulletOuCudaCppF32Buffer* dst_l2b,
    BulletOuCudaCppF32Buffer* dst_l3w,
    BulletOuCudaCppF32Buffer* dst_l3b) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (l1_group_count != 1 || l1_common_size != 0 || l1_shard_size != 0) {
        return fail_message("sfnn quantized GPU proxy currently supports only dense L1 source weights");
    }
    if (src_input_size == 0 || dst_input_size == 0 || ft_size == 0 || l2_size == 0 || num_stacks == 0) {
        return fail_message("sfnn quantized GPU proxy dimensions must be non-zero");
    }
    if (!(src_input_size == dst_input_size ||
          (virtual_rows != 0 && src_input_size == dst_input_size + virtual_rows))) {
        return fail_message("sfnn quantized GPU proxy input sizes do not match base/factorized layout");
    }
    if (validate_sfnn_shape(
            src_input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            1,
            1) != 0) {
        return -1;
    }

    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    const size_t l2_in = l1_hidden * 2;
    const size_t axis_count = sfnn_factorizer_axis_count(
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair);
    const size_t src_l0w_len = src_input_size * ft_size;
    const size_t dst_l0w_len = dst_input_size * ft_size;
    const size_t l1w_len = num_stacks * l1_out * ft_size;
    const size_t l1b_len = num_stacks * l1_out;
    const size_t l2w_len = num_stacks * l2_size * l2_in;
    const size_t l2b_len = num_stacks * l2_size;
    const size_t l3w_len = num_stacks * l2_size;
    const size_t l3b_len = num_stacks;
    const float* factorizer_axis_confidences_ptr = nullptr;

    if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l0w), src_l0w_len, "sfnn proxy src l0w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l0b), ft_size, "sfnn proxy src l0b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l1w), l1w_len, "sfnn proxy src l1w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l1b), l1b_len, "sfnn proxy src l1b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l2w), l2w_len, "sfnn proxy src l2w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l2b), l2b_len, "sfnn proxy src l2b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l3w), l3w_len, "sfnn proxy src l3w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l3b), l3b_len, "sfnn proxy src l3b") != 0 ||
        validate_buffer(ctx, dst_l0w, dst_l0w_len, "sfnn proxy dst l0w") != 0 ||
        validate_buffer(ctx, dst_l0b, ft_size, "sfnn proxy dst l0b") != 0 ||
        validate_buffer(ctx, dst_l1w, l1w_len, "sfnn proxy dst l1w") != 0 ||
        validate_buffer(ctx, dst_l1b, l1b_len, "sfnn proxy dst l1b") != 0 ||
        validate_buffer(ctx, dst_l2w, l2w_len, "sfnn proxy dst l2w") != 0 ||
        validate_buffer(ctx, dst_l2b, l2b_len, "sfnn proxy dst l2b") != 0 ||
        validate_buffer(ctx, dst_l3w, l3w_len, "sfnn proxy dst l3w") != 0 ||
        validate_buffer(ctx, dst_l3b, l3b_len, "sfnn proxy dst l3b") != 0) {
        return -1;
    }

    if (has_l1f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l1fw), ft_size * l1_out, "sfnn proxy src l1fw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l1fb), l1_out, "sfnn proxy src l1fb") != 0) {
            return -1;
        }
    }
    if (has_l1ax != 0) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l1axw), axis_count * ft_size * l1_out, "sfnn proxy src l1axw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l1axb), axis_count * l1_out, "sfnn proxy src l1axb") != 0) {
            return -1;
        }
    }
    if (has_l2f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l2fw), l2_size * l2_in, "sfnn proxy src l2fw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l2fb), l2_size, "sfnn proxy src l2fb") != 0) {
            return -1;
        }
    }
    if (has_l2ax != 0) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l2axw), axis_count * l2_size * l2_in, "sfnn proxy src l2axw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l2axb), axis_count * l2_size, "sfnn proxy src l2axb") != 0) {
            return -1;
        }
    }
    if (has_l3f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l3fw), l2_size, "sfnn proxy src l3fw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l3fb), 1, "sfnn proxy src l3fb") != 0) {
            return -1;
        }
    }
    if (has_l3ax != 0) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l3axw), axis_count * l2_size, "sfnn proxy src l3axw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(src_l3axb), axis_count, "sfnn proxy src l3axb") != 0) {
            return -1;
        }
    }
    if (factorizer_axis_confidences != nullptr) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(factorizer_axis_confidences), axis_count, "sfnn proxy factorizer axis confidences") != 0) {
            return -1;
        }
        factorizer_axis_confidences_ptr = factorizer_axis_confidences->ptr;
    }

    constexpr int threads = 256;
    int blocks = 0;
    constexpr float qa = 127.0f;
    constexpr float qb = 64.0f;
    constexpr float fc_bias_scale = qa * qb;

    if (block_count_1d(dst_l0w_len, threads, &blocks, "sfnn proxy l0w") != 0) return -1;
    sfnn_build_quantized_proxy_l0w_kernel<<<blocks, threads, 0, ctx->stream>>>(
        src_l0w->ptr,
        dst_l0w->ptr,
        src_input_size,
        dst_input_size,
        virtual_rows,
        ft_size,
        qa);
    if (check_kernel_launch("sfnn_build_quantized_proxy_l0w_kernel launch") != 0) return -1;

    if (block_count_1d(ft_size, threads, &blocks, "sfnn proxy l0b") != 0) return -1;
    sfnn_build_quantized_proxy_vector_kernel<<<blocks, threads, 0, ctx->stream>>>(
        src_l0b->ptr,
        dst_l0b->ptr,
        ft_size,
        qa,
        -32768.0f,
        32767.0f);
    if (check_kernel_launch("sfnn_build_quantized_proxy_l0b_kernel launch") != 0) return -1;

    if (block_count_1d(l1w_len, threads, &blocks, "sfnn proxy l1w") != 0) return -1;
    sfnn_build_quantized_proxy_stacked_weights_kernel<<<blocks, threads, 0, ctx->stream>>>(
        src_l1w->ptr,
        has_l1f != 0 ? src_l1fw->ptr : nullptr,
        has_l1ax != 0 ? src_l1axw->ptr : nullptr,
        dst_l1w->ptr,
        ft_size,
        l1_out,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l1f,
        1,
        has_l1ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        qb);
    if (check_kernel_launch("sfnn_build_quantized_proxy_l1w_kernel launch") != 0) return -1;

    if (block_count_1d(l1b_len, threads, &blocks, "sfnn proxy l1b") != 0) return -1;
    sfnn_build_quantized_proxy_stacked_bias_kernel<<<blocks, threads, 0, ctx->stream>>>(
        src_l1b->ptr,
        has_l1f != 0 ? src_l1fb->ptr : nullptr,
        has_l1ax != 0 ? src_l1axb->ptr : nullptr,
        dst_l1b->ptr,
        l1_out,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l1f,
        has_l1ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        fc_bias_scale);
    if (check_kernel_launch("sfnn_build_quantized_proxy_l1b_kernel launch") != 0) return -1;

    if (block_count_1d(l2w_len, threads, &blocks, "sfnn proxy l2w") != 0) return -1;
    sfnn_build_quantized_proxy_stacked_weights_kernel<<<blocks, threads, 0, ctx->stream>>>(
        src_l2w->ptr,
        has_l2f != 0 ? src_l2fw->ptr : nullptr,
        has_l2ax != 0 ? src_l2axw->ptr : nullptr,
        dst_l2w->ptr,
        l2_in,
        l2_size,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l2f,
        0,
        has_l2ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        qb);
    if (check_kernel_launch("sfnn_build_quantized_proxy_l2w_kernel launch") != 0) return -1;

    if (block_count_1d(l2b_len, threads, &blocks, "sfnn proxy l2b") != 0) return -1;
    sfnn_build_quantized_proxy_stacked_bias_kernel<<<blocks, threads, 0, ctx->stream>>>(
        src_l2b->ptr,
        has_l2f != 0 ? src_l2fb->ptr : nullptr,
        has_l2ax != 0 ? src_l2axb->ptr : nullptr,
        dst_l2b->ptr,
        l2_size,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l2f,
        has_l2ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        fc_bias_scale);
    if (check_kernel_launch("sfnn_build_quantized_proxy_l2b_kernel launch") != 0) return -1;

    if (block_count_1d(l3w_len, threads, &blocks, "sfnn proxy l3w") != 0) return -1;
    sfnn_build_quantized_proxy_stacked_weights_kernel<<<blocks, threads, 0, ctx->stream>>>(
        src_l3w->ptr,
        has_l3f != 0 ? src_l3fw->ptr : nullptr,
        has_l3ax != 0 ? src_l3axw->ptr : nullptr,
        dst_l3w->ptr,
        l2_size,
        1,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l3f,
        0,
        has_l3ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        qb);
    if (check_kernel_launch("sfnn_build_quantized_proxy_l3w_kernel launch") != 0) return -1;

    if (block_count_1d(l3b_len, threads, &blocks, "sfnn proxy l3b") != 0) return -1;
    sfnn_build_quantized_proxy_stacked_bias_kernel<<<blocks, threads, 0, ctx->stream>>>(
        src_l3b->ptr,
        has_l3f != 0 ? src_l3fb->ptr : nullptr,
        has_l3ax != 0 ? src_l3axb->ptr : nullptr,
        dst_l3b->ptr,
        1,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_l3f,
        has_l3ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        fc_bias_scale);
    if (check_kernel_launch("sfnn_build_quantized_proxy_l3b_kernel launch") != 0) return -1;

    return ok();
}

extern "C" int bulletou_cuda_cpp_sfnn_forward_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppI32Buffer* buckets,
    const BulletOuCudaCppF32Buffer* l0w,
    BulletOuCudaCppF32Buffer* folded_l0w,
    const BulletOuCudaCppF32Buffer* l0b,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l1b,
    const BulletOuCudaCppF32Buffer* l1fw,
    const BulletOuCudaCppF32Buffer* l1fb,
    int has_l1f,
    const BulletOuCudaCppF32Buffer* l1axw,
    const BulletOuCudaCppF32Buffer* l1axb,
    int has_l1ax,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l2b,
    const BulletOuCudaCppF32Buffer* l2fw,
    const BulletOuCudaCppF32Buffer* l2fb,
    int has_l2f,
    const BulletOuCudaCppF32Buffer* l2axw,
    const BulletOuCudaCppF32Buffer* l2axb,
    int has_l2ax,
    const BulletOuCudaCppF32Buffer* l3w,
    const BulletOuCudaCppF32Buffer* l3b,
    const BulletOuCudaCppF32Buffer* l3fw,
    const BulletOuCudaCppF32Buffer* l3fb,
    int has_l3f,
    const BulletOuCudaCppF32Buffer* l3axw,
    const BulletOuCudaCppF32Buffer* l3axb,
    int has_l3ax,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const BulletOuCudaCppF32Buffer* factorizer_axis_confidences,
    BulletOuCudaCppF32Buffer* stm_l0,
    BulletOuCudaCppF32Buffer* nstm_l0,
    BulletOuCudaCppF32Buffer* combined,
    BulletOuCudaCppF32Buffer* l1,
    BulletOuCudaCppF32Buffer* l2_input,
    BulletOuCudaCppF32Buffer* l2,
    BulletOuCudaCppF32Buffer* output) {
    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    const size_t l2_in = l1_hidden * 2;
    const size_t axis_count = sfnn_factorizer_axis_count(
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair);
    const size_t l1w_len =
        sfnn_l1w_len_for_shape(ft_size, l1_hidden, l1_skip, num_stacks, l1_group_count, l1_common_size, l1_shard_size);
    const bool common_shard_l1 = sfnn_is_common_shard_l1_shape(l1_common_size, l1_shard_size);
    const bool grouped_l1 = !common_shard_l1 && sfnn_is_grouped_l1_shape(l1_group_count);
    const float* factorizer_axis_confidences_ptr = nullptr;
    if (validate_sfnn_shape(
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            batch,
            max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), batch * max_active, "sfnn stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), batch * max_active, "sfnn nstm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(buckets), batch, "sfnn buckets") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l0w), input_size * ft_size, "sfnn l0w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l0b), ft_size, "sfnn l0b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), l1w_len, "sfnn l1w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1b), num_stacks * l1_out, "sfnn l1b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2w), num_stacks * l2_size * l2_in, "sfnn l2w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2b), num_stacks * l2_size, "sfnn l2b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3w), num_stacks * l2_size, "sfnn l3w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3b), num_stacks, "sfnn l3b") != 0 ||
        validate_buffer(ctx, stm_l0, batch * ft_size, "sfnn stm_l0") != 0 ||
        validate_buffer(ctx, nstm_l0, batch * ft_size, "sfnn nstm_l0") != 0 ||
        validate_buffer(ctx, combined, batch * ft_size, "sfnn combined") != 0 ||
        validate_buffer(ctx, l1, batch * l1_out, "sfnn l1") != 0 ||
        validate_buffer(ctx, l2_input, batch * l2_in, "sfnn l2_input") != 0 ||
        validate_buffer(ctx, l2, batch * l2_size, "sfnn l2") != 0 ||
        validate_buffer(ctx, output, batch, "sfnn output") != 0) {
        return -1;
    }
    if (folded_l0w != nullptr &&
        validate_buffer(ctx, folded_l0w, input_size * ft_size, "sfnn folded_l0w scratch") != 0) {
        return -1;
    }
    if (has_l1f != 0) {
        if (grouped_l1 || common_shard_l1) {
            return fail_message("sfnn compact L1 does not support l1fw/l1fb");
        }
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1fw), ft_size * l1_out, "sfnn l1fw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1fb), l1_out, "sfnn l1fb") != 0) {
            return -1;
        }
    } else if (l1fw != nullptr || l1fb != nullptr) {
        return fail_message("sfnn l1fw/l1fb must be null when has_l1f is false");
    }
    if (has_l1ax != 0) {
        if (grouped_l1 || common_shard_l1) {
            return fail_message("sfnn compact L1 does not support l1axw/l1axb");
        }
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1axw), axis_count * ft_size * l1_out, "sfnn l1axw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1axb), axis_count * l1_out, "sfnn l1axb") != 0) {
            return -1;
        }
    } else if (l1axw != nullptr || l1axb != nullptr) {
        return fail_message("sfnn l1axw/l1axb must be null when has_l1ax is false");
    }
    if (has_l2f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2fw), l2_size * l2_in, "sfnn l2fw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2fb), l2_size, "sfnn l2fb") != 0) {
            return -1;
        }
    } else if (l2fw != nullptr || l2fb != nullptr) {
        return fail_message("sfnn l2fw/l2fb must be null when has_l2f is false");
    }
    if (has_l2ax != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2axw), axis_count * l2_size * l2_in, "sfnn l2axw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2axb), axis_count * l2_size, "sfnn l2axb") != 0) {
            return -1;
        }
    } else if (l2axw != nullptr || l2axb != nullptr) {
        return fail_message("sfnn l2axw/l2axb must be null when has_l2ax is false");
    }
    if (has_l3f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3fw), l2_size, "sfnn l3fw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3fb), 1, "sfnn l3fb") != 0) {
            return -1;
        }
    } else if (l3fw != nullptr || l3fb != nullptr) {
        return fail_message("sfnn l3fw/l3fb must be null when has_l3f is false");
    }
    if (has_l3ax != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3axw), axis_count * l2_size, "sfnn l3axw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3axb), axis_count, "sfnn l3axb") != 0) {
            return -1;
        }
    } else if (l3axw != nullptr || l3axb != nullptr) {
        return fail_message("sfnn l3axw/l3axb must be null when has_l3ax is false");
    }
    if (factorizer_axis_confidences != nullptr) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(factorizer_axis_confidences), axis_count, "sfnn factorizer axis confidences") != 0) {
            return -1;
        }
        factorizer_axis_confidences_ptr = factorizer_axis_confidences->ptr;
    }

    if (launch_sfnn_forward_kernels(
            ctx,
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            batch,
            max_active,
            stm_indices->ptr,
            nstm_indices->ptr,
            buckets->ptr,
            l0w->ptr,
            folded_l0w != nullptr ? folded_l0w->ptr : nullptr,
            l0b->ptr,
            l1w->ptr,
            l1b->ptr,
            has_l1f != 0 ? l1fw->ptr : nullptr,
            has_l1f != 0 ? l1fb->ptr : nullptr,
            has_l1f,
            has_l1ax != 0 ? l1axw->ptr : nullptr,
            has_l1ax != 0 ? l1axb->ptr : nullptr,
            has_l1ax,
            l2w->ptr,
            l2b->ptr,
            has_l2f != 0 ? l2fw->ptr : nullptr,
            has_l2f != 0 ? l2fb->ptr : nullptr,
            has_l2f,
            has_l2ax != 0 ? l2axw->ptr : nullptr,
            has_l2ax != 0 ? l2axb->ptr : nullptr,
            has_l2ax,
            l3w->ptr,
            l3b->ptr,
            has_l3f != 0 ? l3fw->ptr : nullptr,
            has_l3f != 0 ? l3fb->ptr : nullptr,
            has_l3f,
            has_l3ax != 0 ? l3axw->ptr : nullptr,
            has_l3ax != 0 ? l3axb->ptr : nullptr,
            has_l3ax,
            use_king_axis,
            use_hand_axis,
            use_progress_axis,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            factorizer_shared_alpha,
            factorizer_king_axis_alpha,
            factorizer_hand_axis_alpha,
            factorizer_progress_axis_alpha,
            factorizer_pair_alpha,
            factorizer_axis_confidences_ptr,
            stm_l0->ptr,
            nstm_l0->ptr,
            combined->ptr,
            l1->ptr,
            l2_input->ptr,
            l2->ptr,
            output->ptr) != 0) {
        return -1;
    }

    return ok();
}

extern "C" int bulletou_cuda_cpp_nnue_forward_host(
    int device,
    size_t input_size,
    size_t l1,
    size_t l2,
    size_t l3,
    size_t batch,
    size_t max_active,
    const int* stm_indices,
    const int* nstm_indices,
    const float* l0w,
    const float* l0b,
    const float* l1w,
    const float* l1b,
    const float* l2w,
    const float* l2b,
    const float* outw,
    const float* outb,
    float* output) {
    if (validate_nnue_shape(input_size, l1, l2, l3, batch, max_active) != 0 ||
        validate_host_ptr(stm_indices, batch * max_active, "stm_indices") != 0 ||
        validate_host_ptr(nstm_indices, batch * max_active, "nstm_indices") != 0 ||
        validate_host_ptr(l0w, nnue_l0w_len_for_shape(input_size, l1), "l0w") != 0 ||
        validate_host_ptr(l0b, l1, "l0b") != 0 ||
        validate_host_ptr(l1w, l1 * 2 * l2, "l1w") != 0 ||
        validate_host_ptr(l1b, l2, "l1b") != 0 ||
        validate_host_ptr(l2w, l2 * l3, "l2w") != 0 ||
        validate_host_ptr(l2b, l3, "l2b") != 0 ||
        validate_host_ptr(outw, l3, "outw") != 0 ||
        validate_host_ptr(outb, 1, "outb") != 0 ||
        validate_host_ptr(output, batch, "output") != 0) {
        return -1;
    }

    BulletOuCudaCppContext* ctx = nullptr;
    if (bulletou_cuda_cpp_context_create(device, &ctx) != 0) {
        return -1;
    }

    BulletOuCudaCppI32Buffer* d_stm = nullptr;
    BulletOuCudaCppI32Buffer* d_nstm = nullptr;
    BulletOuCudaCppF32Buffer* d_l0w = nullptr;
    BulletOuCudaCppF32Buffer* d_l0b = nullptr;
    BulletOuCudaCppF32Buffer* d_l1w = nullptr;
    BulletOuCudaCppF32Buffer* d_l1b = nullptr;
    BulletOuCudaCppF32Buffer* d_l2w = nullptr;
    BulletOuCudaCppF32Buffer* d_l2b = nullptr;
    BulletOuCudaCppF32Buffer* d_outw = nullptr;
    BulletOuCudaCppF32Buffer* d_outb = nullptr;
    BulletOuCudaCppF32Buffer* d_stm_l0 = nullptr;
    BulletOuCudaCppF32Buffer* d_nstm_l0 = nullptr;
    BulletOuCudaCppF32Buffer* d_combined = nullptr;
    BulletOuCudaCppF32Buffer* d_hidden1 = nullptr;
    BulletOuCudaCppF32Buffer* d_hidden2 = nullptr;
    BulletOuCudaCppF32Buffer* d_output = nullptr;

    int rc = 0;
    if (rc == 0) rc = bulletou_cuda_cpp_i32_buffer_create(ctx, batch * max_active, &d_stm);
    if (rc == 0) rc = bulletou_cuda_cpp_i32_buffer_create(ctx, batch * max_active, &d_nstm);
    const size_t l0w_len = nnue_l0w_len_for_shape(input_size, l1);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, l0w_len, &d_l0w);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, l1, &d_l0b);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, l1 * 2 * l2, &d_l1w);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, l2, &d_l1b);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, l2 * l3, &d_l2w);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, l3, &d_l2b);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, l3, &d_outw);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, 1, &d_outb);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch * l1, &d_stm_l0);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch * l1, &d_nstm_l0);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch * l1 * 2, &d_combined);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch * l2, &d_hidden1);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch * l3, &d_hidden2);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch, &d_output);

    if (rc == 0) {
        if (rc == 0) rc = bulletou_cuda_cpp_i32_upload(ctx, d_stm, stm_indices, batch * max_active);
        if (rc == 0) rc = bulletou_cuda_cpp_i32_upload(ctx, d_nstm, nstm_indices, batch * max_active);
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_l0w, l0w, l0w_len);
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_l0b, l0b, l1);
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_l1w, l1w, l1 * 2 * l2);
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_l1b, l1b, l2);
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_l2w, l2w, l2 * l3);
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_l2b, l2b, l3);
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_outw, outw, l3);
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_outb, outb, 1);
    }

    if (rc == 0) {
        rc = bulletou_cuda_cpp_nnue_forward_device(
            ctx,
            input_size,
            l1,
            l2,
            l3,
            batch,
            max_active,
            d_stm,
            d_nstm,
            d_l0w,
            d_l0b,
            d_l1w,
            d_l1b,
            d_l2w,
            d_l2b,
            d_outw,
            d_outb,
            d_stm_l0,
            d_nstm_l0,
            d_combined,
            d_hidden1,
            d_hidden2,
            d_output);
    }
    if (rc == 0) {
        rc = bulletou_cuda_cpp_f32_download(ctx, d_output, output, batch);
    }

    bulletou_cuda_cpp_i32_buffer_destroy(d_stm);
    bulletou_cuda_cpp_i32_buffer_destroy(d_nstm);
    bulletou_cuda_cpp_f32_buffer_destroy(d_l0w);
    bulletou_cuda_cpp_f32_buffer_destroy(d_l0b);
    bulletou_cuda_cpp_f32_buffer_destroy(d_l1w);
    bulletou_cuda_cpp_f32_buffer_destroy(d_l1b);
    bulletou_cuda_cpp_f32_buffer_destroy(d_l2w);
    bulletou_cuda_cpp_f32_buffer_destroy(d_l2b);
    bulletou_cuda_cpp_f32_buffer_destroy(d_outw);
    bulletou_cuda_cpp_f32_buffer_destroy(d_outb);
    bulletou_cuda_cpp_f32_buffer_destroy(d_stm_l0);
    bulletou_cuda_cpp_f32_buffer_destroy(d_nstm_l0);
    bulletou_cuda_cpp_f32_buffer_destroy(d_combined);
    bulletou_cuda_cpp_f32_buffer_destroy(d_hidden1);
    bulletou_cuda_cpp_f32_buffer_destroy(d_hidden2);
    bulletou_cuda_cpp_f32_buffer_destroy(d_output);
    bulletou_cuda_cpp_context_destroy(ctx);

    return rc == 0 ? ok() : rc;
}

extern "C" int bulletou_cuda_cpp_scalar_loss_device_with_finalize(
    BulletOuCudaCppContext* ctx,
    int kind,
    float output_inv_scale,
    float loss_pow_exp,
    float loss_param,
    size_t batch,
    const BulletOuCudaCppF32Buffer* outputs,
    const BulletOuCudaCppF32Buffer* targets,
    const BulletOuCudaCppF32Buffer* entry_weights,
    BulletOuCudaCppF32Buffer* per_sample,
    BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* weighted_sum,
    BulletOuCudaCppF32Buffer* mean,
    int finalize_loss) {
    if (validate_scalar_loss(batch, kind, loss_pow_exp, loss_param) != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(outputs), batch, "outputs") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(targets), batch, "targets") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(entry_weights), batch, "entry_weights") != 0 ||
        validate_buffer(ctx, per_sample, batch, "per_sample") != 0 ||
        validate_buffer(ctx, mean_output_gradients, batch, "mean_output_gradients") != 0 ||
        validate_buffer(ctx, weighted_sum, 1, "weighted_sum") != 0 ||
        validate_buffer(ctx, mean, 1, "mean") != 0) {
        return -1;
    }

    if (launch_scalar_loss_kernels(
            ctx,
            kind,
            output_inv_scale,
            loss_pow_exp,
            loss_param,
            batch,
            outputs->ptr,
            targets->ptr,
            entry_weights->ptr,
            per_sample->ptr,
            mean_output_gradients->ptr,
            weighted_sum->ptr,
            mean->ptr,
            finalize_loss) != 0) {
        return -1;
    }

    return ok();
}

extern "C" int bulletou_cuda_cpp_scalar_loss_device(
    BulletOuCudaCppContext* ctx,
    int kind,
    float output_inv_scale,
    float loss_pow_exp,
    float unused_loss_param,
    size_t batch,
    const BulletOuCudaCppF32Buffer* outputs,
    const BulletOuCudaCppF32Buffer* targets,
    const BulletOuCudaCppF32Buffer* entry_weights,
    BulletOuCudaCppF32Buffer* per_sample,
    BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* weighted_sum,
    BulletOuCudaCppF32Buffer* mean) {
    return bulletou_cuda_cpp_scalar_loss_device_with_finalize(
        ctx,
        kind,
        output_inv_scale,
        loss_pow_exp,
        unused_loss_param,
        batch,
        outputs,
        targets,
        entry_weights,
        per_sample,
        mean_output_gradients,
        weighted_sum,
        mean,
        1);
}

extern "C" int bulletou_cuda_cpp_kppt_forward_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppF32Buffer* table_w,
    const BulletOuCudaCppF32Buffer* table_b,
    const BulletOuCudaCppF32Buffer* outw,
    const BulletOuCudaCppF32Buffer* outb,
    BulletOuCudaCppF32Buffer* stm_eval,
    BulletOuCudaCppF32Buffer* nstm_eval,
    BulletOuCudaCppF32Buffer* outputs) {
    size_t sparse_len = batch * max_active;
    if (validate_kppt_table_shape(input_size, batch, max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), sparse_len, "kppt stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), sparse_len, "kppt nstm_indices") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(table_w), input_size, "kppt table_w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(table_b), 1, "kppt table_b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(outw), 2, "kppt outw") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(outb), 1, "kppt outb") != 0 ||
        validate_buffer(ctx, stm_eval, batch, "kppt stm_eval") != 0 ||
        validate_buffer(ctx, nstm_eval, batch, "kppt nstm_eval") != 0 ||
        validate_buffer(ctx, outputs, batch, "kppt outputs") != 0) {
        return -1;
    }

    if (launch_kppt_table_forward(
            ctx,
            stm_indices->ptr,
            nstm_indices->ptr,
            table_w->ptr,
            table_b->ptr,
            outw->ptr,
            outb->ptr,
            stm_eval->ptr,
            nstm_eval->ptr,
            outputs->ptr,
            input_size,
            batch,
            max_active) != 0) {
        return -1;
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_kppt_backward_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppF32Buffer* stm_eval,
    const BulletOuCudaCppF32Buffer* nstm_eval,
    const BulletOuCudaCppF32Buffer* outw,
    const BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* table_w_gradients,
    BulletOuCudaCppF32Buffer* table_b_gradients,
    BulletOuCudaCppF32Buffer* outw_gradients,
    BulletOuCudaCppF32Buffer* outb_gradients) {
    size_t sparse_len = batch * max_active;
    if (validate_kppt_table_shape(input_size, batch, max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), sparse_len, "kppt stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), sparse_len, "kppt nstm_indices") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(stm_eval), batch, "kppt stm_eval") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(nstm_eval), batch, "kppt nstm_eval") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(outw), 2, "kppt outw") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(mean_output_gradients), batch, "kppt mean_output_gradients") != 0 ||
        validate_buffer(ctx, table_w_gradients, input_size, "kppt table_w_gradients") != 0 ||
        validate_buffer(ctx, table_b_gradients, 1, "kppt table_b_gradients") != 0 ||
        validate_buffer(ctx, outw_gradients, 2, "kppt outw_gradients") != 0 ||
        validate_buffer(ctx, outb_gradients, 1, "kppt outb_gradients") != 0) {
        return -1;
    }

    if (launch_kppt_table_backward(
            ctx,
            stm_indices->ptr,
            nstm_indices->ptr,
            stm_eval->ptr,
            nstm_eval->ptr,
            outw->ptr,
            mean_output_gradients->ptr,
            table_w_gradients->ptr,
            table_b_gradients->ptr,
            outw_gradients->ptr,
            outb_gradients->ptr,
            input_size,
            batch,
            max_active) != 0) {
        return -1;
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_scalar_loss_host(
    int device,
    int kind,
    float output_inv_scale,
    float loss_pow_exp,
    float unused_loss_param,
    size_t batch,
    const float* outputs,
    const float* targets,
    const float* entry_weights,
    float* per_sample,
    float* mean_output_gradients,
    float* weighted_sum,
    float* mean) {
    if (validate_scalar_loss(batch, kind, loss_pow_exp, unused_loss_param) != 0 ||
        validate_host_ptr(outputs, batch, "outputs") != 0 ||
        validate_host_ptr(targets, batch, "targets") != 0 ||
        validate_host_ptr(entry_weights, batch, "entry_weights") != 0 ||
        validate_host_ptr(per_sample, batch, "per_sample") != 0 ||
        validate_host_ptr(mean_output_gradients, batch, "mean_output_gradients") != 0 ||
        validate_host_ptr(weighted_sum, 1, "weighted_sum") != 0 ||
        validate_host_ptr(mean, 1, "mean") != 0) {
        return -1;
    }

    BulletOuCudaCppContext* ctx = nullptr;
    if (bulletou_cuda_cpp_context_create(device, &ctx) != 0) {
        return -1;
    }

    BulletOuCudaCppF32Buffer* d_outputs = nullptr;
    BulletOuCudaCppF32Buffer* d_targets = nullptr;
    BulletOuCudaCppF32Buffer* d_entry_weights = nullptr;
    BulletOuCudaCppF32Buffer* d_per_sample = nullptr;
    BulletOuCudaCppF32Buffer* d_mean_output_gradients = nullptr;
    BulletOuCudaCppF32Buffer* d_weighted_sum = nullptr;
    BulletOuCudaCppF32Buffer* d_mean = nullptr;

    int rc = 0;
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch, &d_outputs);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch, &d_targets);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch, &d_entry_weights);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch, &d_per_sample);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, batch, &d_mean_output_gradients);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, 1, &d_weighted_sum);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, 1, &d_mean);

    if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_outputs, outputs, batch);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_targets, targets, batch);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_entry_weights, entry_weights, batch);

    if (rc == 0) {
        rc = bulletou_cuda_cpp_scalar_loss_device(
            ctx,
            kind,
            output_inv_scale,
            loss_pow_exp,
            unused_loss_param,
            batch,
            d_outputs,
            d_targets,
            d_entry_weights,
            d_per_sample,
            d_mean_output_gradients,
            d_weighted_sum,
            d_mean);
    }

    if (rc == 0) rc = bulletou_cuda_cpp_f32_download(ctx, d_per_sample, per_sample, batch);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_download(ctx, d_mean_output_gradients, mean_output_gradients, batch);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_download(ctx, d_weighted_sum, weighted_sum, 1);
    if (rc == 0) rc = bulletou_cuda_cpp_f32_download(ctx, d_mean, mean, 1);

    bulletou_cuda_cpp_f32_buffer_destroy(d_outputs);
    bulletou_cuda_cpp_f32_buffer_destroy(d_targets);
    bulletou_cuda_cpp_f32_buffer_destroy(d_entry_weights);
    bulletou_cuda_cpp_f32_buffer_destroy(d_per_sample);
    bulletou_cuda_cpp_f32_buffer_destroy(d_mean_output_gradients);
    bulletou_cuda_cpp_f32_buffer_destroy(d_weighted_sum);
    bulletou_cuda_cpp_f32_buffer_destroy(d_mean);
    bulletou_cuda_cpp_context_destroy(ctx);

    return rc == 0 ? ok() : rc;
}

extern "C" int bulletou_cuda_cpp_nnue_backward_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t l1,
    size_t l2,
    size_t l3,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppF32Buffer* combined,
    const BulletOuCudaCppF32Buffer* hidden1,
    const BulletOuCudaCppF32Buffer* hidden2,
    const BulletOuCudaCppF32Buffer* stm_l0,
    const BulletOuCudaCppF32Buffer* nstm_l0,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* outw,
    const BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* hidden2_gradients,
    BulletOuCudaCppF32Buffer* hidden1_gradients,
    BulletOuCudaCppF32Buffer* combined_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_gradients,
    BulletOuCudaCppF32Buffer* l0w_gradients,
    BulletOuCudaCppF32Buffer* l0b_gradients,
    BulletOuCudaCppF32Buffer* l1w_gradients,
    BulletOuCudaCppF32Buffer* l1b_gradients,
    BulletOuCudaCppF32Buffer* l2w_gradients,
    BulletOuCudaCppF32Buffer* l2b_gradients,
    BulletOuCudaCppF32Buffer* outw_gradients,
    BulletOuCudaCppF32Buffer* outb_gradients,
    int zero_l0_gradients) {
    if (validate_nnue_shape(input_size, l1, l2, l3, batch, max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), batch * max_active, "stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), batch * max_active, "nstm_indices") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(combined), batch * l1 * 2, "combined") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(hidden1), batch * l2, "hidden1") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(hidden2), batch * l3, "hidden2") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(stm_l0), batch * l1, "stm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(nstm_l0), batch * l1, "nstm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), l1 * 2 * l2, "l1w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2w), l2 * l3, "l2w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(outw), l3, "outw") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(mean_output_gradients), batch, "mean_output_gradients") != 0 ||
        validate_buffer(ctx, hidden2_gradients, batch * l3, "hidden2_gradients") != 0 ||
        validate_buffer(ctx, hidden1_gradients, batch * l2, "hidden1_gradients") != 0 ||
        validate_buffer(ctx, combined_gradients, batch * l1 * 2, "combined_gradients") != 0 ||
        validate_buffer(ctx, stm_l0_gradients, batch * l1, "stm_l0_gradients") != 0 ||
        validate_buffer(ctx, nstm_l0_gradients, batch * l1, "nstm_l0_gradients") != 0 ||
        validate_buffer(ctx, l0w_gradients, nnue_l0w_len_for_shape(input_size, l1), "l0w_gradients") != 0 ||
        validate_buffer(ctx, l0b_gradients, l1, "l0b_gradients") != 0 ||
        validate_buffer(ctx, l1w_gradients, l1 * 2 * l2, "l1w_gradients") != 0 ||
        validate_buffer(ctx, l1b_gradients, l2, "l1b_gradients") != 0 ||
        validate_buffer(ctx, l2w_gradients, l2 * l3, "l2w_gradients") != 0 ||
        validate_buffer(ctx, l2b_gradients, l3, "l2b_gradients") != 0 ||
        validate_buffer(ctx, outw_gradients, l3, "outw_gradients") != 0 ||
        validate_buffer(ctx, outb_gradients, 1, "outb_gradients") != 0) {
        return -1;
    }

    if (launch_nnue_backward_kernels(
            ctx,
            input_size,
            l1,
            l2,
            l3,
            batch,
            max_active,
            stm_indices->ptr,
            nstm_indices->ptr,
            combined->ptr,
            hidden1->ptr,
            hidden2->ptr,
            stm_l0->ptr,
            nstm_l0->ptr,
            l1w->ptr,
            l2w->ptr,
            outw->ptr,
            mean_output_gradients->ptr,
            hidden2_gradients->ptr,
            hidden1_gradients->ptr,
            combined_gradients->ptr,
            stm_l0_gradients->ptr,
            nstm_l0_gradients->ptr,
            l0w_gradients->ptr,
            l0b_gradients->ptr,
            l1w_gradients->ptr,
            l1b_gradients->ptr,
            l2w_gradients->ptr,
            l2b_gradients->ptr,
            outw_gradients->ptr,
            outb_gradients->ptr,
            zero_l0_gradients) != 0) {
        return -1;
    }

    return ok();
}

extern "C" int bulletou_cuda_cpp_nnue_train_warmup_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t l1,
    size_t l2,
    size_t l3,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppF32Buffer* combined,
    const BulletOuCudaCppF32Buffer* hidden1,
    const BulletOuCudaCppF32Buffer* hidden2,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* outw,
    const BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* hidden2_gradients,
    BulletOuCudaCppF32Buffer* hidden1_gradients,
    BulletOuCudaCppF32Buffer* combined_gradients,
    BulletOuCudaCppF32Buffer* l1w_gradients,
    BulletOuCudaCppF32Buffer* l1b_gradients,
    BulletOuCudaCppF32Buffer* l2w_gradients,
    BulletOuCudaCppF32Buffer* l2b_gradients,
    BulletOuCudaCppF32Buffer* outw_gradients,
    BulletOuCudaCppF32Buffer* outb_gradients) {
    if (validate_nnue_shape(input_size, l1, l2, l3, batch, max_active) != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(combined), batch * l1 * 2, "warmup combined") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(hidden1), batch * l2, "warmup hidden1") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(hidden2), batch * l3, "warmup hidden2") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), l1 * 2 * l2, "warmup l1w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2w), l2 * l3, "warmup l2w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(outw), l3, "warmup outw") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(mean_output_gradients), batch, "warmup mean_output_gradients") != 0 ||
        validate_buffer(ctx, hidden2_gradients, batch * l3, "warmup hidden2_gradients") != 0 ||
        validate_buffer(ctx, hidden1_gradients, batch * l2, "warmup hidden1_gradients") != 0 ||
        validate_buffer(ctx, combined_gradients, batch * l1 * 2, "warmup combined_gradients") != 0 ||
        validate_buffer(ctx, l1w_gradients, l1 * 2 * l2, "warmup l1w_gradients") != 0 ||
        validate_buffer(ctx, l1b_gradients, l2, "warmup l1b_gradients") != 0 ||
        validate_buffer(ctx, l2w_gradients, l2 * l3, "warmup l2w_gradients") != 0 ||
        validate_buffer(ctx, l2b_gradients, l3, "warmup l2b_gradients") != 0 ||
        validate_buffer(ctx, outw_gradients, l3, "warmup outw_gradients") != 0 ||
        validate_buffer(ctx, outb_gradients, 1, "warmup outb_gradients") != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }

    constexpr int threads = 256;
    int blocks = 0;
    size_t out_threads = std::max(batch * l3, std::max(l3, static_cast<size_t>(1)));
    if (block_count_1d(out_threads, threads, &blocks, "warmup dense_output_backward_kernel") != 0) {
        return -1;
    }
    dense_output_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        hidden2->ptr,
        mean_output_gradients->ptr,
        outw->ptr,
        hidden2_gradients->ptr,
        outw_gradients->ptr,
        outb_gradients->ptr,
        batch,
        l3);
    if (check_kernel_launch("warmup dense_output_backward_kernel launch") != 0) {
        return -1;
    }

    if (launch_dense_crelu_backward_gemm(
            ctx,
            "warmup dense_l2_crelu_backward_gemm",
            hidden1->ptr,
            hidden2->ptr,
            hidden2_gradients->ptr,
            l2w->ptr,
            hidden1_gradients->ptr,
            l2w_gradients->ptr,
            l2b_gradients->ptr,
            batch,
            l2,
            l3) != 0) {
        return -1;
    }

    if (launch_dense_crelu_backward_gemm(
            ctx,
            "warmup dense_l1_crelu_backward_gemm",
            combined->ptr,
            hidden1->ptr,
            hidden1_gradients->ptr,
            l1w->ptr,
            combined_gradients->ptr,
            l1w_gradients->ptr,
            l1b_gradients->ptr,
            batch,
            l1 * 2,
            l2) != 0) {
        return -1;
    }

    cudaError_t status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        return fail("cudaStreamSynchronize NNUE train warmup", status);
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_sfnn_backward_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppI32Buffer* buckets,
    const BulletOuCudaCppF32Buffer* stm_l0,
    const BulletOuCudaCppF32Buffer* nstm_l0,
    const BulletOuCudaCppF32Buffer* combined,
    const BulletOuCudaCppF32Buffer* l1,
    const BulletOuCudaCppF32Buffer* l2_input,
    const BulletOuCudaCppF32Buffer* l2,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l1fw,
    int has_l1f,
    const BulletOuCudaCppF32Buffer* l1axw,
    int has_l1ax,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l2fw,
    int has_l2f,
    const BulletOuCudaCppF32Buffer* l2axw,
    int has_l2ax,
    const BulletOuCudaCppF32Buffer* l3w,
    const BulletOuCudaCppF32Buffer* l3fw,
    int has_l3f,
    const BulletOuCudaCppF32Buffer* l3axw,
    int has_l3ax,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const BulletOuCudaCppF32Buffer* factorizer_axis_confidences,
    const BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* l2_gradients,
    BulletOuCudaCppF32Buffer* l1_gradients,
    BulletOuCudaCppF32Buffer* l2_input_gradients,
    BulletOuCudaCppF32Buffer* combined_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_pre_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_pre_gradients,
    BulletOuCudaCppF32Buffer* l0w_gradients,
    BulletOuCudaCppF32Buffer* l0b_gradients,
    BulletOuCudaCppF32Buffer* l1w_gradients,
    BulletOuCudaCppF32Buffer* l1b_gradients,
    BulletOuCudaCppF32Buffer* l1fw_gradients,
    BulletOuCudaCppF32Buffer* l1fb_gradients,
    BulletOuCudaCppF32Buffer* l1axw_gradients,
    BulletOuCudaCppF32Buffer* l1axb_gradients,
    BulletOuCudaCppF32Buffer* l2w_gradients,
    BulletOuCudaCppF32Buffer* l2b_gradients,
    BulletOuCudaCppF32Buffer* l2fw_gradients,
    BulletOuCudaCppF32Buffer* l2fb_gradients,
    BulletOuCudaCppF32Buffer* l2axw_gradients,
    BulletOuCudaCppF32Buffer* l2axb_gradients,
    BulletOuCudaCppF32Buffer* l3w_gradients,
    BulletOuCudaCppF32Buffer* l3b_gradients,
    BulletOuCudaCppF32Buffer* l3fw_gradients,
    BulletOuCudaCppF32Buffer* l3fb_gradients,
    BulletOuCudaCppF32Buffer* l3axw_gradients,
    BulletOuCudaCppF32Buffer* l3axb_gradients) {
    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    const size_t l2_in = l1_hidden * 2;
    const size_t axis_count = sfnn_factorizer_axis_count(
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair);
    const size_t l1w_len =
        sfnn_l1w_len_for_shape(ft_size, l1_hidden, l1_skip, num_stacks, l1_group_count, l1_common_size, l1_shard_size);
    const bool common_shard_l1 = sfnn_is_common_shard_l1_shape(l1_common_size, l1_shard_size);
    const bool grouped_l1 = !common_shard_l1 && sfnn_is_grouped_l1_shape(l1_group_count);
    const float* factorizer_axis_confidences_ptr = nullptr;
    if (validate_sfnn_shape(
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            batch,
            max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), batch * max_active, "sfnn stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), batch * max_active, "sfnn nstm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(buckets), batch, "sfnn buckets") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(stm_l0), batch * ft_size, "sfnn stm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(nstm_l0), batch * ft_size, "sfnn nstm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(combined), batch * ft_size, "sfnn combined") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1), batch * l1_out, "sfnn l1") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2_input), batch * l2_in, "sfnn l2_input") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2), batch * l2_size, "sfnn l2") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), l1w_len, "sfnn l1w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2w), num_stacks * l2_size * l2_in, "sfnn l2w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3w), num_stacks * l2_size, "sfnn l3w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(mean_output_gradients), batch, "sfnn mean_output_gradients") != 0 ||
        validate_buffer(ctx, l2_gradients, batch * l2_size, "sfnn l2_gradients") != 0 ||
        validate_buffer(ctx, l1_gradients, batch * l1_out, "sfnn l1_gradients") != 0 ||
        validate_buffer(ctx, l2_input_gradients, batch * l2_in, "sfnn l2_input_gradients") != 0 ||
        validate_buffer(ctx, combined_gradients, batch * ft_size, "sfnn combined_gradients") != 0 ||
        validate_buffer(ctx, stm_l0_gradients, batch * ft_size, "sfnn stm_l0_gradients") != 0 ||
        validate_buffer(ctx, nstm_l0_gradients, batch * ft_size, "sfnn nstm_l0_gradients") != 0 ||
        validate_buffer(ctx, stm_l0_pre_gradients, batch * ft_size, "sfnn stm_l0_pre_gradients") != 0 ||
        validate_buffer(ctx, nstm_l0_pre_gradients, batch * ft_size, "sfnn nstm_l0_pre_gradients") != 0 ||
        validate_buffer(ctx, l0w_gradients, input_size * ft_size, "sfnn l0w_gradients") != 0 ||
        validate_buffer(ctx, l0b_gradients, ft_size, "sfnn l0b_gradients") != 0 ||
        validate_buffer(ctx, l1w_gradients, l1w_len, "sfnn l1w_gradients") != 0 ||
        validate_buffer(ctx, l1b_gradients, num_stacks * l1_out, "sfnn l1b_gradients") != 0 ||
        validate_buffer(ctx, l1fw_gradients, ft_size * l1_out, "sfnn l1fw_gradients") != 0 ||
        validate_buffer(ctx, l1fb_gradients, l1_out, "sfnn l1fb_gradients") != 0 ||
        validate_buffer(ctx, l1axw_gradients, axis_count * ft_size * l1_out, "sfnn l1axw_gradients") != 0 ||
        validate_buffer(ctx, l1axb_gradients, axis_count * l1_out, "sfnn l1axb_gradients") != 0 ||
        validate_buffer(ctx, l2w_gradients, num_stacks * l2_size * l2_in, "sfnn l2w_gradients") != 0 ||
        validate_buffer(ctx, l2b_gradients, num_stacks * l2_size, "sfnn l2b_gradients") != 0 ||
        validate_buffer(ctx, l2fw_gradients, l2_size * l2_in, "sfnn l2fw_gradients") != 0 ||
        validate_buffer(ctx, l2fb_gradients, l2_size, "sfnn l2fb_gradients") != 0 ||
        validate_buffer(ctx, l2axw_gradients, axis_count * l2_size * l2_in, "sfnn l2axw_gradients") != 0 ||
        validate_buffer(ctx, l2axb_gradients, axis_count * l2_size, "sfnn l2axb_gradients") != 0 ||
        validate_buffer(ctx, l3w_gradients, num_stacks * l2_size, "sfnn l3w_gradients") != 0 ||
        validate_buffer(ctx, l3b_gradients, num_stacks, "sfnn l3b_gradients") != 0 ||
        validate_buffer(ctx, l3fw_gradients, l2_size, "sfnn l3fw_gradients") != 0 ||
        validate_buffer(ctx, l3fb_gradients, 1, "sfnn l3fb_gradients") != 0 ||
        validate_buffer(ctx, l3axw_gradients, axis_count * l2_size, "sfnn l3axw_gradients") != 0 ||
        validate_buffer(ctx, l3axb_gradients, axis_count, "sfnn l3axb_gradients") != 0) {
        return -1;
    }
    if (has_l1f != 0) {
        if (grouped_l1 || common_shard_l1) {
            return fail_message("sfnn compact L1 does not support l1fw");
        }
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1fw), ft_size * l1_out, "sfnn l1fw") != 0) {
            return -1;
        }
    } else if (l1fw != nullptr) {
        return fail_message("sfnn l1fw must be null when has_l1f is false");
    }
    if (has_l1ax != 0) {
        if (grouped_l1 || common_shard_l1) {
            return fail_message("sfnn compact L1 does not support l1axw");
        }
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1axw), axis_count * ft_size * l1_out, "sfnn l1axw") != 0) {
            return -1;
        }
    } else if (l1axw != nullptr) {
        return fail_message("sfnn l1axw must be null when has_l1ax is false");
    }
    if (has_l2f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2fw), l2_size * l2_in, "sfnn l2fw") != 0) {
            return -1;
        }
    } else if (l2fw != nullptr) {
        return fail_message("sfnn l2fw must be null when has_l2f is false");
    }
    if (has_l2ax != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2axw), axis_count * l2_size * l2_in, "sfnn l2axw") != 0) {
            return -1;
        }
    } else if (l2axw != nullptr) {
        return fail_message("sfnn l2axw must be null when has_l2ax is false");
    }
    if (has_l3f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3fw), l2_size, "sfnn l3fw") != 0) {
            return -1;
        }
    } else if (l3fw != nullptr) {
        return fail_message("sfnn l3fw must be null when has_l3f is false");
    }
    if (has_l3ax != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3axw), axis_count * l2_size, "sfnn l3axw") != 0) {
            return -1;
        }
    } else if (l3axw != nullptr) {
        return fail_message("sfnn l3axw must be null when has_l3ax is false");
    }
    if (factorizer_axis_confidences != nullptr) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(factorizer_axis_confidences), axis_count, "sfnn factorizer axis confidences") != 0) {
            return -1;
        }
        factorizer_axis_confidences_ptr = factorizer_axis_confidences->ptr;
    }

    if (launch_sfnn_backward_kernels(
            ctx,
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            batch,
            max_active,
            stm_indices->ptr,
            nstm_indices->ptr,
            buckets->ptr,
            stm_l0->ptr,
            nstm_l0->ptr,
            combined->ptr,
            l1->ptr,
            l2_input->ptr,
            l2->ptr,
            l1w->ptr,
            has_l1f != 0 ? l1fw->ptr : nullptr,
            has_l1f,
            has_l1ax != 0 ? l1axw->ptr : nullptr,
            has_l1ax,
            l2w->ptr,
            has_l2f != 0 ? l2fw->ptr : nullptr,
            has_l2f,
            has_l2ax != 0 ? l2axw->ptr : nullptr,
            has_l2ax,
            l3w->ptr,
            has_l3f != 0 ? l3fw->ptr : nullptr,
            has_l3f,
            has_l3ax != 0 ? l3axw->ptr : nullptr,
            has_l3ax,
            use_king_axis,
            use_hand_axis,
            use_progress_axis,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            factorizer_shared_alpha,
            factorizer_king_axis_alpha,
            factorizer_hand_axis_alpha,
            factorizer_progress_axis_alpha,
            factorizer_pair_alpha,
            factorizer_axis_confidences_ptr,
            mean_output_gradients->ptr,
            l2_gradients->ptr,
            l1_gradients->ptr,
            l2_input_gradients->ptr,
            combined_gradients->ptr,
            stm_l0_gradients->ptr,
            nstm_l0_gradients->ptr,
            stm_l0_pre_gradients->ptr,
            nstm_l0_pre_gradients->ptr,
            l0w_gradients->ptr,
            l0b_gradients->ptr,
            l1w_gradients->ptr,
            l1b_gradients->ptr,
            l1fw_gradients->ptr,
            l1fb_gradients->ptr,
            l1axw_gradients->ptr,
            l1axb_gradients->ptr,
            l2w_gradients->ptr,
            l2b_gradients->ptr,
            l2fw_gradients->ptr,
            l2fb_gradients->ptr,
            l2axw_gradients->ptr,
            l2axb_gradients->ptr,
            l3w_gradients->ptr,
            l3b_gradients->ptr,
            l3fw_gradients->ptr,
            l3fb_gradients->ptr,
            l3axw_gradients->ptr,
            l3axb_gradients->ptr,
            1,
            0,
            nullptr,
            0) != 0) {
        return -1;
    }

    return ok();
}

int sfnn_backward_train_device_impl(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppI32Buffer* buckets,
    const BulletOuCudaCppF32Buffer* stm_l0,
    const BulletOuCudaCppF32Buffer* nstm_l0,
    const BulletOuCudaCppF32Buffer* combined,
    const BulletOuCudaCppF32Buffer* l1,
    const BulletOuCudaCppF32Buffer* l2_input,
    const BulletOuCudaCppF32Buffer* l2,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l1fw,
    int has_l1f,
    const BulletOuCudaCppF32Buffer* l1axw,
    int has_l1ax,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l2fw,
    int has_l2f,
    const BulletOuCudaCppF32Buffer* l2axw,
    int has_l2ax,
    const BulletOuCudaCppF32Buffer* l3w,
    const BulletOuCudaCppF32Buffer* l3fw,
    int has_l3f,
    const BulletOuCudaCppF32Buffer* l3axw,
    int has_l3ax,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const float* factorizer_axis_confidences,
    const BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* l2_gradients,
    BulletOuCudaCppF32Buffer* l1_gradients,
    BulletOuCudaCppF32Buffer* l2_input_gradients,
    BulletOuCudaCppF32Buffer* combined_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_pre_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_pre_gradients,
    BulletOuCudaCppF32Buffer* l0w_gradients,
    BulletOuCudaCppF32Buffer* l0b_gradients,
    BulletOuCudaCppF32Buffer* l1w_gradients,
    BulletOuCudaCppF32Buffer* l1b_gradients,
    BulletOuCudaCppF32Buffer* l1fw_gradients,
    BulletOuCudaCppF32Buffer* l1fb_gradients,
    BulletOuCudaCppF32Buffer* l1axw_gradients,
    BulletOuCudaCppF32Buffer* l1axb_gradients,
    BulletOuCudaCppF32Buffer* l2w_gradients,
    BulletOuCudaCppF32Buffer* l2b_gradients,
    BulletOuCudaCppF32Buffer* l2fw_gradients,
    BulletOuCudaCppF32Buffer* l2fb_gradients,
    BulletOuCudaCppF32Buffer* l2axw_gradients,
    BulletOuCudaCppF32Buffer* l2axb_gradients,
    BulletOuCudaCppF32Buffer* l3w_gradients,
    BulletOuCudaCppF32Buffer* l3b_gradients,
    BulletOuCudaCppF32Buffer* l3fw_gradients,
    BulletOuCudaCppF32Buffer* l3fb_gradients,
    BulletOuCudaCppF32Buffer* l3axw_gradients,
    BulletOuCudaCppF32Buffer* l3axb_gradients,
    int zero_parameter_gradients,
    float* profile_ms,
    size_t profile_ms_len) {
    const size_t l1_out = sfnn_l1_out_for_shape(l1_hidden, l1_skip);
    const size_t l2_in = l1_hidden * 2;
    const size_t axis_count = sfnn_factorizer_axis_count(
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair);
    const size_t l1w_len =
        sfnn_l1w_len_for_shape(ft_size, l1_hidden, l1_skip, num_stacks, l1_group_count, l1_common_size, l1_shard_size);
    const bool common_shard_l1 = sfnn_is_common_shard_l1_shape(l1_common_size, l1_shard_size);
    const bool grouped_l1 = !common_shard_l1 && sfnn_is_grouped_l1_shape(l1_group_count);
    if (validate_sfnn_shape(
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            batch,
            max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), batch * max_active, "sfnn stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), batch * max_active, "sfnn nstm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(buckets), batch, "sfnn buckets") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(stm_l0), batch * ft_size, "sfnn stm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(nstm_l0), batch * ft_size, "sfnn nstm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(combined), batch * ft_size, "sfnn combined") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1), batch * l1_out, "sfnn l1") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2_input), batch * l2_in, "sfnn l2_input") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2), batch * l2_size, "sfnn l2") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), l1w_len, "sfnn l1w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2w), num_stacks * l2_size * l2_in, "sfnn l2w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3w), num_stacks * l2_size, "sfnn l3w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(mean_output_gradients), batch, "sfnn mean_output_gradients") != 0 ||
        validate_buffer(ctx, l2_gradients, batch * l2_size, "sfnn l2_gradients") != 0 ||
        validate_buffer(ctx, l1_gradients, batch * l1_out, "sfnn l1_gradients") != 0 ||
        validate_buffer(ctx, l2_input_gradients, batch * l2_in, "sfnn l2_input_gradients") != 0 ||
        validate_buffer(ctx, combined_gradients, batch * ft_size, "sfnn combined_gradients") != 0 ||
        validate_buffer(ctx, stm_l0_gradients, batch * ft_size, "sfnn stm_l0_gradients") != 0 ||
        validate_buffer(ctx, nstm_l0_gradients, batch * ft_size, "sfnn nstm_l0_gradients") != 0 ||
        validate_buffer(ctx, stm_l0_pre_gradients, batch * ft_size, "sfnn stm_l0_pre_gradients") != 0 ||
        validate_buffer(ctx, nstm_l0_pre_gradients, batch * ft_size, "sfnn nstm_l0_pre_gradients") != 0 ||
        validate_buffer(ctx, l0w_gradients, input_size * ft_size, "sfnn l0w_gradients") != 0 ||
        validate_buffer(ctx, l0b_gradients, ft_size, "sfnn l0b_gradients") != 0 ||
        validate_buffer(ctx, l1w_gradients, l1w_len, "sfnn l1w_gradients") != 0 ||
        validate_buffer(ctx, l1b_gradients, num_stacks * l1_out, "sfnn l1b_gradients") != 0 ||
        validate_buffer(ctx, l1fw_gradients, ft_size * l1_out, "sfnn l1fw_gradients") != 0 ||
        validate_buffer(ctx, l1fb_gradients, l1_out, "sfnn l1fb_gradients") != 0 ||
        validate_buffer(ctx, l1axw_gradients, axis_count * ft_size * l1_out, "sfnn l1axw_gradients") != 0 ||
        validate_buffer(ctx, l1axb_gradients, axis_count * l1_out, "sfnn l1axb_gradients") != 0 ||
        validate_buffer(ctx, l2w_gradients, num_stacks * l2_size * l2_in, "sfnn l2w_gradients") != 0 ||
        validate_buffer(ctx, l2b_gradients, num_stacks * l2_size, "sfnn l2b_gradients") != 0 ||
        validate_buffer(ctx, l2fw_gradients, l2_size * l2_in, "sfnn l2fw_gradients") != 0 ||
        validate_buffer(ctx, l2fb_gradients, l2_size, "sfnn l2fb_gradients") != 0 ||
        validate_buffer(ctx, l2axw_gradients, axis_count * l2_size * l2_in, "sfnn l2axw_gradients") != 0 ||
        validate_buffer(ctx, l2axb_gradients, axis_count * l2_size, "sfnn l2axb_gradients") != 0 ||
        validate_buffer(ctx, l3w_gradients, num_stacks * l2_size, "sfnn l3w_gradients") != 0 ||
        validate_buffer(ctx, l3b_gradients, num_stacks, "sfnn l3b_gradients") != 0 ||
        validate_buffer(ctx, l3fw_gradients, l2_size, "sfnn l3fw_gradients") != 0 ||
        validate_buffer(ctx, l3fb_gradients, 1, "sfnn l3fb_gradients") != 0 ||
        validate_buffer(ctx, l3axw_gradients, axis_count * l2_size, "sfnn l3axw_gradients") != 0 ||
        validate_buffer(ctx, l3axb_gradients, axis_count, "sfnn l3axb_gradients") != 0) {
        return -1;
    }
    if (has_l1f != 0) {
        if (grouped_l1 || common_shard_l1) {
            return fail_message("sfnn compact L1 does not support l1fw");
        }
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1fw), ft_size * l1_out, "sfnn l1fw") != 0) {
            return -1;
        }
    } else if (l1fw != nullptr) {
        return fail_message("sfnn l1fw must be null when has_l1f is false");
    }
    if (has_l1ax != 0) {
        if (grouped_l1 || common_shard_l1) {
            return fail_message("sfnn compact L1 does not support l1axw");
        }
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1axw), axis_count * ft_size * l1_out, "sfnn l1axw") != 0) {
            return -1;
        }
    } else if (l1axw != nullptr) {
        return fail_message("sfnn l1axw must be null when has_l1ax is false");
    }
    if (has_l2f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2fw), l2_size * l2_in, "sfnn l2fw") != 0) {
            return -1;
        }
    } else if (l2fw != nullptr) {
        return fail_message("sfnn l2fw must be null when has_l2f is false");
    }
    if (has_l2ax != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2axw), axis_count * l2_size * l2_in, "sfnn l2axw") != 0) {
            return -1;
        }
    } else if (l2axw != nullptr) {
        return fail_message("sfnn l2axw must be null when has_l2ax is false");
    }
    if (has_l3f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3fw), l2_size, "sfnn l3fw") != 0) {
            return -1;
        }
    } else if (l3fw != nullptr) {
        return fail_message("sfnn l3fw must be null when has_l3f is false");
    }
    if (has_l3ax != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l3axw), axis_count * l2_size, "sfnn l3axw") != 0) {
            return -1;
        }
    } else if (l3axw != nullptr) {
        return fail_message("sfnn l3axw must be null when has_l3ax is false");
    }

    if (launch_sfnn_backward_kernels(
            ctx,
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_group_count,
            l1_common_size,
            l1_shard_size,
            factorizer_king_axis_dim,
            factorizer_hand_axis_dim,
            batch,
            max_active,
            stm_indices->ptr,
            nstm_indices->ptr,
            buckets->ptr,
            stm_l0->ptr,
            nstm_l0->ptr,
            combined->ptr,
            l1->ptr,
            l2_input->ptr,
            l2->ptr,
            l1w->ptr,
            has_l1f != 0 ? l1fw->ptr : nullptr,
            has_l1f,
            has_l1ax != 0 ? l1axw->ptr : nullptr,
            has_l1ax,
            l2w->ptr,
            has_l2f != 0 ? l2fw->ptr : nullptr,
            has_l2f,
            has_l2ax != 0 ? l2axw->ptr : nullptr,
            has_l2ax,
            l3w->ptr,
            has_l3f != 0 ? l3fw->ptr : nullptr,
            has_l3f,
            has_l3ax != 0 ? l3axw->ptr : nullptr,
            has_l3ax,
            use_king_axis,
            use_hand_axis,
            use_progress_axis,
            use_king_hand_pair,
            use_king_progress_pair,
            use_hand_progress_pair,
            factorizer_shared_alpha,
            factorizer_king_axis_alpha,
            factorizer_hand_axis_alpha,
            factorizer_progress_axis_alpha,
            factorizer_pair_alpha,
            factorizer_axis_confidences,
            mean_output_gradients->ptr,
            l2_gradients->ptr,
            l1_gradients->ptr,
            l2_input_gradients->ptr,
            combined_gradients->ptr,
            stm_l0_gradients->ptr,
            nstm_l0_gradients->ptr,
            stm_l0_pre_gradients->ptr,
            nstm_l0_pre_gradients->ptr,
            l0w_gradients->ptr,
            l0b_gradients->ptr,
            l1w_gradients->ptr,
            l1b_gradients->ptr,
            l1fw_gradients->ptr,
            l1fb_gradients->ptr,
            l1axw_gradients->ptr,
            l1axb_gradients->ptr,
            l2w_gradients->ptr,
            l2b_gradients->ptr,
            l2fw_gradients->ptr,
            l2fb_gradients->ptr,
            l2axw_gradients->ptr,
            l2axb_gradients->ptr,
            l3w_gradients->ptr,
            l3b_gradients->ptr,
            l3fw_gradients->ptr,
            l3fb_gradients->ptr,
            l3axw_gradients->ptr,
            l3axb_gradients->ptr,
            zero_parameter_gradients,
            1,
            profile_ms,
            profile_ms_len) != 0) {
        return -1;
    }

    return ok();
}

extern "C" int bulletou_cuda_cpp_sfnn_backward_train_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppI32Buffer* buckets,
    const BulletOuCudaCppF32Buffer* stm_l0,
    const BulletOuCudaCppF32Buffer* nstm_l0,
    const BulletOuCudaCppF32Buffer* combined,
    const BulletOuCudaCppF32Buffer* l1,
    const BulletOuCudaCppF32Buffer* l2_input,
    const BulletOuCudaCppF32Buffer* l2,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l1fw,
    int has_l1f,
    const BulletOuCudaCppF32Buffer* l1axw,
    int has_l1ax,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l2fw,
    int has_l2f,
    const BulletOuCudaCppF32Buffer* l2axw,
    int has_l2ax,
    const BulletOuCudaCppF32Buffer* l3w,
    const BulletOuCudaCppF32Buffer* l3fw,
    int has_l3f,
    const BulletOuCudaCppF32Buffer* l3axw,
    int has_l3ax,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const BulletOuCudaCppF32Buffer* factorizer_axis_confidences,
    const BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* l2_gradients,
    BulletOuCudaCppF32Buffer* l1_gradients,
    BulletOuCudaCppF32Buffer* l2_input_gradients,
    BulletOuCudaCppF32Buffer* combined_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_pre_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_pre_gradients,
    BulletOuCudaCppF32Buffer* l0w_gradients,
    BulletOuCudaCppF32Buffer* l0b_gradients,
    BulletOuCudaCppF32Buffer* l1w_gradients,
    BulletOuCudaCppF32Buffer* l1b_gradients,
    BulletOuCudaCppF32Buffer* l1fw_gradients,
    BulletOuCudaCppF32Buffer* l1fb_gradients,
    BulletOuCudaCppF32Buffer* l1axw_gradients,
    BulletOuCudaCppF32Buffer* l1axb_gradients,
    BulletOuCudaCppF32Buffer* l2w_gradients,
    BulletOuCudaCppF32Buffer* l2b_gradients,
    BulletOuCudaCppF32Buffer* l2fw_gradients,
    BulletOuCudaCppF32Buffer* l2fb_gradients,
    BulletOuCudaCppF32Buffer* l2axw_gradients,
    BulletOuCudaCppF32Buffer* l2axb_gradients,
    BulletOuCudaCppF32Buffer* l3w_gradients,
    BulletOuCudaCppF32Buffer* l3b_gradients,
    BulletOuCudaCppF32Buffer* l3fw_gradients,
    BulletOuCudaCppF32Buffer* l3fb_gradients,
    BulletOuCudaCppF32Buffer* l3axw_gradients,
    BulletOuCudaCppF32Buffer* l3axb_gradients,
    int zero_parameter_gradients) {
    const size_t axis_count = sfnn_factorizer_axis_count(
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair);
    const float* factorizer_axis_confidences_ptr = nullptr;
    if (factorizer_axis_confidences != nullptr) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(factorizer_axis_confidences), axis_count, "sfnn train factorizer axis confidences") != 0) {
            return -1;
        }
        factorizer_axis_confidences_ptr = factorizer_axis_confidences->ptr;
    }
    return sfnn_backward_train_device_impl(
        ctx,
        input_size,
        ft_size,
        l1_hidden,
        l1_skip,
        l2_size,
        num_stacks,
        l1_group_count,
        l1_common_size,
        l1_shard_size,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        batch,
        max_active,
        stm_indices,
        nstm_indices,
        buckets,
        stm_l0,
        nstm_l0,
        combined,
        l1,
        l2_input,
        l2,
        l1w,
        l1fw,
        has_l1f,
        l1axw,
        has_l1ax,
        l2w,
        l2fw,
        has_l2f,
        l2axw,
        has_l2ax,
        l3w,
        l3fw,
        has_l3f,
        l3axw,
        has_l3ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        mean_output_gradients,
        l2_gradients,
        l1_gradients,
        l2_input_gradients,
        combined_gradients,
        stm_l0_gradients,
        nstm_l0_gradients,
        stm_l0_pre_gradients,
        nstm_l0_pre_gradients,
        l0w_gradients,
        l0b_gradients,
        l1w_gradients,
        l1b_gradients,
        l1fw_gradients,
        l1fb_gradients,
        l1axw_gradients,
        l1axb_gradients,
        l2w_gradients,
        l2b_gradients,
        l2fw_gradients,
        l2fb_gradients,
        l2axw_gradients,
        l2axb_gradients,
        l3w_gradients,
        l3b_gradients,
        l3fw_gradients,
        l3fb_gradients,
        l3axw_gradients,
        l3axb_gradients,
        zero_parameter_gradients,
        nullptr,
        0);
}

extern "C" int bulletou_cuda_cpp_sfnn_backward_train_profile_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    int l1_skip,
    size_t l2_size,
    size_t num_stacks,
    size_t l1_group_count,
    size_t l1_common_size,
    size_t l1_shard_size,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppI32Buffer* buckets,
    const BulletOuCudaCppF32Buffer* stm_l0,
    const BulletOuCudaCppF32Buffer* nstm_l0,
    const BulletOuCudaCppF32Buffer* combined,
    const BulletOuCudaCppF32Buffer* l1,
    const BulletOuCudaCppF32Buffer* l2_input,
    const BulletOuCudaCppF32Buffer* l2,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l1fw,
    int has_l1f,
    const BulletOuCudaCppF32Buffer* l1axw,
    int has_l1ax,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l2fw,
    int has_l2f,
    const BulletOuCudaCppF32Buffer* l2axw,
    int has_l2ax,
    const BulletOuCudaCppF32Buffer* l3w,
    const BulletOuCudaCppF32Buffer* l3fw,
    int has_l3f,
    const BulletOuCudaCppF32Buffer* l3axw,
    int has_l3ax,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const BulletOuCudaCppF32Buffer* factorizer_axis_confidences,
    const BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* l2_gradients,
    BulletOuCudaCppF32Buffer* l1_gradients,
    BulletOuCudaCppF32Buffer* l2_input_gradients,
    BulletOuCudaCppF32Buffer* combined_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_gradients,
    BulletOuCudaCppF32Buffer* stm_l0_pre_gradients,
    BulletOuCudaCppF32Buffer* nstm_l0_pre_gradients,
    BulletOuCudaCppF32Buffer* l0w_gradients,
    BulletOuCudaCppF32Buffer* l0b_gradients,
    BulletOuCudaCppF32Buffer* l1w_gradients,
    BulletOuCudaCppF32Buffer* l1b_gradients,
    BulletOuCudaCppF32Buffer* l1fw_gradients,
    BulletOuCudaCppF32Buffer* l1fb_gradients,
    BulletOuCudaCppF32Buffer* l1axw_gradients,
    BulletOuCudaCppF32Buffer* l1axb_gradients,
    BulletOuCudaCppF32Buffer* l2w_gradients,
    BulletOuCudaCppF32Buffer* l2b_gradients,
    BulletOuCudaCppF32Buffer* l2fw_gradients,
    BulletOuCudaCppF32Buffer* l2fb_gradients,
    BulletOuCudaCppF32Buffer* l2axw_gradients,
    BulletOuCudaCppF32Buffer* l2axb_gradients,
    BulletOuCudaCppF32Buffer* l3w_gradients,
    BulletOuCudaCppF32Buffer* l3b_gradients,
    BulletOuCudaCppF32Buffer* l3fw_gradients,
    BulletOuCudaCppF32Buffer* l3fb_gradients,
    BulletOuCudaCppF32Buffer* l3axw_gradients,
    BulletOuCudaCppF32Buffer* l3axb_gradients,
    int zero_parameter_gradients,
    float* profile_ms,
    size_t profile_ms_len) {
    const size_t axis_count = sfnn_factorizer_axis_count(
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair);
    const float* factorizer_axis_confidences_ptr = nullptr;
    if (factorizer_axis_confidences != nullptr) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(factorizer_axis_confidences), axis_count, "sfnn train factorizer axis confidences") != 0) {
            return -1;
        }
        factorizer_axis_confidences_ptr = factorizer_axis_confidences->ptr;
    }
    return sfnn_backward_train_device_impl(
        ctx,
        input_size,
        ft_size,
        l1_hidden,
        l1_skip,
        l2_size,
        num_stacks,
        l1_group_count,
        l1_common_size,
        l1_shard_size,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        batch,
        max_active,
        stm_indices,
        nstm_indices,
        buckets,
        stm_l0,
        nstm_l0,
        combined,
        l1,
        l2_input,
        l2,
        l1w,
        l1fw,
        has_l1f,
        l1axw,
        has_l1ax,
        l2w,
        l2fw,
        has_l2f,
        l2axw,
        has_l2ax,
        l3w,
        l3fw,
        has_l3f,
        l3axw,
        has_l3ax,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        mean_output_gradients,
        l2_gradients,
        l1_gradients,
        l2_input_gradients,
        combined_gradients,
        stm_l0_gradients,
        nstm_l0_gradients,
        stm_l0_pre_gradients,
        nstm_l0_pre_gradients,
        l0w_gradients,
        l0b_gradients,
        l1w_gradients,
        l1b_gradients,
        l1fw_gradients,
        l1fb_gradients,
        l1axw_gradients,
        l1axb_gradients,
        l2w_gradients,
        l2b_gradients,
        l2fw_gradients,
        l2fb_gradients,
        l2axw_gradients,
        l2axb_gradients,
        l3w_gradients,
        l3b_gradients,
        l3fw_gradients,
        l3fb_gradients,
        l3axw_gradients,
        l3axb_gradients,
        zero_parameter_gradients,
        profile_ms,
        profile_ms_len);
}

extern "C" int bulletou_cuda_cpp_sfnn_add_saturation_penalty_gradients_device(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* weights,
    BulletOuCudaCppF32Buffer* shared_weights,
    int has_shared,
    BulletOuCudaCppF32Buffer* axis_weights,
    int has_axis,
    BulletOuCudaCppF32Buffer* gradients,
    BulletOuCudaCppF32Buffer* shared_gradients,
    BulletOuCudaCppF32Buffer* axis_gradients,
    BulletOuCudaCppI32Buffer* dirty_buckets,
    size_t dirty_count,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    size_t factorizer_king_axis_dim,
    size_t factorizer_hand_axis_dim,
    int use_king_axis,
    int use_hand_axis,
    int use_progress_axis,
    int use_king_hand_pair,
    int use_king_progress_pair,
    int use_hand_progress_pair,
    float factorizer_shared_alpha,
    float factorizer_king_axis_alpha,
    float factorizer_hand_axis_alpha,
    float factorizer_progress_axis_alpha,
    float factorizer_pair_alpha,
    const BulletOuCudaCppF32Buffer* factorizer_axis_confidences,
    int shared_weight_input_major,
    float weight_scale,
    float threshold,
    float penalty) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (input_dim == 0 || output_dim == 0 || num_stacks == 0) {
        return fail_message("sfnn saturation penalty dimensions must be non-zero");
    }
    if (input_dim > SIZE_MAX / output_dim) {
        return fail_message("sfnn saturation penalty weight cell overflow");
    }
    const size_t weight_cell_count = input_dim * output_dim;
    if (num_stacks > SIZE_MAX / weight_cell_count) {
        return fail_message("sfnn saturation penalty weight length overflow");
    }
    const size_t weight_total = num_stacks * weight_cell_count;
    const size_t axis_count = sfnn_factorizer_axis_count(
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair);
    if (validate_buffer(ctx, weights, weight_total, "sfnn saturation weights") != 0 ||
        validate_buffer(ctx, gradients, weight_total, "sfnn saturation gradients") != 0) {
        return -1;
    }
    if (has_shared != 0) {
        if (validate_buffer(ctx, shared_weights, weight_cell_count, "sfnn saturation shared weights") != 0 ||
            validate_buffer(ctx, shared_gradients, weight_cell_count, "sfnn saturation shared gradients") != 0) {
            return -1;
        }
    }
    if (has_axis != 0) {
        if (axis_count == 0 || axis_count > SIZE_MAX / weight_cell_count) {
            return fail_message("sfnn saturation axis weight length overflow");
        }
        const size_t axis_weight_total = axis_count * weight_cell_count;
        if (validate_buffer(ctx, axis_weights, axis_weight_total, "sfnn saturation axis weights") != 0 ||
            validate_buffer(ctx, axis_gradients, axis_weight_total, "sfnn saturation axis gradients") != 0) {
            return -1;
        }
    }
    const float* factorizer_axis_confidences_ptr = nullptr;
    if (factorizer_axis_confidences != nullptr) {
        if (axis_count == 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(factorizer_axis_confidences), axis_count, "sfnn saturation factorizer axis confidences") != 0) {
            return -1;
        }
        factorizer_axis_confidences_ptr = factorizer_axis_confidences->ptr;
    }
    if (dirty_count != 0 &&
        validate_i32_buffer(ctx, dirty_buckets, dirty_count, "sfnn saturation dirty_buckets") != 0) {
        return -1;
    }

    return launch_sfnn_add_saturation_penalty_gradients(
        ctx,
        weights->ptr,
        has_shared != 0 ? shared_weights->ptr : nullptr,
        has_axis != 0 ? axis_weights->ptr : nullptr,
        gradients->ptr,
        has_shared != 0 ? shared_gradients->ptr : nullptr,
        has_axis != 0 ? axis_gradients->ptr : nullptr,
        dirty_count != 0 ? dirty_buckets->ptr : nullptr,
        dirty_count,
        input_dim,
        output_dim,
        num_stacks,
        factorizer_king_axis_dim,
        factorizer_hand_axis_dim,
        has_shared,
        shared_weight_input_major,
        has_axis,
        use_king_axis,
        use_hand_axis,
        use_progress_axis,
        use_king_hand_pair,
        use_king_progress_pair,
        use_hand_progress_pair,
        factorizer_shared_alpha,
        factorizer_king_axis_alpha,
        factorizer_hand_axis_alpha,
        factorizer_progress_axis_alpha,
        factorizer_pair_alpha,
        factorizer_axis_confidences_ptr,
        weight_scale,
        threshold,
        penalty,
        "sfnn saturation penalty gradients");
}

extern "C" int bulletou_cuda_cpp_sfnn_add_residual_count_decay_gradients_device(
    BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* weights,
    BulletOuCudaCppF32Buffer* gradients,
    BulletOuCudaCppF32Buffer* decay_by_stack,
    BulletOuCudaCppI32Buffer* dirty_buckets,
    size_t dirty_count,
    size_t stride,
    size_t num_stacks) {
    if (set_context_device(ctx) != 0) {
        return -1;
    }
    if (stride == 0 || num_stacks == 0) {
        return fail_message("sfnn residual count decay dimensions must be non-zero");
    }
    if (num_stacks > SIZE_MAX / stride) {
        return fail_message("sfnn residual count decay weight length overflow");
    }
    const size_t total = num_stacks * stride;
    if (validate_buffer(ctx, weights, total, "sfnn residual count decay weights") != 0 ||
        validate_buffer(ctx, gradients, total, "sfnn residual count decay gradients") != 0 ||
        validate_buffer(ctx, decay_by_stack, num_stacks, "sfnn residual count decay by-stack coefficients") != 0) {
        return -1;
    }
    if (dirty_count != 0 &&
        validate_i32_buffer(ctx, dirty_buckets, dirty_count, "sfnn residual count decay dirty_buckets") != 0) {
        return -1;
    }
    return launch_sfnn_add_residual_count_decay_gradients(
        ctx,
        weights->ptr,
        gradients->ptr,
        decay_by_stack->ptr,
        dirty_count != 0 ? dirty_buckets->ptr : nullptr,
        dirty_count,
        stride,
        num_stacks,
        "sfnn residual count decay gradients");
}

extern "C" int bulletou_cuda_cpp_axpy_host(
    int device,
    size_t len,
    float a,
    const float* x,
    const float* y,
    float* out) {
    if (validate_host_ptr(x, len, "x") != 0 || validate_host_ptr(y, len, "y") != 0 ||
        validate_host_ptr(out, len, "out") != 0) {
        return -1;
    }

    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        return fail("cudaSetDevice", status);
    }

    DeviceFloats dx;
    DeviceFloats dy;
    DeviceFloats dout;
    if (dx.allocate(len, "cudaMalloc x") != 0 || dy.allocate(len, "cudaMalloc y") != 0 ||
        dout.allocate(len, "cudaMalloc out") != 0) {
        return -1;
    }
    if (copy_h2d(dx.ptr, x, len, "cudaMemcpy x H2D") != 0 || copy_h2d(dy.ptr, y, len, "cudaMemcpy y H2D") != 0) {
        return -1;
    }

    if (len != 0) {
        int threads = 256;
        int blocks = static_cast<int>((len + threads - 1) / threads);
        axpy_kernel<<<blocks, threads>>>(len, a, dx.ptr, dy.ptr, dout.ptr);
        if (sync_after_kernel("axpy_kernel launch", "axpy_kernel sync") != 0) {
            return -1;
        }
    }

    if (copy_d2h(out, dout.ptr, len, "cudaMemcpy out D2H") != 0) {
        return -1;
    }
    return ok();
}

extern "C" int bulletou_cuda_cpp_ranger_update_host(
    int device,
    size_t len,
    float gradient_factor,
    float learning_rate,
    float step_size,
    int use_denom,
    float decay,
    float beta1,
    float beta2,
    float epsilon,
    float min_weight,
    float max_weight,
    int do_lookahead,
    float lookahead_alpha,
    float* gradients,
    float* weights,
    float* momentum,
    float* velocity,
    float* slow_params) {
    if (validate_host_ptr(gradients, len, "gradients") != 0 || validate_host_ptr(weights, len, "weights") != 0 ||
        validate_host_ptr(momentum, len, "momentum") != 0 || validate_host_ptr(velocity, len, "velocity") != 0 ||
        validate_host_ptr(slow_params, len, "slow_params") != 0) {
        return -1;
    }

    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        return fail("cudaSetDevice", status);
    }

    DeviceFloats dgrad;
    DeviceFloats dw;
    DeviceFloats dm;
    DeviceFloats dv;
    DeviceFloats dslow;
    if (dgrad.allocate(len, "cudaMalloc gradients") != 0 || dw.allocate(len, "cudaMalloc weights") != 0 ||
        dm.allocate(len, "cudaMalloc momentum") != 0 || dv.allocate(len, "cudaMalloc velocity") != 0 ||
        dslow.allocate(len, "cudaMalloc slow_params") != 0) {
        return -1;
    }

    if (copy_h2d(dgrad.ptr, gradients, len, "cudaMemcpy gradients H2D") != 0 ||
        copy_h2d(dw.ptr, weights, len, "cudaMemcpy weights H2D") != 0 ||
        copy_h2d(dm.ptr, momentum, len, "cudaMemcpy momentum H2D") != 0 ||
        copy_h2d(dv.ptr, velocity, len, "cudaMemcpy velocity H2D") != 0 ||
        copy_h2d(dslow.ptr, slow_params, len, "cudaMemcpy slow_params H2D") != 0) {
        return -1;
    }

    if (len != 0) {
        int threads = 256;
        int blocks = static_cast<int>((len + threads - 1) / threads);
        radam_update_reset_gradients_kernel<<<blocks, threads>>>(
            dgrad.ptr,
            dw.ptr,
            dm.ptr,
            dv.ptr,
            len,
            gradient_factor,
            learning_rate,
            step_size,
            use_denom,
            decay,
            beta1,
            beta2,
            epsilon,
            min_weight,
            max_weight);
        if (sync_after_kernel("radam_update_reset_gradients_kernel launch", "radam_update_reset_gradients_kernel sync") != 0) {
            return -1;
        }

        if (do_lookahead != 0) {
            ranger_lookahead_kernel<<<blocks, threads>>>(dw.ptr, dslow.ptr, len, lookahead_alpha);
            if (sync_after_kernel("ranger_lookahead_kernel launch", "ranger_lookahead_kernel sync") != 0) {
                return -1;
            }
        }
    }

    if (copy_d2h(gradients, dgrad.ptr, len, "cudaMemcpy gradients D2H") != 0 ||
        copy_d2h(weights, dw.ptr, len, "cudaMemcpy weights D2H") != 0 ||
        copy_d2h(momentum, dm.ptr, len, "cudaMemcpy momentum D2H") != 0 ||
        copy_d2h(velocity, dv.ptr, len, "cudaMemcpy velocity D2H") != 0 ||
        copy_d2h(slow_params, dslow.ptr, len, "cudaMemcpy slow_params D2H") != 0) {
        return -1;
    }

    return ok();
}
