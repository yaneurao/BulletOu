#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

thread_local std::string g_last_error;

struct BulletOuCudaCppContext {
    int device = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
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

__device__ float crelu(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

constexpr size_t NNUE_HALFKP_BASE_INPUT_SIZE = 125388;
constexpr size_t NNUE_HALFKP_PIECE_INPUTS = 1548;
constexpr size_t NNUE_HALFKP_FACTORIZED_INPUT_SIZE = NNUE_HALFKP_BASE_INPUT_SIZE + NNUE_HALFKP_PIECE_INPUTS;

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

constexpr size_t SFNN_HALFKA2_BASE_INPUT_SIZE = 131949;
constexpr size_t SFNN_HALFKA2_PIECE_INPUTS = 1629;
constexpr size_t SFNN_HALFKA2_FACTORIZED_INPUT_SIZE = SFNN_HALFKA2_BASE_INPUT_SIZE + SFNN_HALFKA2_PIECE_INPUTS;
constexpr float SFNN_PAIRWISE_SCALE = 127.0f / 128.0f;

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

__global__ void sfnn_stacked_l1_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    const float* shared_weights,
    const float* shared_bias,
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks,
    int has_shared) {
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
        sum += shared_bias[out_col];
    }
    for (size_t in_col = 0; in_col < input_dim; ++in_col) {
        float input_value = input[input_base + in_col];
        sum += input_value * weights[stack_base + out_col * input_dim + in_col];
        if (has_shared != 0) {
            sum += input_value * shared_weights[in_col * output_dim + out_col];
        }
    }
    output[tid] = sum;
}

__global__ void sfnn_l2_input_kernel(const float* l1, float* output, size_t batch, size_t l1_hidden) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t l2_input_dim = l1_hidden * 2;
    size_t total = batch * l2_input_dim;
    if (tid >= total) {
        return;
    }

    size_t col = tid % l2_input_dim;
    size_t sample = tid / l2_input_dim;
    size_t source_col = col % l1_hidden;
    size_t l1_out = l1_hidden + 1;
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
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks) {
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
    for (size_t in_col = 0; in_col < input_dim; ++in_col) {
        sum += input[input_base + in_col] * weights[stack_base + out_col * input_dim + in_col];
    }
    output[tid] = crelu(sum);
}

__global__ void sfnn_stacked_l3_output_kernel(
    const float* input,
    const float* l1,
    const float* weights,
    const float* bias,
    const int* buckets,
    float* output,
    size_t batch,
    size_t input_dim,
    size_t l1_hidden,
    size_t num_stacks) {
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
    for (size_t in_col = 0; in_col < input_dim; ++in_col) {
        sum += input[input_base + in_col] * weights[stack * input_dim + in_col];
    }
    output[sample] = sum + l1[sample * (l1_hidden + 1) + l1_hidden];
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

__global__ void loss_sigmoid_mse_reduce_kernel(
    const float* outputs,
    const float* targets,
    const float* entry_weights,
    float* per_sample,
    float* mean_output_gradients,
    float output_inv_scale,
    size_t batch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }

    float prediction = loss_sigmoid(outputs[idx] * output_inv_scale);
    float error = prediction - targets[idx];
    float weighted = entry_weights[idx] * error * error;
    float gradient = 2.0f * error * prediction * (1.0f - prediction) * output_inv_scale;
    per_sample[idx] = weighted;
    mean_output_gradients[idx] = entry_weights[idx] * gradient / static_cast<float>(batch);
}

__global__ void loss_nnue_pytorch_wrm_reduce_kernel(
    const float* outputs,
    const float* targets,
    const float* entry_weights,
    float* per_sample,
    float* mean_output_gradients,
    size_t batch) {
    constexpr float NNUE2SCORE = 600.0f;
    constexpr float IN_OFFSET = 270.0f;
    constexpr float IN_SCALING = 340.0f;
    constexpr float POW_EXP = 2.5f;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }

    float output = outputs[idx];
    float target = targets[idx];
    float scorenet = output * NNUE2SCORE;
    float q = loss_sigmoid((scorenet - IN_OFFSET) / IN_SCALING);
    float qm = loss_sigmoid((-scorenet - IN_OFFSET) / IN_SCALING);
    float prediction = (1.0f + q - qm) * 0.5f;
    float error = prediction - target;
    float abs_error = fabsf(error);
    float loss = powf(abs_error, POW_EXP);
    float q_prime = q * (1.0f - q);
    float qm_prime = qm * (1.0f - qm);
    float prediction_gradient = 0.5f * (NNUE2SCORE / IN_SCALING) * (q_prime + qm_prime);
    float loss_gradient = POW_EXP * sign_f32(error) * powf(abs_error, POW_EXP - 1.0f);
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

__global__ void sfnn_stacked_l3_backward_kernel(
    const float* inputs,
    const float* output_gradients,
    const float* weights,
    const int* buckets,
    float* input_gradients,
    float* l1_gradients,
    float* weight_gradients,
    float* bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t l1_out,
    size_t num_stacks) {
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
            value = output_gradient * weights[stack * input_dim + row];
            float input_value = inputs[tid];
            if (output_gradient != 0.0f && input_value != 0.0f) {
                atomicAdd(&weight_gradients[stack * input_dim + row], output_gradient * input_value);
            }
        }
        input_gradients[tid] = value;
    }

    if (tid < l1_gradient_len) {
        size_t sample = tid / l1_out;
        size_t col = tid - sample * l1_out;
        l1_gradients[tid] = (col + 1 == l1_out) ? output_gradients[sample] : 0.0f;
    }

    if (tid < batch) {
        int stack_i32 = buckets[tid];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            float grad = output_gradients[tid];
            if (grad != 0.0f) {
                atomicAdd(&bias_gradients[static_cast<size_t>(stack_i32)], grad);
            }
        }
    }
}

__global__ void sfnn_stacked_crelu_backward_kernel(
    const float* inputs,
    const float* activations,
    const float* output_gradients,
    const float* weights,
    const int* buckets,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks) {
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
                size_t out_idx = sample * output_dim + out_col;
                float grad = crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
                if (grad != 0.0f) {
                    sum += grad * weights[stack_base + out_col * input_dim + in_col];
                }
            }
        }
        input_gradients[tid] = sum;
    }

    if (tid < weight_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t input_entry = tid / output_dim;
        size_t in_col = input_entry % input_dim;
        size_t sample = input_entry / input_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t out_idx = sample * output_dim + out_col;
            float grad = crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
            float input_value = inputs[sample * input_dim + in_col];
            if (grad != 0.0f && input_value != 0.0f) {
                size_t weight_idx = stack * output_dim * input_dim + out_col * input_dim + in_col;
                atomicAdd(&weight_gradients[weight_idx], grad * input_value);
            }
        }
    }

    if (tid < bias_scatter_len) {
        size_t out_col = tid % output_dim;
        size_t sample = tid / output_dim;
        int stack_i32 = buckets[sample];
        if (stack_i32 >= 0 && static_cast<size_t>(stack_i32) < num_stacks) {
            size_t stack = static_cast<size_t>(stack_i32);
            size_t out_idx = sample * output_dim + out_col;
            float grad = crelu_pre_gradient_from_value(activations[out_idx], output_gradients[out_idx]);
            if (grad != 0.0f) {
                atomicAdd(&bias_gradients[stack * output_dim + out_col], grad);
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
    size_t l1_hidden) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t l1_out = l1_hidden + 1;
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
    size_t num_stacks) {
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

    if (tid < weight_scatter_len) {
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
    const int* buckets,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    float* shared_weight_gradients,
    float* shared_bias_gradients,
    size_t batch,
    size_t input_dim,
    size_t output_dim,
    size_t num_stacks) {
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
                    float stacked_weight = weights[stack_base + out_col * input_dim + in_col];
                    float shared_weight = shared_weights[in_col * output_dim + out_col];
                    sum += grad * (stacked_weight + shared_weight);
                }
            }
        }
        input_gradients[tid] = sum;
    }

    if (tid < weight_scatter_len) {
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
                float grad = output_gradient * input_value;
                size_t weight_idx = stack * output_dim * input_dim + out_col * input_dim + in_col;
                size_t shared_weight_idx = in_col * output_dim + out_col;
                atomicAdd(&weight_gradients[weight_idx], grad);
                atomicAdd(&shared_weight_gradients[shared_weight_idx], grad);
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
                atomicAdd(&shared_bias_gradients[out_col], grad);
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
    size_t weight_len = input_size * l1;

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
        if (nnue_halfkp_factorized_feature(static_cast<size_t>(stm_feature), input_size, &base_feature, &virtual_feature)) {
            atomicAdd(&l0w_gradients[base_feature * l1 + row], value);
            atomicAdd(&l0w_gradients[virtual_feature * l1 + row], value);
        } else {
            atomicAdd(&l0w_gradients[static_cast<size_t>(stm_feature) * l1 + row], value);
        }
    }

    int nstm_feature = nstm_indices[sparse_base];
    if (nstm_feature >= 0 && static_cast<size_t>(nstm_feature) < input_size) {
        size_t base_feature = 0;
        size_t virtual_feature = 0;
        float value = nstm_gradients[sample * l1 + row];
        if (nnue_halfkp_factorized_feature(static_cast<size_t>(nstm_feature), input_size, &base_feature, &virtual_feature)) {
            atomicAdd(&l0w_gradients[base_feature * l1 + row], value);
            atomicAdd(&l0w_gradients[virtual_feature * l1 + row], value);
        } else {
            atomicAdd(&l0w_gradients[static_cast<size_t>(nstm_feature) * l1 + row], value);
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
    size_t l2_size,
    size_t num_stacks,
    size_t batch,
    size_t max_active) {
    if (input_size == 0 || ft_size == 0 || l1_hidden == 0 || l2_size == 0 || num_stacks == 0) {
        return fail_message("SFNN shape dimensions must be greater than zero");
    }
    if ((ft_size % 2) != 0) {
        return fail_message("SFNN ft_size must be even");
    }
    if (batch == 0) {
        return fail_message("SFNN batch size must be greater than zero");
    }
    if (max_active == 0) {
        return fail_message("SFNN max_active must be greater than zero");
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
    size_t l2_size,
    size_t num_stacks,
    size_t batch,
    size_t max_active,
    const int* stm_indices,
    const int* nstm_indices,
    const int* buckets,
    const float* l0w,
    const float* l0b,
    const float* l1w,
    const float* l1b,
    const float* l1fw,
    const float* l1fb,
    int has_l1f,
    const float* l2w,
    const float* l2b,
    const float* l3w,
    const float* l3b,
    float* stm_l0,
    float* nstm_l0,
    float* combined,
    float* l1,
    float* l2_input,
    float* l2,
    float* output) {
    if (validate_sfnn_shape(input_size, ft_size, l1_hidden, l2_size, num_stacks, batch, max_active) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }

    constexpr int threads = 256;
    int blocks = 0;
    const size_t pairwise = ft_size / 2;
    const size_t l1_out = l1_hidden + 1;
    const size_t l2_in = l1_hidden * 2;

    if (block_count_1d(batch * pairwise, threads, &blocks, "sfnn_sparse_l0_pairwise_concat_kernel") != 0) {
        return -1;
    }
    sfnn_sparse_l0_pairwise_concat_kernel<<<blocks, threads, 0, ctx->stream>>>(
        stm_indices,
        nstm_indices,
        l0w,
        l0b,
        stm_l0,
        nstm_l0,
        combined,
        batch,
        max_active,
        input_size,
        ft_size);
    if (check_kernel_launch("sfnn_sparse_l0_pairwise_concat_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l1_out, threads, &blocks, "sfnn_stacked_l1_kernel") != 0) {
        return -1;
    }
    sfnn_stacked_l1_kernel<<<blocks, threads, 0, ctx->stream>>>(
        combined,
        l1w,
        l1b,
        l1fw,
        l1fb,
        buckets,
        l1,
        batch,
        ft_size,
        l1_out,
        num_stacks,
        has_l1f);
    if (check_kernel_launch("sfnn_stacked_l1_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l2_in, threads, &blocks, "sfnn_l2_input_kernel") != 0) {
        return -1;
    }
    sfnn_l2_input_kernel<<<blocks, threads, 0, ctx->stream>>>(l1, l2_input, batch, l1_hidden);
    if (check_kernel_launch("sfnn_l2_input_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l2_size, threads, &blocks, "sfnn_stacked_l2_crelu_kernel") != 0) {
        return -1;
    }
    sfnn_stacked_l2_crelu_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l2_input, l2w, l2b, buckets, l2, batch, l2_in, l2_size, num_stacks);
    if (check_kernel_launch("sfnn_stacked_l2_crelu_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch, threads, &blocks, "sfnn_stacked_l3_output_kernel") != 0) {
        return -1;
    }
    sfnn_stacked_l3_output_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l2, l1, l3w, l3b, buckets, output, batch, l2_size, l1_hidden, num_stacks);
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
    size_t l2_size,
    size_t num_stacks,
    float* l0w_gradients,
    float* l0b_gradients,
    float* l1w_gradients,
    float* l1b_gradients,
    float* l1fw_gradients,
    float* l1fb_gradients,
    float* l2w_gradients,
    float* l2b_gradients,
    float* l3w_gradients,
    float* l3b_gradients) {
    const size_t l1_out = l1_hidden + 1;
    const size_t l2_in = l1_hidden * 2;
    if (launch_fill_f32_raw(ctx, l0w_gradients, input_size * ft_size, 0.0f, "sfnn zero l0w_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l0b_gradients, ft_size, 0.0f, "sfnn zero l0b_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1w_gradients, num_stacks * l1_out * ft_size, 0.0f, "sfnn zero l1w_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1b_gradients, num_stacks * l1_out, 0.0f, "sfnn zero l1b_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1fw_gradients, ft_size * l1_out, 0.0f, "sfnn zero l1fw_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l1fb_gradients, l1_out, 0.0f, "sfnn zero l1fb_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l2w_gradients, num_stacks * l2_size * l2_in, 0.0f, "sfnn zero l2w_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l2b_gradients, num_stacks * l2_size, 0.0f, "sfnn zero l2b_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l3w_gradients, num_stacks * l2_size, 0.0f, "sfnn zero l3w_gradients") != 0 ||
        launch_fill_f32_raw(ctx, l3b_gradients, num_stacks, 0.0f, "sfnn zero l3b_gradients") != 0) {
        return -1;
    }
    return 0;
}

int launch_sfnn_backward_kernels(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    size_t l2_size,
    size_t num_stacks,
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
    const float* l2w,
    const float* l3w,
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
    float* l2w_gradients,
    float* l2b_gradients,
    float* l3w_gradients,
    float* l3b_gradients,
    int zero_parameter_gradients,
    int fuse_pairwise_l0) {
    if (validate_sfnn_shape(input_size, ft_size, l1_hidden, l2_size, num_stacks, batch, max_active) != 0) {
        return -1;
    }
    if (set_context_device(ctx) != 0) {
        return -1;
    }

    const size_t l1_out = l1_hidden + 1;
    const size_t l2_in = l1_hidden * 2;
    constexpr int threads = 256;
    int blocks = 0;

    if (zero_parameter_gradients != 0) {
        if (launch_zero_sfnn_backward_parameter_gradients(
                ctx,
                input_size,
                ft_size,
                l1_hidden,
                l2_size,
                num_stacks,
                l0w_gradients,
                l0b_gradients,
                l1w_gradients,
                l1b_gradients,
                l1fw_gradients,
                l1fb_gradients,
                l2w_gradients,
                l2b_gradients,
                l3w_gradients,
                l3b_gradients) != 0) {
            return -1;
        }
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
        buckets,
        l2_gradients,
        l1_gradients,
        l3w_gradients,
        l3b_gradients,
        batch,
        l2_size,
        l1_out,
        num_stacks);
    if (check_kernel_launch("sfnn_stacked_l3_backward_kernel launch") != 0) {
        return -1;
    }

    size_t l2_threads = std::max(batch * l2_in, batch * l2_in * l2_size);
    l2_threads = std::max(l2_threads, batch * l2_size);
    if (block_count_1d(l2_threads, threads, &blocks, "sfnn_stacked_crelu_backward_kernel") != 0) {
        return -1;
    }
    sfnn_stacked_crelu_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l2_input,
        l2,
        l2_gradients,
        l2w,
        buckets,
        l2_input_gradients,
        l2w_gradients,
        l2b_gradients,
        batch,
        l2_in,
        l2_size,
        num_stacks);
    if (check_kernel_launch("sfnn_stacked_crelu_backward_kernel launch") != 0) {
        return -1;
    }

    if (block_count_1d(batch * l1_out, threads, &blocks, "sfnn_l2_input_backward_kernel") != 0) {
        return -1;
    }
    sfnn_l2_input_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
        l1, l2_input, l2_input_gradients, l1_gradients, batch, l1_hidden);
    if (check_kernel_launch("sfnn_l2_input_backward_kernel launch") != 0) {
        return -1;
    }

    if (has_l1f != 0) {
        size_t l1_threads = std::max(batch * ft_size, batch * ft_size * l1_out);
        l1_threads = std::max(l1_threads, batch * l1_out);
        if (block_count_1d(l1_threads, threads, &blocks, "sfnn_factorized_l1_backward_kernel") != 0) {
            return -1;
        }
        sfnn_factorized_l1_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
            combined,
            l1_gradients,
            l1w,
            l1fw,
            buckets,
            combined_gradients,
            l1w_gradients,
            l1b_gradients,
            l1fw_gradients,
            l1fb_gradients,
            batch,
            ft_size,
            l1_out,
            num_stacks);
        if (check_kernel_launch("sfnn_factorized_l1_backward_kernel launch") != 0) {
            return -1;
        }
    } else {
        size_t l1_threads = std::max(batch * ft_size, batch * ft_size * l1_out);
        l1_threads = std::max(l1_threads, batch * l1_out);
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
            num_stacks);
        if (check_kernel_launch("sfnn_stacked_affine_backward_kernel launch") != 0) {
            return -1;
        }
    }

    if (fuse_pairwise_l0 != 0) {
        if (block_count_1d(batch * (ft_size / 2), threads, &blocks, "sfnn_pairwise_l0_sparse_backward_kernel") != 0) {
            return -1;
        }
        sfnn_pairwise_l0_sparse_backward_kernel<<<blocks, threads, 0, ctx->stream>>>(
            stm_indices,
            nstm_indices,
            stm_l0,
            nstm_l0,
            combined_gradients,
            l0w_gradients,
            l0b_gradients,
            batch,
            max_active,
            input_size,
            ft_size);
        if (check_kernel_launch("sfnn_pairwise_l0_sparse_backward_kernel launch") != 0) {
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

    return 0;
}

int validate_scalar_loss(size_t batch, int kind) {
    if (batch == 0) {
        return fail_message("scalar loss batch size must be greater than zero");
    }
    if (kind != 0 && kind != 1) {
        return fail_message("scalar loss kind must be 0 (sigmoid-mse) or 1 (nnue-pytorch-wrm)");
    }
    return 0;
}

int launch_scalar_loss_kernels(
    BulletOuCudaCppContext* ctx,
    int kind,
    float output_inv_scale,
    size_t batch,
    const float* outputs,
    const float* targets,
    const float* entry_weights,
    float* per_sample,
    float* mean_output_gradients,
    float* weighted_sum,
    float* mean,
    int finalize_loss) {
    if (validate_scalar_loss(batch, kind) != 0) {
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
        loss_sigmoid_mse_reduce_kernel<<<blocks, threads, 0, ctx->stream>>>(
            outputs,
            targets,
            entry_weights,
            per_sample,
            mean_output_gradients,
            output_inv_scale,
            batch);
        if (check_kernel_launch("loss_sigmoid_mse_reduce_kernel launch") != 0) {
            return -1;
        }
    } else {
        loss_nnue_pytorch_wrm_reduce_kernel<<<blocks, threads, 0, ctx->stream>>>(
            outputs,
            targets,
            entry_weights,
            per_sample,
            mean_output_gradients,
            batch);
        if (check_kernel_launch("loss_nnue_pytorch_wrm_reduce_kernel launch") != 0) {
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
        size_t l0_zero_threads = std::max(input_size * l1, l1);
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
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l0w), input_size * l1, "l0w") != 0 ||
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

extern "C" int bulletou_cuda_cpp_sfnn_forward_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    size_t l2_size,
    size_t num_stacks,
    size_t batch,
    size_t max_active,
    const BulletOuCudaCppI32Buffer* stm_indices,
    const BulletOuCudaCppI32Buffer* nstm_indices,
    const BulletOuCudaCppI32Buffer* buckets,
    const BulletOuCudaCppF32Buffer* l0w,
    const BulletOuCudaCppF32Buffer* l0b,
    const BulletOuCudaCppF32Buffer* l1w,
    const BulletOuCudaCppF32Buffer* l1b,
    const BulletOuCudaCppF32Buffer* l1fw,
    const BulletOuCudaCppF32Buffer* l1fb,
    int has_l1f,
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l2b,
    const BulletOuCudaCppF32Buffer* l3w,
    const BulletOuCudaCppF32Buffer* l3b,
    BulletOuCudaCppF32Buffer* stm_l0,
    BulletOuCudaCppF32Buffer* nstm_l0,
    BulletOuCudaCppF32Buffer* combined,
    BulletOuCudaCppF32Buffer* l1,
    BulletOuCudaCppF32Buffer* l2_input,
    BulletOuCudaCppF32Buffer* l2,
    BulletOuCudaCppF32Buffer* output) {
    const size_t l1_out = l1_hidden + 1;
    const size_t l2_in = l1_hidden * 2;
    if (validate_sfnn_shape(input_size, ft_size, l1_hidden, l2_size, num_stacks, batch, max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), batch * max_active, "sfnn stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), batch * max_active, "sfnn nstm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(buckets), batch, "sfnn buckets") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l0w), input_size * ft_size, "sfnn l0w") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l0b), ft_size, "sfnn l0b") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), num_stacks * l1_out * ft_size, "sfnn l1w") != 0 ||
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
    if (has_l1f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1fw), ft_size * l1_out, "sfnn l1fw") != 0 ||
            validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1fb), l1_out, "sfnn l1fb") != 0) {
            return -1;
        }
    } else if (l1fw != nullptr || l1fb != nullptr) {
        return fail_message("sfnn l1fw/l1fb must be null when has_l1f is false");
    }

    if (launch_sfnn_forward_kernels(
            ctx,
            input_size,
            ft_size,
            l1_hidden,
            l2_size,
            num_stacks,
            batch,
            max_active,
            stm_indices->ptr,
            nstm_indices->ptr,
            buckets->ptr,
            l0w->ptr,
            l0b->ptr,
            l1w->ptr,
            l1b->ptr,
            has_l1f != 0 ? l1fw->ptr : nullptr,
            has_l1f != 0 ? l1fb->ptr : nullptr,
            has_l1f,
            l2w->ptr,
            l2b->ptr,
            l3w->ptr,
            l3b->ptr,
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
        validate_host_ptr(l0w, input_size * l1, "l0w") != 0 ||
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
    if (rc == 0) rc = bulletou_cuda_cpp_f32_buffer_create(ctx, input_size * l1, &d_l0w);
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
        if (rc == 0) rc = bulletou_cuda_cpp_f32_upload(ctx, d_l0w, l0w, input_size * l1);
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
    size_t batch,
    const BulletOuCudaCppF32Buffer* outputs,
    const BulletOuCudaCppF32Buffer* targets,
    const BulletOuCudaCppF32Buffer* entry_weights,
    BulletOuCudaCppF32Buffer* per_sample,
    BulletOuCudaCppF32Buffer* mean_output_gradients,
    BulletOuCudaCppF32Buffer* weighted_sum,
    BulletOuCudaCppF32Buffer* mean,
    int finalize_loss) {
    if (validate_scalar_loss(batch, kind) != 0 ||
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

extern "C" int bulletou_cuda_cpp_scalar_loss_host(
    int device,
    int kind,
    float output_inv_scale,
    size_t batch,
    const float* outputs,
    const float* targets,
    const float* entry_weights,
    float* per_sample,
    float* mean_output_gradients,
    float* weighted_sum,
    float* mean) {
    if (validate_scalar_loss(batch, kind) != 0 ||
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
        validate_buffer(ctx, l0w_gradients, input_size * l1, "l0w_gradients") != 0 ||
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

extern "C" int bulletou_cuda_cpp_sfnn_backward_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    size_t l2_size,
    size_t num_stacks,
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
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l3w,
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
    BulletOuCudaCppF32Buffer* l2w_gradients,
    BulletOuCudaCppF32Buffer* l2b_gradients,
    BulletOuCudaCppF32Buffer* l3w_gradients,
    BulletOuCudaCppF32Buffer* l3b_gradients) {
    const size_t l1_out = l1_hidden + 1;
    const size_t l2_in = l1_hidden * 2;
    if (validate_sfnn_shape(input_size, ft_size, l1_hidden, l2_size, num_stacks, batch, max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), batch * max_active, "sfnn stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), batch * max_active, "sfnn nstm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(buckets), batch, "sfnn buckets") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(stm_l0), batch * ft_size, "sfnn stm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(nstm_l0), batch * ft_size, "sfnn nstm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(combined), batch * ft_size, "sfnn combined") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1), batch * l1_out, "sfnn l1") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2_input), batch * l2_in, "sfnn l2_input") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2), batch * l2_size, "sfnn l2") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), num_stacks * l1_out * ft_size, "sfnn l1w") != 0 ||
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
        validate_buffer(ctx, l1w_gradients, num_stacks * l1_out * ft_size, "sfnn l1w_gradients") != 0 ||
        validate_buffer(ctx, l1b_gradients, num_stacks * l1_out, "sfnn l1b_gradients") != 0 ||
        validate_buffer(ctx, l1fw_gradients, ft_size * l1_out, "sfnn l1fw_gradients") != 0 ||
        validate_buffer(ctx, l1fb_gradients, l1_out, "sfnn l1fb_gradients") != 0 ||
        validate_buffer(ctx, l2w_gradients, num_stacks * l2_size * l2_in, "sfnn l2w_gradients") != 0 ||
        validate_buffer(ctx, l2b_gradients, num_stacks * l2_size, "sfnn l2b_gradients") != 0 ||
        validate_buffer(ctx, l3w_gradients, num_stacks * l2_size, "sfnn l3w_gradients") != 0 ||
        validate_buffer(ctx, l3b_gradients, num_stacks, "sfnn l3b_gradients") != 0) {
        return -1;
    }
    if (has_l1f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1fw), ft_size * l1_out, "sfnn l1fw") != 0) {
            return -1;
        }
    } else if (l1fw != nullptr) {
        return fail_message("sfnn l1fw must be null when has_l1f is false");
    }

    if (launch_sfnn_backward_kernels(
            ctx,
            input_size,
            ft_size,
            l1_hidden,
            l2_size,
            num_stacks,
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
            l2w->ptr,
            l3w->ptr,
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
            l2w_gradients->ptr,
            l2b_gradients->ptr,
            l3w_gradients->ptr,
            l3b_gradients->ptr,
            1,
            0) != 0) {
        return -1;
    }

    return ok();
}

extern "C" int bulletou_cuda_cpp_sfnn_backward_train_device(
    BulletOuCudaCppContext* ctx,
    size_t input_size,
    size_t ft_size,
    size_t l1_hidden,
    size_t l2_size,
    size_t num_stacks,
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
    const BulletOuCudaCppF32Buffer* l2w,
    const BulletOuCudaCppF32Buffer* l3w,
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
    BulletOuCudaCppF32Buffer* l2w_gradients,
    BulletOuCudaCppF32Buffer* l2b_gradients,
    BulletOuCudaCppF32Buffer* l3w_gradients,
    BulletOuCudaCppF32Buffer* l3b_gradients,
    int zero_parameter_gradients) {
    const size_t l1_out = l1_hidden + 1;
    const size_t l2_in = l1_hidden * 2;
    if (validate_sfnn_shape(input_size, ft_size, l1_hidden, l2_size, num_stacks, batch, max_active) != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(stm_indices), batch * max_active, "sfnn stm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(nstm_indices), batch * max_active, "sfnn nstm_indices") != 0 ||
        validate_i32_buffer(ctx, const_cast<BulletOuCudaCppI32Buffer*>(buckets), batch, "sfnn buckets") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(stm_l0), batch * ft_size, "sfnn stm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(nstm_l0), batch * ft_size, "sfnn nstm_l0") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(combined), batch * ft_size, "sfnn combined") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1), batch * l1_out, "sfnn l1") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2_input), batch * l2_in, "sfnn l2_input") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l2), batch * l2_size, "sfnn l2") != 0 ||
        validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1w), num_stacks * l1_out * ft_size, "sfnn l1w") != 0 ||
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
        validate_buffer(ctx, l1w_gradients, num_stacks * l1_out * ft_size, "sfnn l1w_gradients") != 0 ||
        validate_buffer(ctx, l1b_gradients, num_stacks * l1_out, "sfnn l1b_gradients") != 0 ||
        validate_buffer(ctx, l1fw_gradients, ft_size * l1_out, "sfnn l1fw_gradients") != 0 ||
        validate_buffer(ctx, l1fb_gradients, l1_out, "sfnn l1fb_gradients") != 0 ||
        validate_buffer(ctx, l2w_gradients, num_stacks * l2_size * l2_in, "sfnn l2w_gradients") != 0 ||
        validate_buffer(ctx, l2b_gradients, num_stacks * l2_size, "sfnn l2b_gradients") != 0 ||
        validate_buffer(ctx, l3w_gradients, num_stacks * l2_size, "sfnn l3w_gradients") != 0 ||
        validate_buffer(ctx, l3b_gradients, num_stacks, "sfnn l3b_gradients") != 0) {
        return -1;
    }
    if (has_l1f != 0) {
        if (validate_buffer(ctx, const_cast<BulletOuCudaCppF32Buffer*>(l1fw), ft_size * l1_out, "sfnn l1fw") != 0) {
            return -1;
        }
    } else if (l1fw != nullptr) {
        return fail_message("sfnn l1fw must be null when has_l1f is false");
    }

    if (launch_sfnn_backward_kernels(
            ctx,
            input_size,
            ft_size,
            l1_hidden,
            l2_size,
            num_stacks,
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
            l2w->ptr,
            l3w->ptr,
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
            l2w_gradients->ptr,
            l2b_gradients->ptr,
            l3w_gradients->ptr,
            l3b_gradients->ptr,
            zero_parameter_gradients,
            1) != 0) {
        return -1;
    }

    return ok();
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
