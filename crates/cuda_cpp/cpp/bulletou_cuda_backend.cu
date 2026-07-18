#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

thread_local std::string g_last_error;

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

int fail_message(const char* message) {
    set_error_message(message);
    return -1;
}

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
    float rate = learning_rate * step_size;
    float weight = weights[idx] * (1.0f - decay * rate);
    float m = beta1 * momentum[idx] + (1.0f - beta1) * grad;
    float v = beta2 * velocity[idx] + (1.0f - beta2) * grad * grad;

    float update = m;
    if (use_denom != 0) {
        update /= sqrtf(v) + epsilon;
    }
    weight -= rate * update;
    weight = fminf(fmaxf(weight, min_weight), max_weight);

    gradients[idx] = 0.0f;
    weights[idx] = weight;
    momentum[idx] = m;
    velocity[idx] = v;
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

int validate_host_ptr(const void* ptr, size_t len, const char* name) {
    if (len != 0 && ptr == nullptr) {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer), "%s must not be null when len > 0", name);
        return fail_message(buffer);
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
