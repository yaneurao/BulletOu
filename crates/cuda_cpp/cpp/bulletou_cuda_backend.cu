#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

thread_local std::string g_last_error;

struct BulletOuCudaCppContext {
    int device = 0;
    cudaStream_t stream = nullptr;
};

struct BulletOuCudaCppF32Buffer {
    int device = 0;
    size_t len = 0;
    float* ptr = nullptr;
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

__global__ void fill_f32_kernel(size_t len, float value, float* out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    out[idx] = value;
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

    int threads = 256;
    int blocks = static_cast<int>((len + threads - 1) / threads);
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

    if (do_lookahead != 0) {
        ranger_lookahead_kernel<<<blocks, threads, 0, ctx->stream>>>(weights->ptr, slow_params->ptr, len, lookahead_alpha);
        if (check_kernel_launch("ranger_lookahead_kernel launch") != 0) {
            return -1;
        }
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
