/*
 * Minimum CUDA version(IDK why the fuck you would use this but whatever bruh): 9.0 (last version supporting sm_35)
 * Recommended CUDA version: 11.4 (for least pain)
 *
 * REQUIRED Visual Studio Setup: C++ Desktop workload, MSVC v142 and VS 2019 as optimal version
 * Publicly available VS2019 Community download: https://aka.ms/vs/16/release/vs_community.exe
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
 
#ifdef _WIN32
  #define N730_API extern "C" __declspec(dllexport)
#else
  #define N730_API extern "C" __attribute__((visibility("default")))
#endif
 
#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        return N730_CUDA_ERR; \
    } \
} while(0)
 
#define CUBLAS_CHECK(x) do { \
    cublasStatus_t s = (x); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
        return N730_CUDA_ERR; \
    } \
} while(0)
 
static const int N730_OK       =  0;
static const int N730_CUDA_ERR = -10;
static const int N730_OOM      = -11;
static const int N730_NULL     = -12; 
struct N730CudaCtx {
    cublasHandle_t cublas;
 
    // Persistent VRAM buffers — allocated once, reused every layer
    float* d_weights;        // dequantized weight matrix (max layer size)
    float* d_activations;    // current hidden states  (seq * hidden)
    float* d_attn_out;       // attention output buffer
    float* d_mlp_out;        // MLP output buffer
    float* d_qkv;            // Q/K/V projections     (seq * 3 * hidden)
    float* d_scores;         // attention scores      (seq * seq * heads)
    float* d_norm_buf;       // RMSNorm workspace

    //repacked
    float* d_q_repacked;
    float* d_k_repacked;
    float* d_v_repacked;
 
    // Sizes
    int max_weight_elements; // largest layer's rows*cols
    int max_seq;
    int hidden_size;
    int num_heads;
    int head_dim;
    int vocab_size;
    float*   h_weights_pinned;   // pinned host buffer for weight transfers
    uint8_t* h_quant_pinned;     // pinned host buffer for raw quantized bytes
    int      pinned_bytes;
    uint8_t* d_quant_staging;
    int      quant_staging_bytes;
};

__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ src,
    float*         __restrict__ dst,
    int            n_elements,
    float          scale,
    float          zero_point
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_bytes = (n_elements + 1) / 2;
    if (i >= n_bytes) return;
 
    uint8_t byte = src[i];
    float lo = ((float)(byte & 0x0F) - zero_point) * scale;
    float hi = ((float)((byte >> 4) & 0x0F) - zero_point) * scale;
 
    int out0 = i * 2;
    int out1 = out0 + 1;
    dst[out0] = lo;
    if (out1 < n_elements) dst[out1] = hi;
}
 
__global__ void dequant_int8_kernel(
    const uint8_t* __restrict__ src,
    float*         __restrict__ dst,
    int            n_elements,
    float          scale,
    float          zero_point
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    dst[i] = ((float)src[i] - zero_point) * scale;
}
 
__global__ void rmsnorm_kernel(
    float*       __restrict__ x,        // (seq, hidden) — modified in place
    const float* __restrict__ w,        // (hidden,) norm weights
    int          seq,
    int          hidden,
    float        eps
) {
    int row = blockIdx.x;
    if (row >= seq) return;

    float* xrow = x + row * hidden;

    // Each thread accumulates partial sum of squares over its strided elements
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = xrow[i];
        sum_sq += v * v;
    }

    // Step 1: warp-level reduction (handles threads within same warp)
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down(sum_sq, offset);

    // Step 2: write each warp's result to shared memory
    // (up to 8 warps for blockDim.x=256; we use blockDim.x/32 slots)
    extern __shared__ float warp_sums[];   // blockDim.x/32 floats
    int lane   = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    // Step 3: first warp reduces the warp partial sums
    int n_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        sum_sq = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down(sum_sq, offset);
        if (lane == 0)
            warp_sums[0] = rsqrtf(sum_sq / hidden + eps);
    }
    __syncthreads();

    float rms_inv = warp_sums[0];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        xrow[i] = xrow[i] * rms_inv * w[i];
}
 
__global__ void silu_kernel(
    float* __restrict__ gate,    // modified in place
    int    n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    float x = gate[i];
    gate[i] = x / (1.0f + expf(-x));
}

__global__ void elemwise_mul_kernel(
    float*       __restrict__ gate,  // modified in place: gate = gate * up
    const float* __restrict__ up,
    int          n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    gate[i] *= up[i];
}
 
/*
 * Residual add: x += delta
 */
__global__ void residual_add_kernel(
    float*       __restrict__ x,
    const float* __restrict__ delta,
    int          n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    x[i] += delta[i];
}

__global__ void add_bias_kernel(
    float*       __restrict__ x,     // (seq, dim), modified in place
    const float* __restrict__ bias,  // (dim,)
    int          seq,
    int          dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = seq * dim;
    if (i >= n) return;
    x[i] += bias[i % dim];
}

__global__ void softmax_kernel(
    float* __restrict__ scores,   // (n_rows, seq) modified in place
    int    n_rows,
    int    seq
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float* s = scores + row * seq;

    extern __shared__ float smem[];  // 2 * n_warps floats: [0..n_warps-1]=max, [n_warps..]=sum
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int n_warps = blockDim.x >> 5;
    float* smem_max = smem;
    float* smem_sum = smem + n_warps;

    // ── Pass 1: find max ──────────────────────────────────────────────────
    float mx = -1e20f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
        mx = fmaxf(mx, s[i]);
    for (int offset = 16; offset > 0; offset >>= 1)
        mx = fmaxf(mx, __shfl_down(mx, offset));
    if (lane == 0) smem_max[warp_id] = mx;
    __syncthreads();
    if (warp_id == 0) {
        mx = (lane < n_warps) ? smem_max[lane] : -1e20f;
        for (int offset = 16; offset > 0; offset >>= 1)
            mx = fmaxf(mx, __shfl_down(mx, offset));
        if (lane == 0) smem_max[0] = mx;
    }
    __syncthreads();
    mx = smem_max[0];

    // ── Pass 2: exp and partial sum ───────────────────────────────────────
    float sum = 0.0f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        s[i] = expf(fmaxf(s[i] - mx, -50.0f));
        sum += s[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) smem_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane < n_warps) ? smem_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) smem_sum[0] = sum + 1e-9f;
    }
    __syncthreads();
    sum = smem_sum[0];

    // ── Pass 3: normalize ─────────────────────────────────────────────────
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
        s[i] /= sum;
}

__global__ void causal_mask_kernel(
    float* __restrict__ scores,  // (n_heads, seq_q, seq_total)
    int    n_heads,
    int    seq_q,
    int    seq_total,
    int    cache_offset
) {
    int h = blockIdx.z;
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads || q >= seq_q || k >= seq_total) return;
 
    int abs_q = cache_offset + q;
    if (k > abs_q) {
        scores[h * seq_q * seq_total + q * seq_total + k] = -1e4f;
    }
}
 
__global__ void rope_kernel(
    float*       __restrict__ x,      // (seq, n_heads, head_dim)
    const float* __restrict__ cos_f,  // (max_seq, head_dim/2)
    const float* __restrict__ sin_f,
    int          seq,
    int          n_heads,
    int          head_dim,
    int          offset                // position offset for KV cache
) {
    int s = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;  // iterates over head_dim/2
    if (s >= seq || h >= n_heads || i >= head_dim / 2) return;
 
    float* xsh = x + s * n_heads * head_dim + h * head_dim;
    int pos = s + offset;
    float c = cos_f[pos * (head_dim / 2) + i];
    float sv = sin_f[pos * (head_dim / 2) + i];

    int d0 = i;                  // first half
    int d1 = i + head_dim / 2;   // second half (rotate_half pairing)

    float x0 = xsh[d0];
    float x1 = xsh[d1];

    xsh[d0] = x0 * c - x1 * sv;
    xsh[d1] = x1 * c + x0 * sv;
}

__global__ void repack_qkv_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int seq,
    int n_heads,
    int head_dim
) {
    int s = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;

    if (s >= seq || h >= n_heads || d >= head_dim)
        return;

    // src: (seq, head, dim)
    int src_idx =
        s * n_heads * head_dim +
        h * head_dim +
        d;

    // dst: (head, seq, dim)
    int dst_idx =
        h * seq * head_dim +
        s * head_dim +
        d;

    dst[dst_idx] = src[src_idx];
}

__global__ void unpack_attn_kernel(
    const float* src,
    float* dst,
    int seq,
    int n_heads,
    int head_dim
) {
    int s = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;

    if (s >= seq || h >= n_heads || d >= head_dim)
        return;

    int src_idx =
        h * seq * head_dim +
        s * head_dim +
        d;

    int dst_idx =
        s * n_heads * head_dim +
        h * head_dim +
        d;

    dst[dst_idx] = src[src_idx];
}
 
N730_API int n730_cuda_init(
    int hidden_size,
    int num_heads,
    int head_dim,
    int vocab_size,
    int max_seq,
    int max_weight_elements,
    void** out_ctx
) {
    N730CudaCtx* ctx = new N730CudaCtx{};
    ctx->hidden_size         = hidden_size;
    ctx->num_heads           = num_heads;
    ctx->head_dim            = head_dim;
    ctx->vocab_size          = vocab_size;
    ctx->max_seq             = max_seq;
    ctx->max_weight_elements = max_weight_elements;
 
    // Init cuBLAS
    if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) {
        delete ctx; return N730_CUDA_ERR;
    }
 
    // Allocate persistent VRAM buffers
    size_t wbytes  = (size_t)max_weight_elements * sizeof(float);
    size_t abytes  = (size_t)max_seq * hidden_size * sizeof(float);
    size_t qkv     = (size_t)max_seq * 3 * hidden_size * sizeof(float);
    size_t scores  = (size_t)num_heads * max_seq * max_seq * sizeof(float);
 
    if (cudaMalloc(&ctx->d_weights,     wbytes)  != cudaSuccess ||
        cudaMalloc(&ctx->d_activations, abytes)  != cudaSuccess ||
        cudaMalloc(&ctx->d_attn_out,    abytes)  != cudaSuccess ||
        cudaMalloc(&ctx->d_mlp_out,     abytes)  != cudaSuccess ||
        cudaMalloc(&ctx->d_qkv,         qkv)     != cudaSuccess ||
        cudaMalloc(&ctx->d_q_repacked,  qkv)     != cudaSuccess ||
        cudaMalloc(&ctx->d_k_repacked,  qkv)     != cudaSuccess ||
        cudaMalloc(&ctx->d_v_repacked,  qkv)     != cudaSuccess ||
        cudaMalloc(&ctx->d_scores,      scores)  != cudaSuccess ||
        cudaMalloc(&ctx->d_norm_buf,    abytes)  != cudaSuccess) {
        delete ctx; return N730_OOM;
    }
 
    // Pinned host memory for fast H2D transfers
    int pinned = max_weight_elements * 4;  // enough for FP32 or INT8
    ctx->pinned_bytes = pinned;
    if (cudaMallocHost(&ctx->h_weights_pinned, pinned) != cudaSuccess ||
        cudaMallocHost(&ctx->h_quant_pinned,   pinned) != cudaSuccess) {
        delete ctx; return N730_OOM;
    }

    ctx->quant_staging_bytes = max_weight_elements;
    if (cudaMalloc(&ctx->d_quant_staging, (size_t)max_weight_elements) != cudaSuccess) {
        delete ctx; return N730_OOM;
    }
 
    *out_ctx = ctx;
    printf("N730 CUDA ready: hidden=%d heads=%d vocab=%d\n",
           hidden_size, num_heads, vocab_size);
    printf("VRAM allocated: weights=%.1fMB activations=%.1fMB\n",
           wbytes/1048576.0f, abytes/1048576.0f);
    return N730_OK;
}
 
N730_API void n730_cuda_destroy(void* ctx_ptr) {
    if (!ctx_ptr) return;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;
    cublasDestroy(ctx->cublas);
    cudaFree(ctx->d_weights);
    cudaFree(ctx->d_activations);
    cudaFree(ctx->d_attn_out);
    cudaFree(ctx->d_mlp_out);
    cudaFree(ctx->d_qkv);
    cudaFree(ctx->d_scores);
    cudaFree(ctx->d_norm_buf);
    cudaFree(ctx->d_q_repacked);
    cudaFree(ctx->d_k_repacked);
    cudaFree(ctx->d_v_repacked);
    cudaFreeHost(ctx->h_weights_pinned);
    cudaFreeHost(ctx->h_quant_pinned);
    cudaFree(ctx->d_quant_staging);
    delete ctx;
}

N730_API int n730_load_activations(
    void*        ctx_ptr,
    const float* host_activations,
    int          seq_len,
    int          hidden_size
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;
    size_t bytes = (size_t)seq_len * hidden_size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ctx->d_activations, host_activations, bytes,
                          cudaMemcpyHostToDevice));
    return N730_OK;
}
 
N730_API int n730_get_activations(
    void*  ctx_ptr,
    float* host_out,
    int    seq_len,
    int    hidden_size
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;
    size_t bytes = (size_t)seq_len * hidden_size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(host_out, ctx->d_activations, bytes,
                          cudaMemcpyDeviceToHost));
    return N730_OK;
}
 
N730_API int n730_upload_weight(
    void*          ctx_ptr,
    const uint8_t* raw_bytes,
    int            prec_id,
    int            n_elements,
    float          scale,
    float          zero_point
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;

    int raw_bytes_count = (prec_id == 4) ? (n_elements + 1) / 2 : n_elements;

    memcpy(ctx->h_quant_pinned, raw_bytes, raw_bytes_count);

    CUDA_CHECK(cudaMemcpy(ctx->d_quant_staging, ctx->h_quant_pinned, raw_bytes_count,
                          cudaMemcpyHostToDevice));

    // Dequantize on GPU: d_quant_staging (raw bytes) → d_weights (float32)
    int threads = 256;
    if (prec_id == 4) {
        int n_bytes = (n_elements + 1) / 2;
        int blocks  = (n_bytes + threads - 1) / threads;
        dequant_int4_kernel<<<blocks, threads>>>(
            ctx->d_quant_staging, ctx->d_weights, n_elements, scale, zero_point);
    } else {
        int blocks = (n_elements + threads - 1) / threads;
        dequant_int8_kernel<<<blocks, threads>>>(
            ctx->d_quant_staging, ctx->d_weights, n_elements, scale, zero_point);
    }

    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}
 
N730_API int n730_upload_norm_weight(
    const float* host_w,
    int          n_elements,
    void**       out_ptr
) {
    float* d_w;
    size_t bytes = n_elements * sizeof(float);
    if (cudaMalloc(&d_w, bytes) != cudaSuccess) return N730_OOM;
    if (cudaMemcpy(d_w, host_w, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_w); return N730_CUDA_ERR;
    }
    *out_ptr = d_w;
    return N730_OK;
}
 
N730_API void n730_free_device_buf(void* ptr) {
    if (ptr) cudaFree(ptr);
}

N730_API int n730_rmsnorm(
    void*        ctx_ptr,
    const float* d_norm_w,   // device pointer to norm weights
    int          seq_len,
    float        eps
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;
 
    // Copy activations → norm_buf, then normalize in place
    size_t bytes = (size_t)seq_len * ctx->hidden_size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ctx->d_norm_buf, ctx->d_activations, bytes,
                          cudaMemcpyDeviceToDevice));

    // 128 threads = 4 warps; shared memory = 4 floats (one per warp)
    int threads = 128;
    int smem    = (threads / 32) * sizeof(float);
    rmsnorm_kernel<<<seq_len, threads, smem>>>(
        ctx->d_norm_buf, d_norm_w, seq_len, ctx->hidden_size, eps);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}
 
N730_API int n730_linear(
    void*  ctx_ptr,
    float* d_out,       // pre-allocated device output buffer
    int    seq_len,
    int    in_dim,
    int    out_dim
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;
 
    const float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        ctx->cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_dim,
        seq_len,
        in_dim,
        &alpha,
        ctx->d_weights,
        in_dim,
        ctx->d_norm_buf,
        in_dim,
        &beta,
        d_out,
        out_dim
    ));

    return N730_OK;
}

N730_API int n730_residual_add(
    void*        ctx_ptr,
    const float* d_delta,
    int          seq_len
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;
    int n = seq_len * ctx->hidden_size;
    int threads = 256, blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads>>>(ctx->d_activations, d_delta, n);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}
 
N730_API int n730_add_bias(
    void*        ctx_ptr,
    float*       d_out,
    const float* d_bias,
    int          seq_len,
    int          dim
) {
    if (!ctx_ptr) return N730_NULL;
    int n = seq_len * dim;
    int threads = 256, blocks = (n + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(d_out, d_bias, seq_len, dim);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}

N730_API int n730_swiglu(
    float* d_gate,
    float* d_up,
    int    seq_len,
    int    intermediate_size
) {
    int n = seq_len * intermediate_size;
    int threads = 256, blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads>>>(d_gate, n);
    elemwise_mul_kernel<<<blocks, threads>>>(d_gate, d_up, n);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}
 
N730_API int n730_apply_rope(
    float*       d_x,         // (seq, n_heads, head_dim) device
    const float* d_cos,       // (max_seq, head_dim/2) device
    const float* d_sin,
    int          seq_len,
    int          n_heads,
    int          head_dim,
    int          position_offset
) {
    dim3 blocks(seq_len, n_heads);
    int threads = min(head_dim / 2, 256);
    rope_kernel<<<blocks, threads>>>(d_x, d_cos, d_sin,
                                     seq_len, n_heads, head_dim,
                                     position_offset);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}

N730_API int n730_rope_precompute(
    int    max_seq,
    int    head_dim,
    float  theta,
    void** d_cos_out,
    void** d_sin_out
) {
    // Build on host first
    int h2 = head_dim / 2;
    float* h_cos = new float[max_seq * h2];
    float* h_sin = new float[max_seq * h2];
 
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < h2; i++) {
            float freq = 1.0f / powf(theta, (float)(2*i) / head_dim);
            float angle = pos * freq;
            h_cos[pos * h2 + i] = cosf(angle);
            h_sin[pos * h2 + i] = sinf(angle);
        }
    }
 
    size_t bytes = (size_t)max_seq * h2 * sizeof(float);
    float *d_cos, *d_sin;
    if (cudaMalloc(&d_cos, bytes) != cudaSuccess ||
        cudaMalloc(&d_sin, bytes) != cudaSuccess) {
        delete[] h_cos; delete[] h_sin;
        return N730_OOM;
    }
    cudaMemcpy(d_cos, h_cos, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin, bytes, cudaMemcpyHostToDevice);
 
    delete[] h_cos; delete[] h_sin;
    *d_cos_out = d_cos;
    *d_sin_out = d_sin;
    return N730_OK;
}

N730_API int n730_softmax_scores(
    float* d_scores,
    int    n_heads,
    int    seq_q,
    int    seq_total,
    int    cache_offset
) {
    // Apply causal mask
    if (seq_q > 1) {
        dim3 threads(16, 16);
        dim3 blocks(
            (seq_total + 15) / 16,
            (seq_q    + 15) / 16,
            n_heads
        );
        causal_mask_kernel<<<blocks, threads>>>(
            d_scores, n_heads, seq_q, seq_total, cache_offset);
        CUDA_CHECK(cudaGetLastError());
    }
 
    // Softmax over each (head, query) row — 64 threads = 2 warps, smem = 2*2 floats
    int n_rows = n_heads * seq_q;
    int sf_threads = 64;
    int sf_smem    = 2 * (sf_threads / 32) * sizeof(float);
    softmax_kernel<<<n_rows, sf_threads, sf_smem>>>(
        d_scores, n_rows, seq_total);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}
 
N730_API int n730_attention_forward(
    void*        ctx_ptr,
    const float* d_q,          // (seq, n_heads, head_dim) device
    const float* d_k_cache,    // (total_seq, n_kv_heads, head_dim) device
    const float* d_v_cache,    // (total_seq, n_kv_heads, head_dim) device
    float*       d_out,        // (seq, n_heads * head_dim) device — output
    int          seq_q,        // number of query tokens (1 in decode, >1 in prefill)
    int          seq_total,    // total KV length (cache + new tokens)
    int          n_heads,
    int          n_kv_heads,
    int          head_dim,
    int          cache_offset  // number of previously cached tokens
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;

    const float scale    = 1.0f / sqrtf((float)head_dim);
    const float alpha1   = scale;
    const float beta0    = 0.0f;
    const float alpha1f  = 1.0f;

    int grp = n_heads / n_kv_heads;  // GQA group size
    dim3 q_blocks(seq_q, n_heads);
    dim3 kv_blocks(seq_total, n_kv_heads);

    int threads = min(head_dim, 256);

    repack_qkv_kernel<<<q_blocks, threads>>>(
        d_q,
        ctx->d_q_repacked,
        seq_q,
        n_heads,
        head_dim
    );

    repack_qkv_kernel<<<kv_blocks, threads>>>(
        d_k_cache,
        ctx->d_k_repacked,
        seq_total,
        n_kv_heads,
        head_dim
    );

    repack_qkv_kernel<<<kv_blocks, threads>>>(
        d_v_cache,
        ctx->d_v_repacked,
        seq_total,
        n_kv_heads,
        head_dim
    );

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / grp;

        const float* Q_h =
            ctx->d_q_repacked +
            (long long)h * seq_q * head_dim;

        const float* K_h =
            ctx->d_k_repacked +
            (long long)kv_h * seq_total * head_dim;

        float*       score_h = ctx->d_scores + (long long)h * seq_q * seq_total;

        CUBLAS_CHECK(cublasSgemm(
            ctx->cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_total, seq_q, head_dim,     // m, n, k
            &alpha1,
            K_h, head_dim,     // A=K, lda=row stride of K
            Q_h, head_dim,     // B=Q, ldb=row stride of Q
            &beta0,
            score_h, seq_total              // C=score, ldc=seq_total (contiguous)
        ));
    }

    // ── Step 2: causal mask + softmax ────────────────────────────────────
    if (seq_q > 1) {
        dim3 threads(16, 16);
        dim3 blocks(
            (seq_total + 15) / 16,
            (seq_q     + 15) / 16,
            n_heads
        );
        causal_mask_kernel<<<blocks, threads>>>(
            ctx->d_scores, n_heads, seq_q, seq_total, cache_offset);
        CUDA_CHECK(cudaGetLastError());
    }

    int n_rows     = n_heads * seq_q;
    int sf_threads = 64;
    int sf_smem    = 2 * (sf_threads / 32) * sizeof(float);
    softmax_kernel<<<n_rows, sf_threads, sf_smem>>>(ctx->d_scores, n_rows, seq_total);
    CUDA_CHECK(cudaGetLastError());

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / grp;

        const float* score_h = ctx->d_scores + (long long)h * seq_q * seq_total;
        const float* V_h =
            ctx->d_v_repacked +
            (long long)kv_h * seq_total * head_dim;

        // Write into internal scratch buffer (head-major layout)
        float* out_h =
            ctx->d_attn_out +
            (long long)h * seq_q * head_dim;

        CUBLAS_CHECK(cublasSgemm(
            ctx->cublas,
            CUBLAS_OP_N,                    // V (no transpose)
            CUBLAS_OP_N,                    // score (no transpose)
            head_dim, seq_q, seq_total,     // m, n, k
            &alpha1f,
            V_h,     head_dim,              // A=V,     lda=head_dim (repacked)
            score_h, seq_total,             // B=score, ldb=seq_total (contiguous)
            &beta0,
            out_h,   head_dim              // C=out_h, ldc=head_dim (head-major)
        ));
    }

    dim3 unpack_blocks(seq_q, n_heads);
    int unpack_threads = min(head_dim, 256);

    unpack_attn_kernel<<<unpack_blocks, unpack_threads>>>(
        ctx->d_attn_out,   // src: head-major scratch
        d_out,             // dst: caller's seq-major output buffer
        seq_q,
        n_heads,
        head_dim
    );

    CUDA_CHECK(cudaGetLastError());

    return N730_OK;
}

N730_API int n730_linear_from_buf(
    void*        ctx_ptr,
    const float* d_in,      // (seq, in_dim) device — arbitrary input
    float*       d_out,     // (seq, out_dim) device — output
    int          seq_len,
    int          in_dim,
    int          out_dim
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;

    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        ctx->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,   // matches n730_linear
        out_dim, seq_len, in_dim,
        &alpha,
        ctx->d_weights, in_dim,     // A = weights (transposed)
        d_in,           in_dim,     // B = input
        &beta,
        d_out,          out_dim
    ));
    return N730_OK;
}

N730_API int n730_get_scores(
    void*  ctx_ptr,
    float* h_out,
    int    n_heads,
    int    seq_q,
    int    seq_total
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;
    size_t n = (size_t)n_heads * seq_q * seq_total;
    CUDA_CHECK(cudaMemcpy(h_out, ctx->d_scores, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return N730_OK;
}

N730_API int n730_attention_qk_only(
    void*        ctx_ptr,
    const float* d_q,
    const float* d_k_cache,
    int          seq_q,
    int          seq_total,
    int          n_heads,
    int          n_kv_heads,
    int          head_dim
) {
    if (!ctx_ptr) return N730_NULL;
    N730CudaCtx* ctx = (N730CudaCtx*)ctx_ptr;

    dim3 q_blocks(seq_q, n_heads);
    dim3 kv_blocks(seq_total, n_kv_heads);

    int threads = min(head_dim, 256);

    repack_qkv_kernel<<<q_blocks, threads>>>(
        d_q,
        ctx->d_q_repacked,
        seq_q,
        n_heads,
        head_dim
    );

    repack_qkv_kernel<<<kv_blocks, threads>>>(
        d_k_cache,
        ctx->d_k_repacked,
        seq_total,
        n_kv_heads,
        head_dim
    );

    int grp = n_heads / n_kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const float beta0 = 0.0f;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / grp;
        
        const float* Q_h =
            ctx->d_q_repacked +
            (long long)h * seq_q * head_dim;

        const float* K_h =
            ctx->d_k_repacked +
            (long long)kv_h * seq_total * head_dim;
        
        float* score_h = ctx->d_scores + (long long)h * seq_q * seq_total;

        CUBLAS_CHECK(cublasSgemm(
            ctx->cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq_total, seq_q, head_dim,
            &scale,
            K_h, head_dim,
            Q_h, head_dim,
            &beta0,
            score_h, seq_total
        ));
    }
    return N730_OK;
}


N730_API int n730_device_alloc(int n_floats, void** out_ptr) {
    float* p;
    if (cudaMalloc(&p, n_floats * sizeof(float)) != cudaSuccess)
        return N730_OOM;
    *out_ptr = p;
    return N730_OK;
}
 
N730_API void n730_device_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

N730_API int n730_memcpy_d2d(void* dst, const void* src, int n_floats) {
    CUDA_CHECK(cudaMemcpy(dst, src, n_floats * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return N730_OK;
}
N730_API int n730_memcpy_h2d(void* dst, const void* src, int n_floats) {
    CUDA_CHECK(cudaMemcpy(dst, src, n_floats * sizeof(float),
                          cudaMemcpyHostToDevice));
    return N730_OK;
}
N730_API int n730_memcpy_d2h(void* dst, const void* src, int n_floats) {
    CUDA_CHECK(cudaMemcpy(dst, src, n_floats * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return N730_OK;
}
 
N730_API int n730_sync() {
    CUDA_CHECK(cudaDeviceSynchronize());
    return N730_OK;
}
 
N730_API const char* n730_cuda_version() {
    return "N730 CUDA Kernel version whatever/ sm_35";
}
