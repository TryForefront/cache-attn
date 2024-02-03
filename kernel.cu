
#include "cuda_fp16.h"
#include <cuda_pipeline.h>
#include <stdio.h>
#include <cmath>
#include <tuple>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define load(x) __ldcg(x)
#define store(x, value) __stcs(x, value)
#define div_ru(a, b) (a + b - 1) / b
#define div_rd(a, b) a / b

#define CUDA_DEVICE_INLINE __device__ __forceinline__

struct __align__(32) half16
{
    half vals[16];
};

struct __align__(16) char16
{
    uint8_t vals[16];
};

struct __align__(64) float16
{
    float vals[16];
};

CUDA_DEVICE_INLINE float16 &operator+=(float16 &lhs, const float16 &rhs)
{
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        lhs.vals[i] += rhs.vals[i];
    }
    return lhs;
}

CUDA_DEVICE_INLINE half16 operator*(const half16 &lhs, const half16 &rhs)
{
    half16 result;
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        result.vals[i] = lhs.vals[i] * rhs.vals[i];
    }
    return result;
}

CUDA_DEVICE_INLINE half16 __char16_to_half16(const char16 &a)
{
    half16 result;
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        result.vals[i] = __float2half((float)a.vals[i]);
    }
    return result;
}

CUDA_DEVICE_INLINE half16 make_half16(half a, half b, half c, half d, half e, half f, half g, half h, half i, half j, half k, half l, half m, half n, half o, half p)
{
    half16 result;
    result.vals[0] = a;
    result.vals[1] = b;
    result.vals[2] = c;
    result.vals[3] = d;
    result.vals[4] = e;
    result.vals[5] = f;
    result.vals[6] = g;
    result.vals[7] = h;
    result.vals[8] = i;
    result.vals[9] = j;
    result.vals[10] = k;
    result.vals[11] = l;
    result.vals[12] = m;
    result.vals[13] = n;
    result.vals[14] = o;
    result.vals[15] = p;
    return result;
}

CUDA_DEVICE_INLINE half16 make_half16_from_float16(float16 a)
{
    half16 result;
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        result.vals[i] = __float2half(a.vals[i]);
    }
    return result;
}

CUDA_DEVICE_INLINE float dot(half16 a, half16 b)
{
    float result = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        result += __half2float(a.vals[i]) * __half2float(b.vals[i]);
    }
    return result;
}

CUDA_DEVICE_INLINE float dotChar(half16 a, char16 b)
{
    float result = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        result += __half2float(a.vals[i]) * (float)b.vals[i];
    }
    return result;
}

CUDA_DEVICE_INLINE float dotScalar(half a, half16 b)
{
    float result = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        result += __half2float(a) * __half2float(b.vals[i]);
    }
    return result;
}

CUDA_DEVICE_INLINE float16 mulScalarWithfloat8(float a, half16 b)
{
    float16 result;
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        result.vals[i] = a * __half2float(b.vals[i]);
    }
    return result;
}

template <int num_heads, int num_kv_heads, int head_size, int vector_size, int window_size, int SEQ_CHUNK_SIZE>
__global__ void stage_one(const half16 *Q, const half16 *K, const int *positions, half __restrict__ *out, int seq_len, int batch_size)
{
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int tid = threadIdx.x;

    constexpr int KV_HEAD_RATIO = num_heads / num_kv_heads;

    const int kv_head_idx = head_idx / KV_HEAD_RATIO;
    const int batch_stride = seq_len * head_size * num_kv_heads;
    const int head_stride = seq_len * head_size;
    const int seq_stride = seq_len;
    const int q_batch_stride = head_size * num_heads;
    const int q_offset = (batch_idx * q_batch_stride + head_idx * head_size) / vector_size;
    const int kv_offset = (batch_idx * batch_stride + kv_head_idx * head_stride) / vector_size;

    const int out_offset = (batch_idx * num_heads * seq_len + head_idx * seq_len);

    __shared__ half16 shared_Q[head_size / vector_size];

    const int start_s = blockIdx.z * SEQ_CHUNK_SIZE;
    const int end_s = min(start_s + SEQ_CHUNK_SIZE, seq_len);

    int mask_position = seq_len - positions[batch_idx];

    // // mask_position= max(mask_position, seq_len - window_size);
    mask_position = max(mask_position, 0);
    mask_position = min(mask_position, seq_len);
    // but mask_positioncan't be less than seq_len - window_size
    mask_position = max(mask_position, seq_len - window_size);

    if (end_s <= mask_position)
    {

        return;
    }

    const half scalingConstant = __float2half(1.f / sqrtf(head_size));

    const half16 vecScalingConstant = make_half16(
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant,
        scalingConstant);

    shared_Q[tid] = Q[q_offset + tid] * vecScalingConstant;

    __syncthreads();

    // int s_start = seq_len - positions[batch_idx];

    // // s_start = max(s_start, seq_len - window_size);

#pragma unroll
    for (int s = start_s; s < end_s; s++)
    {

        float acc_sum = 0.f;
#pragma unroll
        for (int h = 0; h < head_size / vector_size; h++)
        {
            acc_sum += dot(shared_Q[h], K[kv_offset + (s) * (head_size / vector_size) + h]);
        }

        out[out_offset + s] = __float2half(acc_sum);
    }
}

template <int head_size, int vector_size, int window_size, int num_heads, int num_kv_heads>
__global__ void stage_two(
    const half16 *QK,
    const half16 *V,
    const int *positions,
    half __restrict__ *out,
    int seq_len,
    int batch_size)

{

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int tid = threadIdx.x;

    constexpr int KV_HEAD_RATIO = num_heads / num_kv_heads;

    const int kv_head_idx = head_idx / KV_HEAD_RATIO;

    const int qk_batch_stride = num_heads * 1 * seq_len;

    const int qk_head_stride = 1 * seq_len;

    const int v_batch_stride = num_kv_heads * seq_len * head_size;

    const int v_head_stride = seq_len * head_size;

    const int qk_offset = (batch_idx * qk_batch_stride + head_idx * qk_head_stride) / vector_size;
    const int v_offset = (batch_idx * v_batch_stride + kv_head_idx * v_head_stride) / vector_size;

    const int out_offset = (batch_idx * num_heads * head_size + head_idx * head_size);

    float acc = 0.f;

    int s_start = seq_len - positions[batch_idx];

    // // s_start = max(s_start, seq_len - window_size);
    s_start = max(s_start, 0);
    s_start = min(s_start, seq_len);
    // but s_start can't be less than seq_len - window_size
    s_start = max(s_start, seq_len - window_size);

    s_start /= vector_size;

    for (int s = s_start; s < seq_len / vector_size; s++)
    {
        acc += dot((QK[qk_offset + s]), (V[v_offset + (tid * seq_len / vector_size) + s]));
    }

    out[out_offset + tid] = __float2half(acc);
}

#define LAUNCH_STAGE_ONE_IF_CONDITION(numHeads, numKVHeads, windowSize, headSize, vecSize, seqChunkSize)            \
    else if (numHeads == num_heads && numKVHeads == num_kv_heads && windowSize == window_size && vecSize == 16)     \
    {                                                                                                               \
                                                                                                                    \
        auto kernelFunc = stage_one<numHeads, numKVHeads, headSize, vecSize, windowSize, seqChunkSize>;             \
                                                                                                                    \
        const dim3 blocksPerGrid = {batch_size, numHeads, seq_len / seqChunkSize};                                  \
        constexpr dim3 threadsPerBlock = {headSize / vecSize};                                                      \
        kernelFunc<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(Q_ptr, K_ptr, P_ptr, O_ptr, seq_len, batch_size); \
    }

void wrapper_stage_one(void *q, void *k, void *p, void *o, const int head_size, const int batch_size, const int seq_len, const int num_kv_heads, const int num_heads, const int window_size, cudaStream_t stream)
{

    const half16 *Q_ptr = reinterpret_cast<const half16 *>(q);
    const half16 *K_ptr = reinterpret_cast<const half16 *>(k);
    const int *P_ptr = reinterpret_cast<const int *>(p);
    half *O_ptr = reinterpret_cast<half *>(o);

    constexpr int seq_chunk_len = 16;
    if (false)
    {
    }
    LAUNCH_STAGE_ONE_IF_CONDITION(32, 8, 4096, 128, 16, 16)
    LAUNCH_STAGE_ONE_IF_CONDITION(32, 32, 4096, 80, 16, 16)
    LAUNCH_STAGE_ONE_IF_CONDITION(64, 8, 4096, 64, 16, 16)
}

#define LAUNCH_STAGE_TWO_IF_CONDITION(numHeads, numKVHeads, windowSize, headSize, vecSize)                           \
    else if (numHeads == num_heads && numKVHeads == num_kv_heads && windowSize == window_size && vecSize == 16)      \
    {                                                                                                                \
                                                                                                                     \
        auto kernelFunc = stage_two<headSize, vecSize, windowSize, numHeads, numKVHeads>;                            \
                                                                                                                     \
        const dim3 blocksPerGrid = {batch_size, numHeads};                                                           \
        constexpr dim3 threadsPerBlock = {headSize};                                                                 \
        kernelFunc<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(QK_ptr, V_ptr, P_ptr, O_ptr, seq_len, batch_size); \
    }

void wrapper_stage_two(void *qk, void *v, void *p, void *o, const int head_size, const int batch_size, const int seq_len, const int num_kv_heads, const int num_heads, const int window_size, cudaStream_t stream)
{

    const half16 *QK_ptr = reinterpret_cast<const half16 *>(qk);
    const half16 *V_ptr = reinterpret_cast<const half16 *>(v);
    const int *P_ptr = reinterpret_cast<const int *>(p);
    half *O_ptr = reinterpret_cast<half *>(o);

    constexpr int seq_chunk_len = 16;
    if (false)
    {
    }
    LAUNCH_STAGE_TWO_IF_CONDITION(32, 8, 4096, 128, 16)
    LAUNCH_STAGE_TWO_IF_CONDITION(32, 32, 4096, 80, 16)
    LAUNCH_STAGE_TWO_IF_CONDITION(64, 8, 4096, 64, 16)
}
