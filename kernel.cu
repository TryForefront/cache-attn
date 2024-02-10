
#include "cuda_fp16.h"
#include <cuda_pipeline.h>
#include <stdio.h>
#include <cmath>
#include <tuple>

#define div_ru(a, b) (a + b - 1) / b

#define WARP_SIZE 32

#define CUDA_DEVICE_INLINE __device__ __forceinline__

// the vllm stuff is copied from the vllm repo
// (https://github.com/vllm-project/vllm/blob/3711811b1d2956e83e626c72f0e1607f2dfbc8fb/csrc/cuda_compat.h)
//  and (https://github.com/vllm-project/vllm/blob/3711811b1d2956e83e626c72f0e1607f2dfbc8fb/csrc/attention/attention_kernels.cu#L45)
#define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)

template <int NUM_WARPS>
CUDA_DEVICE_INLINE float block_sum(float *red_smem, float sum)
{
    // Decompose the thread index into warp / lane.
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Compute the sum per warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
    {
        sum += VLLM_SHFL_XOR_SYNC(sum, mask);
    }

    // Warp leaders store the data to shared memory.
    if (lane == 0)
    {
        red_smem[warp] = sum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The warps compute the final sums.
    if (lane < NUM_WARPS)
    {
        sum = red_smem[lane];
    }

    // Parallel reduction inside the warp.
#pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2)
    {
        sum += VLLM_SHFL_XOR_SYNC(sum, mask);
    }

    // Broadcast to other threads.
    return VLLM_SHFL_SYNC(sum, 0);
}

template <int headSize, int numQheads, int numKVheads, int numBlocks>
__global__ void cache_attn(const half *q,  // [numQheads, batchSize, headSize]
                           const half *k,  // [numKVheads, batchSize, headSize]
                           const half *v,  // [numKVheads, batchSize, headSize]
                           half *cache,    // [numBlocks, 2, numKVheads, totalSeq, headSize]
                           const int *pos, // [batchSize]
                           const int *pc,  // [batchSize]
                           half *out,      // [numQheads, batchSize, headSize]
                           int batchSize,
                           int totalSeq,
                           int layer_index)
{

    const int batchIndex = blockIdx.x;
    const int headIndex = blockIdx.y;

    const int kvHeadIndex = headIndex / (numQheads / numKVheads);

    const int qOffset = headIndex * batchSize * headSize + batchIndex * headSize;
    const int kvOffset = kvHeadIndex * batchSize * headSize + batchIndex * headSize;
    const int outOffset = qOffset;

    const int thisSeqLen = pos[batchIndex];

    const int cumulSeqLen = pc[batchIndex];

    const int kCacheOffset = layer_index * 2 * numKVheads * totalSeq * headSize + 0 * numKVheads * totalSeq * headSize + kvHeadIndex * totalSeq * headSize;
    const int vCacheOffset = layer_index * 2 * numKVheads * totalSeq * headSize + 1 * numKVheads * totalSeq * headSize + kvHeadIndex * totalSeq * headSize;

    const int kvOutOffset = kvHeadIndex * (totalSeq + batchSize) * headSize;

    __shared__ half sharedQ[headSize];

    constexpr int NUM_WARPS = headSize / WARP_SIZE;

    __shared__ float blockReduction[2 * NUM_WARPS];

    if (thisSeqLen == 0)
    {
        return;
    }

    sharedQ[threadIdx.x] = __float2half(__half2float(q[qOffset + threadIdx.x]) * 1.44269504 * (1.f / sqrt(headSize)));

    __syncthreads();

    float acc = 0.0f;

    float mi = -50000.f;
    float li = 0.f;

    for (int s = cumulSeqLen + thisSeqLen; s > cumulSeqLen; s--)
    {

        half kItem = (cache[kCacheOffset + s * headSize + threadIdx.x]);

        float thisQk = __half2float(sharedQ[threadIdx.x]) * __half2float(kItem);

        float qk = block_sum<128 / 32>(&blockReduction[NUM_WARPS], thisQk);

        cache[kCacheOffset + (s + batchIndex) * headSize + threadIdx.x] = kItem;

        float mi_new = max(mi, qk);

        float alpha = exp2f(mi - mi_new);

        float p = exp2f(qk - mi_new);

        half vitem = cache[vCacheOffset + s * headSize + threadIdx.x];

        acc *= alpha;

        acc += p * __half2float(vitem);

        li = li * alpha + p;
        mi = mi_new;

        cache[vCacheOffset + (s + batchIndex) * headSize + threadIdx.x] = vitem;
    }

    float qk = 0.f;

    for (int h = 0; h < headSize; h++)
    {

        half kitem = k[kvOffset + h];

        qk += __half2float(sharedQ[h]) * __half2float(kitem);

        cache[kCacheOffset + (cumulSeqLen + thisSeqLen + batchIndex) * headSize + h] = kitem;
    }

    float mi_new = max(mi, qk);

    float alpha = exp2f(mi_new - mi);

    float p = exp2f(qk - mi_new);

    half vitem = v[kvOffset + threadIdx.x];

    acc *= alpha;

    acc += p * __half2float(vitem);

    li = li * alpha + p;

    cache[vCacheOffset + (cumulSeqLen + thisSeqLen + batchIndex) * headSize + threadIdx.x] = vitem;

    out[outOffset + threadIdx.x] = __float2half(acc / li);
}

#define LAUNCH_KERNEL_IF_CONDITION(headDim, numHeads, numKVHeads, numBlocks)                                                                                                 \
    else if (num_heads == numHeads && num_kv_heads == numKVHeads && head_dim == headDim && num_blocks == numBlocks)                                                          \
    {                                                                                                                                                                        \
        auto kernelFunc = cache_attn<headDim, numHeads, numKVHeads, numBlocks>;                                                                                              \
                                                                                                                                                                             \
        constexpr unsigned int smem = 0;                                                                                                                                     \
                                                                                                                                                                             \
        cudaFuncSetAttribute(                                                                                                                                                \
            kernelFunc,                                                                                                                                                      \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                                                                                                                     \
            smem);                                                                                                                                                           \
                                                                                                                                                                             \
        dim3 blocks_per_grid(batchSize, numHeads);                                                                                                                           \
        constexpr dim3 threads_per_block(headDim);                                                                                                                           \
        kernelFunc<<<blocks_per_grid, threads_per_block, smem, stream>>>(Q_ptr, K_ptr, V_ptr, Cache_ptr, P_ptr, PCumulative_ptr, Out_ptr, batchSize, totalSeq, layer_index); \
        return;                                                                                                                                                              \
    }

void wrapper(void *q, void *k, void *v, void *c, void *p, void *pc, void *o, const int batchSize, const int totalSeq, const int layer_index, const int num_heads, const int num_kv_heads, const int head_dim, const int num_blocks, cudaStream_t stream)
{

    const half *Q_ptr = reinterpret_cast<const half *>(q);
    const half *K_ptr = reinterpret_cast<const half *>(k);
    const half *V_ptr = reinterpret_cast<const half *>(v);
    half *Cache_ptr = reinterpret_cast<half *>(c);
    const int *P_ptr = reinterpret_cast<const int *>(p);
    const int *PCumulative_ptr = reinterpret_cast<const int *>(pc);
    half *Out_ptr = reinterpret_cast<half *>(o);

    if (false)
    {
    }
    LAUNCH_KERNEL_IF_CONDITION(128, 32, 8, 32)
    LAUNCH_KERNEL_IF_CONDITION(80, 32, 32, 32)
    LAUNCH_KERNEL_IF_CONDITION(128, 64, 8, 80)
}