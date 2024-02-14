/*
 * Copyright (C) 2024 Forefront Industries, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void wrapper(void *q, void *k, void *v, void *c, void *p, void *pc, void *o, const int batchSize, const int totalSeq, const int layer_index, const int num_heads, const int num_kv_heads, const int head_dim, const int num_blocks, cudaStream_t stream);

void cache_attn_function(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor cache, torch::Tensor positions, torch::Tensor cumuPositions, torch::Tensor o, int layer_index)
{

    int head_dim = q.size(-1);
    int batch_size = q.size(0);
    int seq_len = cache.size(-2);
    int num_kv_heads = k.size(1);
    int num_heads = q.size(1);
    int num_blocks = cache.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());

    wrapper(q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            cache.data_ptr(),
            positions.data_ptr(),
            cumuPositions.data_ptr(),
            o.data_ptr(),
            batch_size,
            seq_len,
            layer_index,
            num_heads,
            num_kv_heads,
            head_dim,
            num_blocks,
            stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cache_attn_function", &cache_attn_function, "FP16xFP4 Matrix Multiplication Kernel");
}
