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

void wrapper(void *q, void *k, void *v, void *c, void *p, void *pc, void *o, const int batchSize, const int totalSeq, const int layer_index, cudaStream_t stream);

void cache_attn_function(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor cache, torch::Tensor positions, torch::Tensor cumuPositions, torch::Tensor o, int layer_index)
{

    // torch::Tensor qk = torch::full({q.size(0), q.size(1), q.size(2), k.size(2)}, -50000, q.options());

    // int head_size = q.size(3);
    int batch_size = q.size(2);
    int seq_len = cache.size(-2);
    // int num_kv_heads = k.size(0);
    // int num_heads = q.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());

    wrapper(q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            cache.data_ptr(),
            // kcache.data_ptr(),
            // vcache.data_ptr(),
            // kout.data_ptr(),
            // vout.data_ptr(),
            positions.data_ptr(),
            cumuPositions.data_ptr(),
            o.data_ptr(),
            batch_size,
            seq_len,
            layer_index,
            // p.data_ptr(),
            // seq_len.data_ptr(),
            stream);

    // torch::Tensor qk_processed = torch::softmax(qk, -1);

    // wrapper_stage_two(qk_processed.data_ptr(),
    //                   v.data_ptr(),
    //                   p.data_ptr(),
    //                   o.data_ptr(),
    //                   head_size,
    //                   batch_size,
    //                   seq_len,
    //                   num_kv_heads,
    //                   num_heads,
    //                   window_size,
    //                   stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cache_attn_function", &cache_attn_function, "FP16xFP4 Matrix Multiplication Kernel");
}
