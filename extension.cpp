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

void wrapper_stage_two(void *qk, void *v, void *p, void *o, const int head_size, const int batch_size, const int seq_len, const int num_kv_heads, const int num_heads, const int window_size, cudaStream_t stream);
void wrapper_stage_one(void *q, void *k, void *p, void *o, const int head_size, const int batch_size, const int seq_len, const int num_kv_heads, const int num_heads, const int window_size, cudaStream_t stream);

void cache_attn_function(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor p, torch::Tensor o, const int window_size)
{

    torch::Tensor qk = torch::full({q.size(0), q.size(1), q.size(2), k.size(2)}, -50000.0, q.options());

    int head_size = q.size(3);
    int batch_size = q.size(0);
    int seq_len = k.size(2);
    int num_kv_heads = k.size(1);
    int num_heads = q.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());

    wrapper_stage_one(q.data_ptr(),
                      k.data_ptr(),
                      p.data_ptr(),
                      qk.data_ptr(),
                      head_size,
                      batch_size,
                      seq_len,
                      num_kv_heads,
                      num_heads,
                      window_size,
                      stream);

    wrapper_stage_two(torch::softmax(qk, -1).data_ptr(),
                      v.data_ptr(),
                      p.data_ptr(),
                      o.data_ptr(),
                      head_size,
                      batch_size,
                      seq_len,
                      num_kv_heads,
                      num_heads,
                      window_size,
                      stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cache_attn_function", &cache_attn_function, "FP16xFP4 Matrix Multiplication Kernel");
}
