import torch
import cache_attn
import math
import time
from torch.nn import functional as F


def cached_attention_function(
    q,
    k_single,
    v_single,
    cache,
    k_out,
    v_out,
    positions,
    cumulativePositions,
    layer_index,
):
    q = q.permute(2, 1, 0, 3).contiguous()
    o = torch.zeros_like(q)

    _k_single = k_single.permute(2, 1, 0, 3).contiguous()
    _v_single = v_single.permute(2, 1, 0, 3).contiguous()

    # _k_cache = k_cache.permute(1, 0, 2, 3).reshape(8, -1, 128).contiguous()
    # _v_cache = v_cache.permute(1, 0, 2, 3).reshape(8, -1, 128).contiguous()

    # _k_out = k_out.permute(1, 0, 2, 3).reshape(8, -1, 128).contiguous()
    # _v_out = v_out.permute(1, 0, 2, 3).reshape(8, -1, 128).contiguous()

    cache_attn.cache_attn_function(
        q.clone(),
        _k_single.clone(),
        _v_single.clone(),
        cache,
        # _k_out.clone(),
        # _v_out.clone(),
        positions,
        cumulativePositions,
        o,
        layer_index,
    )

    return (
        o.permute(2, 1, 0, 3).contiguous(),
        None,
        None,
        # _k_out.reshape(8, -1, 513, 128).permute(1, 0, 2, 3).contiguous(),
        # _v_out.reshape(8, -1, 513, 128).permute(1, 0, 2, 3).contiguous(),
    )


def baseline(q, k_single, v_single, k_cache, v_cache, k_out, v_out, mask):
    k = torch.cat([k_cache, k_single], dim=2)
    v = torch.cat([v_cache, v_single], dim=2)

    out_k = k.clone()
    out_v = v.clone()

    # return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    k, v = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), out_k, out_v

    qk = torch.matmul(q, k.transpose(-2, -1))

    qk = torch.softmax(qk, dim=-1)

    return qk @ v, out_k, out_v


if __name__ == "__main__":

    def compare_samples(a, b):
        print(a.flatten()[:16])
        print(b.flatten()[:16])

        print("###")
        print(a.flatten()[-16:])
        print(b.flatten()[-16:])

        print(torch.allclose(a, b, atol=2e-3))
        print(torch.abs(a - b).max())
        print(torch.abs(a - b).mean())

    def create_mask(bs, seq_length, window_size):
        # Create a square matrix filled with 1 on and below the diagonal and 0 elsewhere
        mask = torch.tril(torch.ones(seq_length, seq_length).cuda()).bool()

        # Update the mask such that for each token all other tokens are masked except the previous window_size tokens and the future tokens
        mask = torch.bitwise_xor(mask, torch.tril(mask, -window_size).bool())

        return (
            torch.where(mask, True, False)
            .cuda()
            .bool()[None, None]
            .repeat_interleave(bs, dim=0)
        )

    batch_size = 1
    q = torch.randn(batch_size, 32, 1, 128, device="cuda", dtype=torch.half).div(10)
    k_kernel = torch.randn(
        32, 1, batch_size, 8, 513, 128, device="cuda", dtype=torch.half
    ).div(10)
    v_kernel = torch.randn(
        32, 1, batch_size, 8, 513, 128, device="cuda", dtype=torch.half
    ).div(10)

    k_out = k_kernel.clone()
    v_out = v_kernel.clone()

    # k = torch.randn(batch_size, 8, 64, 128, device="cuda", dtype=torch.half).div(10)
    # v = torch.randn(batch_size, 8, 64, 128, device="cuda", dtype=torch.half).div(10)

    k = k_kernel.clone()
    v = v_kernel.clone()

    mask = create_mask(batch_size, 513, 4096)[:, :, -1:, :]

    # prev_k, k = k[:, :, :-1], k[:, :, -1:]
    # prev_v, v = v[:, :, :-1], v[:, :, -1:]

    # k_cache = k.clone()
    # v_cache = v.clone()

    k_single = k[0, 0, :, :, -1:].contiguous()
    v_single = v[0, 0, :, :, -1:].contiguous()

    k_cache = k[:, :, :, :, :-1].contiguous()
    v_cache = v[:, :, :, :, :-1].contiguous()

    print(k_single.shape, k_cache.shape)

    # k_cache_fp8 = k_cache.clone()  # f16_to_f8(k_cache)
    # v_cache_fp8 = v_cache.clone()  # f16_to_f8(v_cache)

    # k_single, k_cache = k[:, :, -1:], k[:, :, :-1]
    # v_single, v_cache = v[:, :, -1:], v[:, :, :-1]

    window_size = 4096

    positions = torch.tensor(
        [513 for i in range(batch_size)],
        device="cuda",
        dtype=torch.int,
    )

    cumulativePositions = torch.cumsum(positions, dim=0) - positions

    cache = torch.cat([k_cache, v_cache], dim=1)

    k_cache = k[:, :, :, :, :].contiguous()
    v_cache = v[:, :, :, :, :].contiguous()

    kernel_cache = torch.cat([k_cache, v_cache], dim=1)

    # # mask = create_mask(batch_size, 512, window_size)[:, :, -1:, :]

    for i, position in enumerate(positions):
        mask[i, :, :, :position].fill_(True)
        mask[i, :, :, position:].fill_(False)

    mask[:, :, :, -1:] = True

    import time

    N_RUNS = 10

    # generated_k, k_cache = k[:, :, -5:], k[:, :, :-5]

    # k_cache = torch.cat([generated_k[:, :, :-1], k_cache], dim=2)
    # k_kernel = generated_k[:, :, -1:].contiguous()

    # generated_v, v_cache = v[:, :, -5:], v[:, :, :-5]

    # v_cache = torch.cat([generated_v[:, :, :-1], v_cache], dim=2)
    # v_kernel = generated_v[:, :, -1:].contiguous()

    # print(v_cache.shape)
    # print(k_cache.shape)

    start = time.time()

    for _ in range(N_RUNS):
        # _k, _v = (
        #     k[:, :, 1:, :].repeat_interleave(4, dim=1),
        #     v[:, :, 1:, :].repeat_interleave(4, dim=1),
        # )
        # _k_single = k_single.repeat_interleave(4, dim=1)
        # _v_single = v_single.repeat_interleave(4, dim=1)
        # _ = torch.cat([_k_single, _k], dim=2)
        # _ = torch.cat([_v_single, _v], dim=2)
        # out = F.scaled_dot_product_attention(
        #     q,
        #     k.repeat_interleave(4, dim=1),
        #     v.repeat_interleave(4, dim=1),
        #     attn_mask=None,
        # )
        baseline(q, k_single, v_single, cache[0, 0], cache[0, 1], k_out, v_out, mask)

    torch.cuda.synchronize()
    end = time.time()

    print(f"Torch Time taken: {(end - start):.4f}")

    # cache = torch.cat([k_cache, v_cache], dim=0)

    for _ in range(N_RUNS):
        _ = cached_attention_function(
            q,
            k_single,
            v_single,
            kernel_cache,
            k_out,
            v_out,
            positions,
            cumulativePositions,
            0,
        )

    start = time.time()
    for _ in range(N_RUNS):
        _ = cached_attention_function(
            q,
            k_single,
            v_single,
            kernel_cache,
            k_out,
            v_out,
            positions,
            cumulativePositions,
            0,
        )
    torch.cuda.synchronize()
    end = time.time()

    print(f"Cache Attn Time taken: {(end - start):.4f}")

    # out_ground_truth = baseline(q, k, v)

    # print(o)
    # print(out_ground_truth)

    # print(torch.allclose(o, out_ground_truth))

    o = cached_attention_function(
        q,
        k_single,
        v_single,
        kernel_cache,
        k_out,
        v_out,
        positions,
        cumulativePositions,
        0,
    )

    # _k = torch.cat([k_cache_fp8, k_single], dim=2)[..., 1:, :]
    # _v = torch.cat([v_cache_fp8, v_single], dim=2)[..., 1:, :]

    # out_ground_truth = F.scaled_dot_product_attention(
    #     q,
    #     k.repeat_interleave(4, dim=1),
    #     v.repeat_interleave(4, dim=1),
    #     attn_mask=mask,
    # ).half()

    # out_ground_truth = F.scaled_dot_product_attention(
    #     q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1), attn_mask=None
    # )

    out_ground_truth, _k, _v = baseline(
        q,
        k_single,
        v_single,
        cache[0, 0].clone(),
        cache[0, 1].clone(),
        k_out,
        v_out,
        mask,
    )

    compare_samples(o[0], out_ground_truth[0])
    compare_samples(kernel_cache[0, 0], _k)
    compare_samples(kernel_cache[0, 1], _v)
