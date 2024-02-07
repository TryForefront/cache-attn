import torch
import cache_attn
import math
import time
from torch.nn import functional as F


def cached_attention_function(q, k, v, window_size, positions):
    o = torch.zeros_like(q)

    cache_attn.cache_attn_function(
        q, k, v.transpose(-2, -1).contiguous(), positions, o, window_size
    )

    return o


def baseline(q, k, v, mask):
    k, v = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


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

    batch_size = 16
    q = torch.randn(batch_size, 32, 1, 128, device="cuda", dtype=torch.half).div(10)
    k_kernel = torch.randn(
        batch_size, 8, 256, 128, device="cuda", dtype=torch.half
    ).div(10)
    v_kernel = torch.randn(
        batch_size, 8, 256, 128, device="cuda", dtype=torch.half
    ).div(10)

    # k = torch.randn(batch_size, 8, 64, 128, device="cuda", dtype=torch.half).div(10)
    # v = torch.randn(batch_size, 8, 64, 128, device="cuda", dtype=torch.half).div(10)

    k = k_kernel.clone()
    v = v_kernel.clone()

    mask = create_mask(batch_size, 256, 4096)[:, :, -1:, :]

    # k_cache = k.clone()
    # v_cache = v.clone()

    # k_single = k_cache[:, :, -1:].contiguous()
    # v_single = v_cache[:, :, -1:].contiguous()

    # k_cache = k_cache[:, :, :-1].contiguous()
    # v_cache = v_cache[:, :, :-1].contiguous()

    # k_cache_fp8 = k_cache.clone()  # f16_to_f8(k_cache)
    # v_cache_fp8 = v_cache.clone()  # f16_to_f8(v_cache)

    window_size = 4096

    positions = torch.tensor(
        [7 if i == 0 else 0 for i in range(batch_size)],
        device="cuda",
        dtype=torch.int,
    ).unsqueeze(-1)

    # # mask = create_mask(batch_size, 512, window_size)[:, :, -1:, :]

    for i, position in enumerate(positions):
        mask[i, :, :, -max(position, 1) :].fill_(True)
        mask[i, :, :, : -max(position, 1)].fill_(False)

    import time

    N_RUNS = 1000

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
        baseline(q, k, v, mask)

    torch.cuda.synchronize()
    end = time.time()

    print(f"Torch Time taken: {(end - start):.4f}")

    for _ in range(N_RUNS):
        _ = cached_attention_function(q, k_kernel, v_kernel, window_size, positions)

    start = time.time()
    for _ in range(N_RUNS):
        _ = cached_attention_function(q, k_kernel, v_kernel, window_size, positions)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Cache Attn Time taken: {(end - start):.4f}")

    # out_ground_truth = baseline(q, k, v)

    # print(o)
    # print(out_ground_truth)

    # print(torch.allclose(o, out_ground_truth))

    o = cached_attention_function(q, k, v, window_size, positions)

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

    out_ground_truth = baseline(q, k, v, mask)

    compare_samples(o[0], out_ground_truth[0])
    # compare_samples(k_test, _k)
    # compare_samples(v_test, _v)
