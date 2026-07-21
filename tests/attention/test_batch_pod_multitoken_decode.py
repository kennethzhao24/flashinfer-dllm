import pytest
import torch

from flashinfer import BatchPODWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper


@pytest.mark.parametrize(
    "q_len,expected_tile_q",
    [(1, 16), (8, 64), (64, 128)],
)
def test_batch_pod_native_multitoken_decode_matches_batch_prefill(q_len, expected_tile_q):
    """The POD decode branch must use the planner's ragged query tile geometry."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("FA2 tile selection differs on pre-Ampere GPUs")

    torch.manual_seed(0)
    dtype = torch.float16
    num_q_heads, num_kv_heads, head_dim = 16, 4, 128
    page_size = 64
    batch_size = 2
    kv_len = 127
    num_pages = (kv_len + page_size - 1) // page_size

    q_p = torch.randn(16, num_q_heads, head_dim, dtype=dtype, device="cuda")
    q_d = torch.randn(
        batch_size * q_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k_p = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_p = torch.randn_like(k_p)
    k_d = torch.randn(batch_size * num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_d = torch.randn_like(k_d)

    qo_indptr_p = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    kv_indptr_p = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    kv_indices_p = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    last_page_p = torch.tensor([kv_len % page_size], dtype=torch.int32, device="cuda")
    qo_indptr_d = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * q_len
    kv_indptr_d = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * num_pages
    kv_indices_d = torch.arange(batch_size * num_pages, dtype=torch.int32, device="cuda")
    last_page_d = torch.full((batch_size,), kv_len % page_size, dtype=torch.int32, device="cuda")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    pod = BatchPODWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    pod.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_d,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    assert pod._plan_info_d[3] == expected_tile_q

    (_, _), (actual_d, actual_lse_d) = pod.run(
        q_p, (k_p, v_p), q_d, (k_d, v_d), return_lse=True
    )

    reference = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    reference.plan(
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_d,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    expected_d, expected_lse_d = reference.run(q_d, (k_d, v_d), return_lse=True)
    torch.testing.assert_close(actual_d, expected_d, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_lse_d, expected_lse_d, rtol=1e-2, atol=1e-2)
