from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from typing import Optional

import torch


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_dtype(value: str) -> torch.dtype:
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    try:
        return choices[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unsupported dtype {value!r}; expected one of {sorted(choices)}"
        ) from exc


def _parse_index_dtype(value: str) -> torch.dtype:
    choices = {
        "i32": torch.int32,
        "int32": torch.int32,
        "i64": torch.int64,
        "int64": torch.int64,
    }
    try:
        return choices[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unsupported index dtype {value!r}; expected one of {sorted(choices)}"
        ) from exc


def _make_indptr(lengths: tuple[int, ...]) -> tuple[int, ...]:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return tuple(values)


def _page_counts(lengths: tuple[int, ...], page_size: int) -> tuple[int, ...]:
    return tuple((int(length) + page_size - 1) // page_size for length in lengths)


def _last_page_lens(lengths: tuple[int, ...], page_size: int) -> tuple[int, ...]:
    return tuple(((int(length) - 1) % page_size) + 1 for length in lengths)


def _make_paged_kv_metadata(
    lengths: tuple[int, ...],
    page_size: int,
    index_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_indptr = _make_indptr(_page_counts(lengths, page_size))
    num_pages = page_indptr[-1]
    return (
        torch.tensor(page_indptr, dtype=index_dtype, device=device),
        torch.arange(num_pages, dtype=index_dtype, device=device),
        torch.tensor(_last_page_lens(lengths, page_size), dtype=index_dtype, device=device),
    )


def _make_paged_kv_cache(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    kv_layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_layout == "NHD":
        cache = torch.randn(
            num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        return cache.unbind(1)
    if kv_layout == "HND":
        cache = torch.randn(
            num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        return cache.unbind(1)
    raise ValueError(f"unsupported kv_layout {kv_layout!r}")


def _event_time_ms(fn, warmup: int, iters: int):
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        output = fn()
    end.record()
    end.synchronize()
    assert output is not None
    return float(start.elapsed_time(end)) / iters, output


def _tensor_shapes(value) -> object:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, (tuple, list)):
        return [_tensor_shapes(item) for item in value]
    return str(type(value))


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads for grouped-query attention."
        )
    if args.page_size <= 0:
        raise ValueError(f"page_size must be positive, got {args.page_size}.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be >= 0 and iters must be > 0.")
    if any(length <= 0 for length in args.prefill_seq_lens):
        raise ValueError(f"prefill_seq_lens must be positive, got {args.prefill_seq_lens}.")
    if any(length <= 0 for length in args.decode_kv_lens):
        raise ValueError(f"decode_kv_lens must be positive, got {args.decode_kv_lens}.")

    if args.decode_q_lens is None:
        decode_q_lens = tuple(1 for _ in args.decode_kv_lens)
    else:
        decode_q_lens = args.decode_q_lens
    if len(decode_q_lens) != len(args.decode_kv_lens):
        raise ValueError(
            "decode_q_lens must have the same batch size as decode_kv_lens: "
            f"got {decode_q_lens} vs {args.decode_kv_lens}."
        )
    if any(length <= 0 for length in decode_q_lens):
        raise ValueError(f"decode_q_lens must be positive, got {decode_q_lens}.")
    if any(q_len > kv_len for q_len, kv_len in zip(decode_q_lens, args.decode_kv_lens)):
        raise ValueError(
            "Each decode_q_len must be <= the matching decode_kv_len: "
            f"got q={decode_q_lens}, kv={args.decode_kv_lens}."
        )
    return decode_q_lens


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    decode_q_lens = _validate_args(args)

    import flashinfer

    torch.manual_seed(args.seed)
    if args.device is not None:
        torch.cuda.set_device(args.device)
    device = torch.device("cuda", torch.cuda.current_device())

    qo_indptr_p = torch.tensor(
        _make_indptr(args.prefill_seq_lens), dtype=args.index_dtype, device=device
    )
    kv_indptr_p, kv_indices_p, last_page_len_p = _make_paged_kv_metadata(
        args.prefill_seq_lens, args.page_size, args.index_dtype, device
    )
    qo_indptr_d = torch.tensor(_make_indptr(decode_q_lens), dtype=args.index_dtype, device=device)
    kv_indptr_d, kv_indices_d, last_page_len_d = _make_paged_kv_metadata(
        args.decode_kv_lens, args.page_size, args.index_dtype, device
    )

    q_p = torch.randn(
        int(qo_indptr_p[-1]),
        args.num_q_heads,
        args.head_dim,
        dtype=args.dtype,
        device=device,
    )
    paged_kv_cache_p = _make_paged_kv_cache(
        int(kv_indptr_p[-1]),
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        args.dtype,
        device,
        args.kv_layout,
    )
    q_d = torch.randn(
        int(qo_indptr_d[-1]),
        args.num_q_heads,
        args.head_dim,
        dtype=args.dtype,
        device=device,
    )
    paged_kv_cache_d = _make_paged_kv_cache(
        int(kv_indptr_d[-1]),
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        args.dtype,
        device,
        args.kv_layout,
    )

    workspace = torch.empty(
        args.workspace_mb * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(workspace, kv_layout=args.kv_layout)

    plan_start = time.perf_counter()
    wrapper.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        pos_encoding_mode=args.pos_encoding_mode,
        window_left=args.window_left,
        q_data_type=args.dtype,
        kv_data_type=args.dtype,
        sm_scale=args.sm_scale,
    )
    torch.cuda.synchronize()
    plan_wall_ms = (time.perf_counter() - plan_start) * 1000.0
    plan_info_p = tuple(int(x) for x in wrapper._plan_info_p)
    plan_info_d = tuple(int(x) for x in wrapper._plan_info_d)

    def run_once():
        return wrapper.run(
            q_p,
            paged_kv_cache_p,
            q_d,
            paged_kv_cache_d,
            causal_p=args.causal_prefill,
            return_lse=args.return_lse,
            use_fp16_qk_reduction=args.use_fp16_qk_reduction,
            enable_pdl=args.enable_pdl,
        )

    total_ms, output = _event_time_ms(run_once, args.warmup, args.iters)

    try:
        flashinfer_version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        flashinfer_version = getattr(flashinfer, "__version__", "unknown")

    return {
        "benchmark": "flashinfer_batch_pod_with_paged_kv_cache",
        "flashinfer_version": flashinfer_version,
        "device": torch.cuda.get_device_name(device),
        "prefill_seq_lens": args.prefill_seq_lens,
        "decode_kv_lens": args.decode_kv_lens,
        "decode_q_lens": decode_q_lens,
        "prefill_batch_size": len(args.prefill_seq_lens),
        "decode_batch_size": len(args.decode_kv_lens),
        "prefill_tokens": sum(args.prefill_seq_lens),
        "decode_query_tokens": sum(decode_q_lens),
        "decode_kv_tokens": sum(args.decode_kv_lens),
        "num_q_heads": args.num_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "page_size": args.page_size,
        "kv_layout": args.kv_layout,
        "dtype": str(args.dtype).replace("torch.", ""),
        "index_dtype": str(args.index_dtype).replace("torch.", ""),
        "causal_prefill": args.causal_prefill,
        "pos_encoding_mode": args.pos_encoding_mode,
        "window_left": args.window_left,
        "sm_scale": args.sm_scale if args.sm_scale is not None else 1.0 / math.sqrt(args.head_dim),
        "workspace_mb": args.workspace_mb,
        "warmup": args.warmup,
        "iters": args.iters,
        "plan_wall_ms": plan_wall_ms,
        "plan_info_p": plan_info_p,
        "plan_info_d": plan_info_d,
        "plan_debug": {
            "prefill_padded_batch_size": plan_info_p[0],
            "prefill_cta_tile_q": plan_info_p[3],
            "prefill_split_kv": bool(plan_info_p[14]),
            "decode_padded_batch_size": plan_info_d[0],
            "decode_cta_tile_q": plan_info_d[3],
            "decode_split_kv": bool(plan_info_d[14]),
        },
        "total_ms": total_ms,
        "output_shape": _tensor_shapes(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FlashInfer BatchPODWithPagedKVCacheWrapper mixed attention."
    )
    parser.add_argument("--prefill-seq-lens", type=_parse_int_list, default=(256, 512, 1024, 2048))
    parser.add_argument("--decode-kv-lens", type=_parse_int_list, default=(256, 512, 1024, 2048))
    parser.add_argument("--decode-q-lens", type=_parse_int_list, default=(64, 128, 64, 32))
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--kv-layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--index-dtype", type=_parse_index_dtype, default=torch.int32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--pos-encoding-mode", default="NONE", choices=("NONE", "ROPE_LLAMA", "ALIBI"))
    parser.add_argument("--window-left", type=int, default=-1)
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument("--causal-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-lse", action="store_true")
    parser.add_argument("--use-fp16-qk-reduction", action="store_true")
    parser.add_argument("--enable-pdl", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    result = run_benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
