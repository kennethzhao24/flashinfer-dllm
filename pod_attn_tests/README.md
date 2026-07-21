# POD Attention Benchmarks

This directory compares two ways of executing mixed block-diffusion prefill and
decode attention with FlashInfer POD:

- `prefill_only_pod`: route both prefill blocks and semantic decode blocks
  through the POD prefill side. This is the original performance baseline.
- `true_prefill_decode_pod`: route prefill blocks through the prefill side and
  semantic decode blocks through the decode side, allowing the POD scheduler to
  distinguish the two types of work.

The benchmark uses GQA with 16 query heads and 4 KV heads. A semantic decode
request produces one block of `block_length` query tokens in one benchmark
execution. Prefill uses a virtual-prefix representation so each query block sees
the same KV prefix as the block-causal SDPA reference.

## Running the Benchmarks

Activate the `pod` Conda environment and run from the repository root:

```bash
conda activate pod
cd /home/ypzhao/flashinfer-dllm
bash pod_attn_tests/benchmark_pod_schedules.sh
bash pod_attn_tests/benchmark_pod_imbalance.sh
```

`benchmark_pod_schedules.sh` covers general medium-to-XL batch configurations.
`benchmark_pod_imbalance.sh` covers:

- `lp_sd`: long prefill sequences with short decode KV sequences.
- `sp_ld`: short prefill sequences with long decode KV sequences.

Each sweep compares `count_ratio`, `work_aware`, `prefill_first`, and, where
enabled, `decode_first`. `count_ratio` is the original scheduling baseline.

## Latest Results

The native multi-token decode sweep is in
`results/imbalance_20260721_135259`. All 48 configurations passed correctness.
The previous pre-native-decode sweep is retained in
`results/imbalance_20260721_131351` for comparison.

| Workload | Prefill-only POD | Native true decode POD | Previous true decode | Native vs previous |
| --- | ---: | ---: | ---: |
| Long prefill, short decode, B4/D4 | 2.635 ms | 2.716 ms | 2.709 ms | 1.00x |
| Long prefill, short decode, B8/D8 | 5.535 ms | 5.659 ms | 5.704 ms | 1.01x |
| Short prefill, long decode, B4/D4 | 0.307 ms | 0.175 ms | 0.784 ms | 4.46x |
| Short prefill, long decode, B8/D8 | 0.455 ms | 0.332 ms | 1.593 ms | 4.80x |

Native decode is substantially faster in decode-dominated workloads and is now
faster than prefill-only POD in those cases. In prefill-dominated workloads, the
native path is effectively unchanged because prefill remains the critical path.

The best policies were generally `count_ratio` for long-prefill cases,
`prefill_first` for the B4/D4 short-prefill case, and `decode_first` for the
B8/D8 short-prefill cases.

## Main Bottleneck

The native true-decode path uses one ragged multi-token decode request per
semantic decode block. The decode planner selects the CTA tile size from the
query length and GQA group size, allowing neighboring query rows to reuse KV
data.

For the short-prefill/long-decode B8/D8 configuration, this produces:

- 8 semantic decode requests.
- Approximately 136 decode CTAs with 4 KV heads and a tile size of 128; the
  planner enables split-KV for this long-context case.
- KV reuse across the query rows in each decode block.

The previous single-token representation used 512 decode slots and 2,048
decode CTAs for this case. Native block decode reduces the CTA count by roughly
15x while preserving the same attention semantics.

The individual JSON files contain the authoritative results, including
`decode_representation`, `decode_cta_tile_q`, `decode_cta_count`, and
`correctness`. The current sweep's `summary.csv` contains only its header due to
an issue with the shell script's `conda run` heredoc append step.

## Optimization Roadmap

### 1. Native multi-token decode (implemented)

The focused GPU test validates direct POD decode at `q_len={1,8,64}` against
standalone FlashInfer batch prefill attention using identical Q, paged KV
metadata, and mask semantics. The benchmark sweep additionally validates the
64-token block used by the block-diffusion workload against SDPA.

The POD launch now dispatches the decode kernel using the same tile size selected
by the decode planner. The generated JIT module instantiates all prefill/decode
tile pairs from `{16,64,128}`.

Success criteria:

- Direct `q_len=64` decode passes the existing correctness tolerances.
- Decode-side padded batch size and CTA count scale with query tiles, not query
  tokens.
- True-decode latency approaches the prefill-only result in decode-heavy cases.

### 2. Block-decode kernel specialization (implemented)

A diffusion decode block is computationally closer to prefill than to
autoregressive one-token decode. Keep block decode in the logical `DECODE`
scheduling queue, but execute it with prefill-style tensor-core tiling and KV
reuse. This preserves independent prefill/decode scheduling without forcing a
64-token block through a `q_len=1`-oriented path.

### 3. Preserve ratios in work-aware scheduling

The current scheduler divides work estimates by their GCD and independently
clamps each weight to 16. This can erase the intended ratio:

- `66896:15616` becomes `16:16`, instead of approximately `4.3:1`.
- `1152:124928` becomes `9:16`, instead of approximately `1:108`.

Normalize both estimates into a shared fixed tag budget instead. With 32 total
tags, the examples above would become approximately `26:6` and `1:31`.

This change should be evaluated after native block decode. It can improve CTA
ordering and tail behavior, but it does not reduce the total decode work.

### 4. Measure the fused resource-envelope cost

The fused POD launch uses the maximum thread count and dynamic shared-memory
requirement of the prefill and decode kernel traits. A lightweight operation can
therefore reserve resources required only by the heavier operation and lose
occupancy.

Compare:

- The current fused POD kernel.
- Specialized prefill and decode kernels on separate CUDA streams.
- A persistent POD kernel with separate prefill and decode work queues.
- Fused variants with more closely matched prefill/decode tile resources.

The two-stream implementation provides a useful lower bound for deciding
whether additional fused-kernel complexity is justified.

### 5. Add length-aware CTA ordering

Within each operation, order or bucket CTAs using estimated KV work:

- Distribute long-KV CTAs across SMs early to reduce the final tail.
- Keep CTAs sharing a KV prefix close enough to benefit from L2 locality.
- Select work using remaining-cost estimates rather than only periodic static
  tags.

### 6. Re-evaluate split-KV for long decode contexts

The current measured plans do not use split-KV. Native block decode will reduce
the number of decode CTAs significantly, so long contexts may then need more KV
parallelism. Benchmark split factors `1`, `2`, and `4` for 8K and 16K decode KV
lengths, including the reduction overhead in the result.

## Measurement Guidance

For kernel optimization, report standalone prefill time, standalone decode time,
fused POD time, CTA counts, and padded batch sizes. The speedup over chunked SDPA
is useful for the end-to-end benchmark, but it includes repeated SDPA decode work
and does not isolate POD kernel efficiency.

Use profiler counters to distinguish the likely causes of decode overhead:

- DRAM read bytes and L2 hit rate for repeated KV traversal.
- Active warps and achieved occupancy for the fused resource envelope.
- Tensor-core utilization for native block decode.
- CTA duration distribution and the final completion tail for scheduling work.

Recommended implementation order: native multi-token correctness, block-decode
specialization, corrected work weights, fused-resource experiments, length-aware
ordering, and split-KV tuning.
