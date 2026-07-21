#!/usr/bin/env bash
set -euo pipefail

# Sample benchmark sweep for BatchPOD CTA scheduling policies.
#
# Usage:
#   bash pod_attn_tests/benchmark_pod_schedules.sh
#
# Optional overrides:
#   OUT_DIR=/tmp/pod_bench bash pod_attn_tests/benchmark_pod_schedules.sh
#   WARMUP=10 ITERS=100 bash pod_attn_tests/benchmark_pod_schedules.sh
#   PYTHON_BIN=python bash pod_attn_tests/benchmark_pod_schedules.sh
#
# Notes:
# - Defaults to `conda run -n pod python`.
# - Writes one JSON file per run plus summary.csv.
# - This script assumes it is run on a machine/container where the `pod`
#   environment can see CUDA.

REPO_DIR="${REPO_DIR:-/home/ypzhao/flashinfer-dllm}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/pod_attn_tests/results/$(date +%Y%m%d_%H%M%S)}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-30}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  PY_CMD=(conda run -n pod python)
fi

mkdir -p "${OUT_DIR}"
cd "${REPO_DIR}"

POLICIES=(
  count_ratio
  work_aware
  prefill_first
)

EXECUTION_MODES=(
  prefill_only_pod
  true_prefill_decode_pod
)

# name|prefill_seq_lens|decode_kv_lens|num_q_heads|num_kv_heads|dtype|kv_layout
CONFIGS=(
  "medium_b3d2_gqa_fp16_nhd|256,512,1024|1024,2048|16|4|fp16|NHD"
  "large_b6d4_gqa_bf16_nhd|512,768,1024,1280,1536,2048|1024,2048,3072,4096|16|4|bf16|NHD"
  "xl_b8d8_gqa_bf16_nhd|512,768,1024,1280,1536,1792,2048,2560|1024,1536,2048,2560,3072,3584,4096,4608|16|4|bf16|NHD"
  "xl_b8d8_gqa_fp16_hnd|512,768,1024,1280,1536,1792,2048,2560|1024,1536,2048,2560,3072,3584,4096,4608|16|4|fp16|HND"
)

COMMON_ARGS=(
  pod_attn_tests/flx_pod.py
  --block-length 64
  --head-dim 128
  --page-size 64
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --compare-correctness
  --benchmark-reference
)

echo "Writing results to: ${OUT_DIR}"
echo "name,execution_mode,policy,correct,flashinfer_ms,chunked_sdpa_ms,speedup,prefill_weight,decode_weight,prefill_work,decode_work,num_sdpa_steps,plan_wall_ms" \
  > "${OUT_DIR}/summary.csv"

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r name prefill_lens decode_lens num_q_heads num_kv_heads dtype kv_layout <<< "${config}"

  for execution_mode in "${EXECUTION_MODES[@]}"; do
    for policy in "${POLICIES[@]}"; do
      json_path="${OUT_DIR}/${name}_${execution_mode}_${policy}.json"
      echo "RUN name=${name} mode=${execution_mode} policy=${policy}"

      "${PY_CMD[@]}" "${COMMON_ARGS[@]}" \
        --prefill-seq-lens "${prefill_lens}" \
        --decode-kv-lens "${decode_lens}" \
        --num-q-heads "${num_q_heads}" \
        --num-kv-heads "${num_kv_heads}" \
        --dtype "${dtype}" \
        --kv-layout "${kv_layout}" \
        --pod-schedule-policy "${policy}" \
        --pod-execution-mode "${execution_mode}" \
        > "${json_path}"

      "${PY_CMD[@]}" - "${json_path}" "${name}" "${execution_mode}" "${policy}" "${OUT_DIR}/summary.csv" <<'PY'
import csv
import json
import sys

json_path, name, execution_mode, policy, csv_path = sys.argv[1:]
with open(json_path) as f:
    data = json.load(f)

row = {
    "name": name,
    "execution_mode": execution_mode,
    "policy": policy,
    "correct": data.get("correctness", {}).get("passed"),
    "flashinfer_ms": data.get("flashinfer_total_ms"),
    "chunked_sdpa_ms": data.get("chunked_sdpa_total_ms"),
    "speedup": data.get("speedup_vs_chunked_sdpa"),
    "prefill_weight": data.get("prefill_schedule_weight"),
    "decode_weight": data.get("decode_schedule_weight"),
    "prefill_work": data.get("prefill_work_estimate"),
    "decode_work": data.get("decode_work_estimate"),
    "num_sdpa_steps": data.get("num_sdpa_steps"),
    "plan_wall_ms": data.get("plan_wall_ms"),
}

print(
    f"{name:22s} {execution_mode:24s} {policy:13s} "
    f"ok={row['correct']} "
    f"fi_ms={row['flashinfer_ms']} "
    f"sdpa_ms={row['chunked_sdpa_ms']} "
    f"speedup={row['speedup']}"
)

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writerow(row)
PY
    done
  done
done

echo "Done."
echo "Summary: ${OUT_DIR}/summary.csv"
