#!/usr/bin/env bash
set -euo pipefail

# Focused sweep for imbalanced POD workloads:
# 1) long prefill lengths with short decode KV lengths
# 2) short prefill lengths with long decode KV lengths
#
# Usage:
#   bash pod_attn_tests/benchmark_pod_imbalance.sh
#
# Optional overrides:
#   OUT_DIR=/tmp/pod_imbalance bash pod_attn_tests/benchmark_pod_imbalance.sh
#   WARMUP=10 ITERS=100 bash pod_attn_tests/benchmark_pod_imbalance.sh
#   PYTHON_BIN=python bash pod_attn_tests/benchmark_pod_imbalance.sh

REPO_DIR="${REPO_DIR:-/home/ypzhao/flashinfer-dllm}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/pod_attn_tests/results/imbalance_$(date +%Y%m%d_%H%M%S)}"
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
  decode_first
)

EXECUTION_MODES=(
  prefill_only_pod
  true_prefill_decode_pod
)

# name|prefill_seq_lens|decode_kv_lens|num_q_heads|num_kv_heads|dtype|kv_layout
# Naming:
# - lp_sd: long prefill sequence lengths, short decode KV lengths
# - sp_ld: short prefill sequence lengths, long decode KV lengths
CONFIGS=(
  "lp_sd_b4d4_bf16_nhd|2048,3072,4096,6144|256,384,512,768|16|4|bf16|NHD"
  "lp_sd_b8d8_bf16_nhd|2048,2560,3072,3584,4096,4608,5120,6144|256,320,384,448,512,576,640,768|16|4|bf16|NHD"
  "lp_sd_b8d8_fp16_hnd|2048,2560,3072,3584,4096,4608,5120,6144|256,320,384,448,512,576,640,768|16|4|fp16|HND"
  "sp_ld_b4d4_bf16_nhd|256,384,512,768|2048,3072,4096,6144|16|4|bf16|NHD"
  "sp_ld_b8d8_bf16_nhd|256,320,384,448,512,576,640,768|2048,2560,3072,3584,4096,4608,5120,6144|16|4|bf16|NHD"
  "sp_ld_b8d8_fp16_hnd|256,320,384,448,512,576,640,768|2048,2560,3072,3584,4096,4608,5120,6144|16|4|fp16|HND"
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
    f"{name:42s} {execution_mode:24s} {policy:13s} "
    f"ok={row['correct']} "
    f"fi_ms={row['flashinfer_ms']} "
    f"sdpa_ms={row['chunked_sdpa_ms']} "
    f"speedup={row['speedup']} "
    f"work={row['prefill_work']}:{row['decode_work']}"
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
