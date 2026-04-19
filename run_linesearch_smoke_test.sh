#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SMOKE_ROOT="${SMOKE_ROOT:-$(pwd)/experiment_runs_smoke/linesearch_multigpu_smoke}"
DATA_ROOT="${DATA_ROOT:-/work/nvme/bgop/cchen47/nanogpt_data}"
DATASET_NAME="${DATASET_NAME:-tiny_openwebtext_linesearch_smoke}"
DATA_DIR="$DATA_ROOT/$DATASET_NAME"
RUN_ROOT="$SMOKE_ROOT/run"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_gpt2.py}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$NPROC_PER_NODE}"
MAX_ITERS="${MAX_ITERS:-20}"

# Keep the smoke test small, but still exercise the real line-search flow:
# multi-GPU launch, line-search closures over fixed batches, training, eval,
# checkpoint output, records output, and stage-two-compatible result writing.
MODEL_N_LAYER="${MODEL_N_LAYER:-2}"
MODEL_N_HEAD="${MODEL_N_HEAD:-2}"
MODEL_N_EMBD="${MODEL_N_EMBD:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"

if (( GRAD_ACCUM_STEPS < 1 )); then
  echo "GRAD_ACCUM_STEPS must be >= 1" >&2
  exit 1
fi

if (( GRAD_ACCUM_STEPS % NPROC_PER_NODE != 0 )); then
  echo "GRAD_ACCUM_STEPS ($GRAD_ACCUM_STEPS) must be divisible by NPROC_PER_NODE ($NPROC_PER_NODE)" >&2
  exit 1
fi

if (( MAX_ITERS < 2 )); then
  echo "MAX_ITERS must be >= 2." >&2
  exit 1
fi

if (( WARMUP_ITERS < 0 )); then
  echo "WARMUP_ITERS must be >= 0." >&2
  exit 1
fi

mkdir -p "$SMOKE_ROOT"
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
mkdir -p "$DATA_DIR"

"$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import pickle
import sys
from pathlib import Path

import numpy as np

dataset_dir = Path(sys.argv[1])
vocab_size = 64
train_tokens = (np.arange(16384, dtype=np.uint16) % vocab_size).astype(np.uint16)
val_tokens = (np.arange(8192, dtype=np.uint16) % vocab_size).astype(np.uint16)
train_tokens.tofile(dataset_dir / "train.bin")
val_tokens.tofile(dataset_dir / "val.bin")
with open(dataset_dir / "meta.pkl", "wb") as f:
    pickle.dump({"vocab_size": vocab_size}, f)
PY

LINESEARCH_INTERVAL="$("$PYTHON_BIN" - "$MAX_ITERS" <<'PY'
import sys

max_iters = int(sys.argv[1])
print(max(1, int(max_iters * 0.1)))
PY
)"

echo "Running line-search smoke test"
echo "Dataset: $DATA_DIR"
echo "Run root: $RUN_ROOT"
echo "Train config: $TRAIN_CONFIG"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "gradient_accumulation_steps: $GRAD_ACCUM_STEPS"
echo "max_iters: $MAX_ITERS"
echo "warmup_iters: $WARMUP_ITERS"
echo "derived_linesearch_interval: $LINESEARCH_INTERVAL"

"$PYTHON_BIN" run_linesearch_stage2.py \
  --run-root "$RUN_ROOT" \
  --train-script "train_linesearch.py" \
  --config-path "$TRAIN_CONFIG" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --experiment-name "linesearch_multigpu_smoke" \
  --trial-id "stage2_final" \
  --save-last-checkpoint true \
  -- \
  "$TRAIN_CONFIG" \
  --dataset="$DATASET_NAME" \
  --compile=False \
  --eval_interval=1 \
  --eval_iters=1 \
  --log_interval=1 \
  --always_save_checkpoint=True \
  --gradient_accumulation_steps="$GRAD_ACCUM_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --block_size="$BLOCK_SIZE" \
  --n_layer="$MODEL_N_LAYER" \
  --n_head="$MODEL_N_HEAD" \
  --n_embd="$MODEL_N_EMBD" \
  --dropout=0.0 \
  --bias=False \
  --weight_decay=0.1 \
  --warmup_iters="$WARMUP_ITERS" \
  --lr_decay_iters="$MAX_ITERS" \
  --min_lr=1.0e-5 \
  --max_iters="$MAX_ITERS"

echo
echo "Smoke test complete."
echo "Stage2 result: $RUN_ROOT/stage2_result.json"
echo "Stage2 manifest: $RUN_ROOT/stage2_manifest.json"
echo "Final summary: $RUN_ROOT/final/summary.json"
echo "Final records: $RUN_ROOT/final/records.jsonl"
echo
