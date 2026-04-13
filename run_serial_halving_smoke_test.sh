#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SMOKE_ROOT="${SMOKE_ROOT:-$(pwd)/experiment_runs_smoke/serial_halving_multigpu_smoke}"
DATASET_NAME="${DATASET_NAME:-tiny_openwebtext_serial_halving_smoke}"
DATA_DIR="/scratch.global/chen8596/nanogpt_data/$DATASET_NAME"
CONFIG_PATH="$SMOKE_ROOT/smoke_config.yaml"
RUN_ROOT="$SMOKE_ROOT/run"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$NPROC_PER_NODE}"

if (( GRAD_ACCUM_STEPS < 1 )); then
  echo "GRAD_ACCUM_STEPS must be >= 1" >&2
  exit 1
fi

if (( GRAD_ACCUM_STEPS % NPROC_PER_NODE != 0 )); then
  echo "GRAD_ACCUM_STEPS ($GRAD_ACCUM_STEPS) must be divisible by NPROC_PER_NODE ($NPROC_PER_NODE)" >&2
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
train_tokens = (np.arange(4096, dtype=np.uint16) % vocab_size).astype(np.uint16)
val_tokens = (np.arange(2048, dtype=np.uint16) % vocab_size).astype(np.uint16)
train_tokens.tofile(dataset_dir / "train.bin")
val_tokens.tofile(dataset_dir / "val.bin")
with open(dataset_dir / "meta.pkl", "wb") as f:
    pickle.dump({"vocab_size": vocab_size}, f)
PY

cat > "$CONFIG_PATH" <<YAML
experiment:
  name: serial_halving_multigpu_smoke
  train_script: train.py
  train_config: config/train_gpt2.py
  output_root: $SMOKE_ROOT
  target_family: GPT
  target_dataset: OpenWebText
  target_model_size: 124M

launch:
  mode: torchrun
  nproc_per_node: $NPROC_PER_NODE

fixed_args:
  dataset: $DATASET_NAME
  compile: false
  wandb_log: false
  eval_interval: 1
  eval_iters: 1
  log_interval: 1
  always_save_checkpoint: false
  gradient_accumulation_steps: $GRAD_ACCUM_STEPS
  batch_size: 2
  block_size: 8
  n_layer: 1
  n_head: 1
  n_embd: 16
  dropout: 0.0
  bias: false
  weight_decay: 0.0
  warmup_iters: 0
  lr_decay_iters: 8
  min_lr: 1.0e-5
  max_iters: 8

hyperparameters:
  learning_rate:
    type: log_uniform
    range: [1.0e-5, 3.0e-4]

task:
  train_metric: train_loss
  test_metric: val_loss
  metric_mode: min
  num_iterations_per_trial: 8
  max_running_time_per_trial_hours: 0.0

optuna:
  max_study_time_hours: 0.002
  max_study_time_hours_levels: [0.001, 0.002]
  pruning:
    enabled: false

checkpoint:
  save_last: true
YAML

echo "Running serial successive halving smoke test"
echo "Dataset: $DATA_DIR"
echo "Run root: $RUN_ROOT"
echo "Config: $CONFIG_PATH"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "gradient_accumulation_steps: $GRAD_ACCUM_STEPS"

"$PYTHON_BIN" run_stage1_optuna.py "$CONFIG_PATH" --run-root "$RUN_ROOT"

echo
echo "Smoke test complete."
echo "Run result: $RUN_ROOT/serial_halving_result.json"
echo "Shared trials root: $RUN_ROOT/shared_trials"
echo
echo "Suggested checks:"
echo "  1. Confirm shared_trials contains ckpt_last.pt for completed trials."
echo "  2. Confirm later levels resumed surviving trials from shared_trials/<trial_id>/."
echo "  3. Confirm each level directory contains result.json and selected_trial/."
