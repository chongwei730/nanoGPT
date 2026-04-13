#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SMOKE_ROOT="${SMOKE_ROOT:-$(pwd)/experiment_runs_smoke/serial_halving_multigpu_smoke}"
DATA_ROOT="${DATA_ROOT:-/data/nanogpt_data}"
DATASET_NAME="${DATASET_NAME:-tiny_openwebtext_serial_halving_smoke}"
DATA_DIR="$DATA_ROOT/$DATASET_NAME"
CONFIG_PATH="$SMOKE_ROOT/smoke_config.yaml"
RUN_ROOT="$SMOKE_ROOT/run"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$NPROC_PER_NODE}"
NUM_TRIALS="${NUM_TRIALS:-16}"
MAX_ITERS="${MAX_ITERS:-20}"

# Keep the smoke test small, but still exercise the real serial-halving flow:
# multi-GPU launch, multiple trials, pruning between rungs, checkpoint-based resume,
# per-rung snapshots, and final result promotion.
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

if (( NUM_TRIALS < 2 )); then
  echo "NUM_TRIALS must be >= 2 so the smoke test can exercise pruning." >&2
  exit 1
fi

if (( MAX_ITERS < 2 )); then
  echo "MAX_ITERS must be >= 2." >&2
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

RUNG_BUDGETS="$("$PYTHON_BIN" - "$MAX_ITERS" "$NUM_TRIALS" <<'PY'
import math
import sys

total_iters = int(sys.argv[1])
initial_trials = int(sys.argv[2])
reduction_factor = 4
num_rungs = 1
active_trials = initial_trials
while active_trials > 1:
    active_trials = int(math.ceil(float(active_trials) / float(reduction_factor)))
    num_rungs += 1
budgets = []
for index in range(num_rungs):
    if index == num_rungs - 1:
        budget = total_iters
    else:
        power = num_rungs - 1 - index
        budget = int(math.ceil(float(total_iters) / (reduction_factor ** power)))
    if budgets and budget <= budgets[-1]:
        budget = min(total_iters, budgets[-1] + 1)
    budgets.append(max(1, budget))
budgets[-1] = total_iters
print(" ".join(str(budget) for budget in budgets))
PY
)"
FIRST_RUNG_ITERS="${RUNG_BUDGETS%% *}"

if (( FIRST_RUNG_ITERS <= WARMUP_ITERS )); then
  echo "First rung budget ($FIRST_RUNG_ITERS) must be greater than WARMUP_ITERS ($WARMUP_ITERS)." >&2
  echo "Increase MAX_ITERS, increase NUM_TRIALS, or lower WARMUP_ITERS." >&2
  exit 1
fi

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
  always_save_checkpoint: true
  gradient_accumulation_steps: $GRAD_ACCUM_STEPS
  batch_size: $BATCH_SIZE
  block_size: $BLOCK_SIZE
  n_layer: $MODEL_N_LAYER
  n_head: $MODEL_N_HEAD
  n_embd: $MODEL_N_EMBD
  dropout: 0.0
  bias: false
  weight_decay: 0.1
  warmup_iters: $WARMUP_ITERS
  lr_decay_iters: $MAX_ITERS
  min_lr: 1.0e-5
  max_iters: $MAX_ITERS

hyperparameters:
  learning_rate:
    type: log_uniform
    range: [1.0e-5, 1.0e-3]

task:
  train_metric: train_loss
  test_metric: val_loss
  metric_mode: min
  num_iterations_per_trial: $MAX_ITERS
  max_running_time_per_trial_hours: 0.0

optuna:
  max_study_time_hours: 0.0
  pruning:
    enabled: true

checkpoint:
  save_last: true
YAML

echo "Running serial successive halving smoke test"
echo "Dataset: $DATA_DIR"
echo "Run root: $RUN_ROOT"
echo "Config: $CONFIG_PATH"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "gradient_accumulation_steps: $GRAD_ACCUM_STEPS"
echo "num_trials: $NUM_TRIALS"
echo "max_iters: $MAX_ITERS"
echo "warmup_iters: $WARMUP_ITERS"
echo "rung budgets: $RUNG_BUDGETS"

"$PYTHON_BIN" run_stage1_optuna.py \
  "$CONFIG_PATH" \
  --run-root "$RUN_ROOT" \
  --num-trials "$NUM_TRIALS"

echo
echo "Smoke test complete."
echo "Run result: $RUN_ROOT/serial_halving_result.json"
echo "Shared trials root: $RUN_ROOT/shared_trials"
echo
