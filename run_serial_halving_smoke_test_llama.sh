#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_SIZE="${MODEL_SIZE:-130m}"
SMOKE_ROOT="${SMOKE_ROOT:-$(pwd)/experiment_runs_smoke/serial_halving_llama_${MODEL_SIZE}_smoke}"
RUN_ROOT="${RUN_ROOT:-$SMOKE_ROOT/run}"
CONFIG_PATH="${CONFIG_PATH:-$SMOKE_ROOT/smoke_config.yaml}"

NUM_TRIALS="${NUM_TRIALS:-8}"
MAX_ITERS="${MAX_ITERS:-20}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$NPROC_PER_NODE}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
EVAL_ITERS="${EVAL_ITERS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-256}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
TOKENIZER_NAME="${TOKENIZER_NAME:-t5-base}"
DATASET_NAME="${DATASET_NAME:-allenai/c4}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-en}"

case "$MODEL_SIZE" in
  60m)
    TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_llama_60m.py}"
    LLAMA_CONFIG_PATH="${LLAMA_CONFIG_PATH:-llama_config/llama_60m.json}"
    TARGET_MODEL_SIZE="${TARGET_MODEL_SIZE:-60M}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-serial_halving_llama60m_smoke}"
    ;;
  130m)
    TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_llama_130m.py}"
    LLAMA_CONFIG_PATH="${LLAMA_CONFIG_PATH:-llama_config/llama_130m.json}"
    TARGET_MODEL_SIZE="${TARGET_MODEL_SIZE:-130M}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-serial_halving_llama130m_smoke}"
    ;;
  350m)
    TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_llama_350m.py}"
    LLAMA_CONFIG_PATH="${LLAMA_CONFIG_PATH:-llama_config/llama_350m.json}"
    TARGET_MODEL_SIZE="${TARGET_MODEL_SIZE:-350M}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-serial_halving_llama350m_smoke}"
    ;;
  *)
    echo "Unsupported MODEL_SIZE: $MODEL_SIZE. Use 60m, 130m, or 350m." >&2
    exit 1
    ;;
esac

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
  name: $EXPERIMENT_NAME
  train_script: train_llama.py
  train_config: $TRAIN_CONFIG
  output_root: $SMOKE_ROOT
  target_family: LLaMA
  target_dataset: C4
  target_model_size: $TARGET_MODEL_SIZE
  skip_table_validation: true

launch:
  mode: torchrun
  nproc_per_node: $NPROC_PER_NODE

fixed_args:
  dataset: $DATASET_NAME
  dataset_config_name: $DATASET_CONFIG_NAME
  tokenizer_name: $TOKENIZER_NAME
  llama_config_path: $LLAMA_CONFIG_PATH
  compile: false
  dtype: float32
  gradient_accumulation_steps: $GRAD_ACCUM_STEPS
  batch_size: $BATCH_SIZE
  block_size: $BLOCK_SIZE
  max_length: $BLOCK_SIZE
  weight_decay: 0.1
  eval_interval: $EVAL_INTERVAL
  eval_iters: $EVAL_ITERS
  log_interval: $LOG_INTERVAL
  max_iters: $MAX_ITERS
  warmup_iters: $WARMUP_ITERS
  lr_decay_iters: $MAX_ITERS
  min_lr: 1.0e-5
  always_save_checkpoint: true
  shuffle_buffer_size: $SHUFFLE_BUFFER_SIZE
  dataloader_num_workers: $DATALOADER_NUM_WORKERS

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

echo "Running LLaMA serial successive halving smoke test"
echo "Model size: $MODEL_SIZE"
echo "Train config: $TRAIN_CONFIG"
echo "LLaMA config: $LLAMA_CONFIG_PATH"
echo "Dataset: $DATASET_NAME/$DATASET_CONFIG_NAME (streaming)"
echo "Run root: $RUN_ROOT"
echo "Config: $CONFIG_PATH"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "gradient_accumulation_steps: $GRAD_ACCUM_STEPS"
echo "batch_size: $BATCH_SIZE"
echo "max_iters: $MAX_ITERS"
echo "warmup_iters: $WARMUP_ITERS"
echo "num_trials: $NUM_TRIALS"
echo "rung budgets: $RUNG_BUDGETS"
echo

"$PYTHON_BIN" run_stage1_optuna.py \
  "$CONFIG_PATH" \
  --run-root "$RUN_ROOT" \
  --num-trials "$NUM_TRIALS"

echo
echo "Smoke test complete."
echo "Run result: $RUN_ROOT/serial_halving_result.json"
echo "Shared trials root: $RUN_ROOT/shared_trials"
echo
