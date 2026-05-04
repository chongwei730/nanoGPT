#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_SIZE="${MODEL_SIZE:-130m}"
SMOKE_ROOT="${SMOKE_ROOT:-$(pwd)/experiment_runs_smoke/tuning_batch_llama_${MODEL_SIZE}_smoke}"
RUN_ROOT="${RUN_ROOT:-$SMOKE_ROOT/run}"
CONFIG_PATH="${CONFIG_PATH:-$SMOKE_ROOT/smoke_config.yaml}"

NUM_TRIALS="${NUM_TRIALS:-12}"
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
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-tuning_batch_llama60m_smoke}"
    ;;
  130m)
    TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_llama_130m.py}"
    LLAMA_CONFIG_PATH="${LLAMA_CONFIG_PATH:-llama_config/llama_130m.json}"
    TARGET_MODEL_SIZE="${TARGET_MODEL_SIZE:-130M}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-tuning_batch_llama130m_smoke}"
    ;;
  350m)
    TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_llama_350m.py}"
    LLAMA_CONFIG_PATH="${LLAMA_CONFIG_PATH:-llama_config/llama_350m.json}"
    TARGET_MODEL_SIZE="${TARGET_MODEL_SIZE:-350M}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-tuning_batch_llama350m_smoke}"
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

if (( NUM_TRIALS != 12 )); then
  echo "NUM_TRIALS must be exactly 12 for the fixed tuning-batch protocol." >&2
  exit 1
fi

if (( MAX_ITERS < 4 )); then
  echo "MAX_ITERS must be >= 4 so the quarter-budget tuning stage is meaningful." >&2
  exit 1
fi

mkdir -p "$SMOKE_ROOT"
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

TUNING_ITERS="$(( (MAX_ITERS + 3) / 4 ))"

if (( TUNING_ITERS <= WARMUP_ITERS )); then
  echo "TUNING_ITERS ($TUNING_ITERS) must be greater than WARMUP_ITERS ($WARMUP_ITERS)." >&2
  echo "Increase MAX_ITERS or lower WARMUP_ITERS." >&2
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
    range: [1.0e-6, 1.0e-3]
  scheduler:
    type: categorical
    values: [cosine_10pct, inv_sqrt, linear_10pct]

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

echo "Running LLaMA fixed tuning-batch smoke test"
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
echo "tuning_iters: $TUNING_ITERS"
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
