#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

TEST_ROOT="${TEST_ROOT:-$(pwd)/exp_log/nanogpt_staged_test}"
PYTHON_BIN="${PYTHON:-python3}"
DATA_DIR="$TEST_ROOT/data/tiny_openwebtext"
RUN_ROOT="$TEST_ROOT/runs"
CONFIG_PATH="$TEST_ROOT/tiny_stage_test.yaml"

rm -rf "$TEST_ROOT"
mkdir -p "$DATA_DIR" "$RUN_ROOT"

"$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import pickle
import sys
from pathlib import Path

import numpy as np

dataset_dir = Path(sys.argv[1])
vocab_size = 32
train_tokens = (np.arange(512, dtype=np.uint16) % vocab_size).astype(np.uint16)
val_tokens = (np.arange(256, dtype=np.uint16) % vocab_size).astype(np.uint16)
train_tokens.tofile(dataset_dir / "train.bin")
val_tokens.tofile(dataset_dir / "val.bin")
with open(dataset_dir / "meta.pkl", "wb") as f:
    pickle.dump({"vocab_size": vocab_size}, f)
PY

DATASET_OVERRIDE="$("$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import os
import sys

print(os.path.relpath(sys.argv[1], "/scratch.global/chen8596/nanogpt_data"))
PY
)"

cat > "$CONFIG_PATH" <<YAML
experiment:
  name: tiny_staged_test
  train_script: train.py
  output_root: $RUN_ROOT
  target_family: GPT
  target_dataset: OpenWebText
  target_model_size: 124M

launch:
  mode: python

fixed_args:
  dataset: $DATASET_OVERRIDE
  device: cpu
  compile: false
  dtype: float32
  gradient_accumulation_steps: 1
  batch_size: 2
  block_size: 8
  n_layer: 1
  n_head: 1
  n_embd: 16
  dropout: 0.0
  bias: false
  weight_decay: 0.0
  eval_interval: 1
  log_interval: 1
  eval_iters: 1
  max_iters: 1
  warmup_iters: 0
  lr_decay_iters: 1
  min_lr: 1.0e-4
  always_save_checkpoint: false

hyperparameters:
  learning_rate:
    type: log_uniform
    range: [1.0e-4, 3.0e-4]

task:
  train_metric: train_loss
  test_metric: val_loss
  metric_mode: min
  max_running_time_per_trial_hours: 0.001

optuna:
  max_study_time_hours: 0.001
  max_study_time_hours_levels: [0.001, 0.002]
  pruning:
    enabled: false

checkpoint:
  save_last: true
YAML

"$PYTHON_BIN" run_stage1_optuna.py "$CONFIG_PATH" --run-root "$RUN_ROOT"
"$PYTHON_BIN" run_stage2_final.py "$RUN_ROOT/stage1_manifest.json"

"$PYTHON_BIN" - "$RUN_ROOT" "$TEST_ROOT" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
test_root = Path(sys.argv[2])
stage1_manifest = json.loads((run_root / "stage1_manifest.json").read_text(encoding="utf-8"))
stage2_manifest = json.loads((test_root / "stage2_manifest.json").read_text(encoding="utf-8"))

assert stage1_manifest["stage"] == "stage1_multilevel"
assert stage2_manifest["stage"] == "stage2_multilevel"
assert len(stage1_manifest["levels"]) == 2
assert len(stage2_manifest["results"]) == 2

for level_entry, stage2_result in zip(stage1_manifest["levels"], stage2_manifest["results"]):
    stage1_result = json.loads(Path(level_entry["stage1_result_path"]).read_text(encoding="utf-8"))
    stage1_summary = json.loads(Path(stage1_result["study_summary_path"]).read_text(encoding="utf-8"))
    assert "selection_metric" in stage1_result
    assert stage1_result["selection_metric"] == "train_loss"
    assert "all_records_path" in stage1_result
    assert Path(stage1_result["all_records_path"]).exists()
    assert Path(level_entry["stage1_records_path"]).exists()
    assert "train_target" not in stage1_summary
    assert "test_target" not in stage1_summary
    assert stage2_result["records_path"]
    assert Path(stage2_result["records_path"]).exists()
    assert "best_train_loss" in stage2_result
    assert "best_val_loss" in stage2_result
    assert "train_target_reached" not in stage2_result
    assert "test_target_reached" not in stage2_result
PY

# persist artifacts to a stable location (mimics real setting where runs are kept)
mkdir -p "$STORE_ROOT"
RUN_ID="tiny_staged_test_$(date +%Y%m%d_%H%M%S)"
cp -a "$RUN_ROOT" "$STORE_ROOT/$RUN_ID"

echo
echo "Tiny staged test complete."
echo "Run root: $RUN_ROOT"
echo "Stage-one manifest: $RUN_ROOT/stage1_manifest.json"
echo "Stage-two manifest: $TEST_ROOT/stage2_manifest.json"
echo "Stored copy: $STORE_ROOT/$RUN_ID"
