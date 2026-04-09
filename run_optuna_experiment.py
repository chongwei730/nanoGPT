import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import yaml
import torch

try:
    import optuna
except ImportError as exc:  # pragma: no cover - import guard for runtime
    raise SystemExit(
        "Optuna is required to run experiments. Install it first, for example with "
        "`pip install optuna` or by updating the project environment."
    ) from exc


TABLE_PATH = os.path.join("docs", "table.txt")
EVAL_LINE_RE = re.compile(r"^step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)$")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a fixed Optuna experiment.")
    parser.add_argument("config", help="Path to the experiment YAML config.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a mapping.")
    return config


def require_keys(mapping, keys, section_name):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing required keys in {section_name}: {missing}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def resolve_nproc_per_node(launch_config):
    raw_value = launch_config.get("nproc_per_node", "auto")
    if raw_value == "auto":
        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            raise ValueError("launch.nproc_per_node=auto requires at least one visible CUDA device.")
        return gpu_count
    nproc = int(raw_value)
    if nproc < 1:
        raise ValueError("launch.nproc_per_node must be >= 1.")
    return nproc


def suggest_value(trial, name, spec):
    spec_type = spec["type"]
    if spec_type == "log_uniform":
        low, high = spec["range"]
        return trial.suggest_float(name, low, high, log=True)
    if spec_type == "uniform":
        low, high = spec["range"]
        return trial.suggest_float(name, low, high)
    if spec_type == "int":
        low, high = spec["range"]
        step = spec.get("step", 1)
        return trial.suggest_int(name, low, high, step=step)
    raise ValueError(f"Unsupported hyperparameter type for {name}: {spec_type}")


def parse_table(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if not parts or parts[0] == "family" or parts[0] == "---":
                continue
            if len(parts) != 8:
                continue
            rows.append(
                {
                    "family": parts[0],
                    "dataset": parts[1],
                    "model_size": parts[2],
                    "tokens_b": parts[3],
                    "train_target": parts[4],
                    "test_target": parts[5],
                    "runtime_hours": parts[6],
                    "notes": parts[7],
                }
            )
    return rows


def parse_optional_float(value):
    if value in {"NA", "", None}:
        return None
    return float(value)


def find_target_row(config):
    experiment = config["experiment"]
    table_rows = parse_table(TABLE_PATH)
    for row in table_rows:
        if (
            row["family"] == experiment["target_family"]
            and row["dataset"] == experiment["target_dataset"]
            and row["model_size"] == experiment["target_model_size"]
        ):
            return row
    raise ValueError(
        "Could not find matching target row in docs/table.txt for "
        f"{experiment['target_family']} / {experiment['target_dataset']} / {experiment['target_model_size']}."
    )


def validate_targets(config, target_row):
    task = config["task"]
    expected_train = parse_optional_float(target_row["train_target"])
    expected_test = parse_optional_float(target_row["test_target"])
    expected_runtime = parse_optional_float(target_row["runtime_hours"])

    if expected_train is not None and float(task["train_target"]) != expected_train:
        raise ValueError(
            f"train_target mismatch: config={task['train_target']} table={expected_train}"
        )
    if expected_test is not None and float(task["test_target"]) != expected_test:
        raise ValueError(
            f"test_target mismatch: config={task['test_target']} table={expected_test}"
        )
    if expected_runtime is not None and float(task["max_running_time_per_trial_hours"]) != expected_runtime:
        raise ValueError(
            "max_running_time_per_trial_hours mismatch: "
            f"config={task['max_running_time_per_trial_hours']} table={expected_runtime}"
        )


def build_pruner(config):
    optuna_cfg = config.get("optuna", {})
    pruning_cfg = optuna_cfg.get("pruning", {})
    if not pruning_cfg.get("enabled", True):
        return optuna.pruners.NopPruner()

    pruner_type = pruning_cfg.get("type", "median")
    if pruner_type != "median":
        raise ValueError(f"Unsupported pruner type: {pruner_type}")
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(pruning_cfg.get("n_startup_trials", 1)),
        n_warmup_steps=int(pruning_cfg.get("n_warmup_steps", 1)),
        interval_steps=int(pruning_cfg.get("interval_steps", 1)),
    )


def build_command(config, trial_dir, trial_id, sampled_params, summary_path, prune_signal_path):
    experiment = config["experiment"]
    task = config["task"]
    checkpoint = config.get("checkpoint", {})
    fixed_args = config.get("fixed_args", {})
    launch = config.get("launch", {})
    wandb_cfg = config.get("wandb", {})

    launch_mode = launch.get("mode", "torchrun")
    if launch_mode != "torchrun":
        raise ValueError(f"Unsupported launch.mode: {launch_mode}")

    nproc_per_node = resolve_nproc_per_node(launch)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
    ]
    master_port = launch.get("master_port")
    if master_port is not None:
        cmd.append(f"--master_port={int(master_port)}")
    cmd.append(experiment["train_script"])
    for key, value in fixed_args.items():
        cmd.append(f"--{key}={value}")
    for key, value in sampled_params.items():
        cmd.append(f"--{key}={value}")
    if wandb_cfg.get("enabled", False):
        project = wandb_cfg.get("project")
        if not project:
            raise ValueError("wandb.project must be set when wandb.enabled is true.")
        run_name = wandb_cfg.get("run_name_prefix", experiment["name"])
        cmd.extend(
            [
                "--wandb_log=True",
                f"--wandb_project={project}",
                f"--wandb_run_name={run_name}_{trial_id}",
                f"--wandb_group={experiment['name']}",
            ]
        )

    cmd.extend(
        [
            f"--out_dir={trial_dir}",
            f"--experiment_name={experiment['name']}",
            f"--trial_id={trial_id}",
            f"--experiment_metric_mode={task['metric_mode']}",
            f"--experiment_train_target_value={task['train_target']}",
            "--experiment_train_target_enabled=True",
            f"--experiment_test_target_value={task['test_target']}",
            "--experiment_test_target_enabled=True",
            f"--max_running_time_hours={task['max_running_time_per_trial_hours']}",
            f"--save_last_checkpoint={checkpoint.get('save_last', True)}",
            f"--experiment_summary_path={summary_path}",
            f"--prune_signal_path={prune_signal_path}",
        ]
    )
    return cmd


def read_summary(summary_path):
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def get_objective_value(summary, metric_name):
    if metric_name != "val_loss":
        raise ValueError(
            "This runner currently expects `task.test_metric: val_loss` because the "
            "training script summarizes `best_val_loss`."
        )
    return float(summary["best_val_loss"])


def get_train_metric_value(summary, metric_name):
    if metric_name != "train_loss":
        raise ValueError(
            "This runner currently expects `task.train_metric: train_loss` because the "
            "training script summarizes `best_train_loss`."
        )
    return float(summary["best_train_loss"])


def write_study_summary(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def stream_trial_process(trial, command, prune_signal_path, log_path):
    reported_steps = set()
    prune_requested = False
    process = subprocess.Popen(
        command,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            match = EVAL_LINE_RE.match(line.strip())
            if match:
                step = int(match.group(1))
                val_loss = float(match.group(3))
                if step not in reported_steps:
                    reported_steps.add(step)
                    trial.report(val_loss, step=step)
                    if trial.should_prune() and not prune_requested:
                        with open(prune_signal_path, "w", encoding="utf-8") as f:
                            f.write("prune\n")
                        prune_requested = True
    returncode = process.wait()
    return returncode, prune_requested


def main():
    args = parse_args()
    config = load_config(args.config)

    require_keys(config, ["experiment", "hyperparameters", "task"], "root")
    require_keys(
        config["experiment"],
        ["name", "train_script", "output_root", "target_family", "target_dataset", "target_model_size"],
        "experiment",
    )
    require_keys(
        config["task"],
        [
            "train_metric",
            "test_metric",
            "metric_mode",
            "train_target",
            "test_target",
            "max_running_time_per_trial_hours",
        ],
        "task",
    )

    target_row = find_target_row(config)
    validate_targets(config, target_row)

    experiment = config["experiment"]
    task = config["task"]
    hyperparameters = config["hyperparameters"]
    checkpoint = config.setdefault("checkpoint", {})
    checkpoint.setdefault("save_last", True)
    launch = config.setdefault("launch", {})
    launch.setdefault("mode", "torchrun")
    launch.setdefault("nproc_per_node", "auto")
    optuna_cfg = config.setdefault("optuna", {})
    max_study_time_hours = float(optuna_cfg.get("max_study_time_hours", task["max_running_time_per_trial_hours"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = ensure_dir(
        os.path.join(experiment["output_root"], f"{experiment['name']}_{timestamp}")
    )
    trials_jsonl_path = os.path.join(experiment_root, "trials.jsonl")
    study_summary_path = os.path.join(experiment_root, "study_summary.json")
    config_snapshot_path = os.path.join(experiment_root, "resolved_config.yaml")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    direction = "minimize" if task["metric_mode"] == "min" else "maximize"
    sampler = optuna.samplers.TPESampler()
    pruner = build_pruner(config)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study_start_time = time.time()
    study_train_reached_time_hours = None
    study_test_reached_time_hours = None
    stop_reason = "study_timeout"

    def objective(trial):
        nonlocal study_train_reached_time_hours
        nonlocal study_test_reached_time_hours
        nonlocal stop_reason

        sampled_params = {
            name: suggest_value(trial, name, spec)
            for name, spec in hyperparameters.items()
        }
        trial_id = f"trial_{trial.number:04d}"
        trial_dir = ensure_dir(os.path.join(experiment_root, trial_id))
        summary_path = os.path.join(trial_dir, "summary.json")
        prune_signal_path = os.path.join(trial_dir, "PRUNE")
        log_path = os.path.join(trial_dir, "trial.log")
        if os.path.exists(prune_signal_path):
            os.remove(prune_signal_path)
        command = build_command(config, trial_dir, trial_id, sampled_params, summary_path, prune_signal_path)

        print(f"[experiment] starting {trial_id} with params {sampled_params}")
        returncode, prune_requested = stream_trial_process(trial, command, prune_signal_path, log_path)
        summary = read_summary(summary_path)
        if summary is None:
            failure_summary = {
                "trial_id": trial_id,
                "params": sampled_params,
                "returncode": returncode,
                "status": "missing_summary",
                "log_path": log_path,
            }
            append_jsonl(trials_jsonl_path, failure_summary)
            raise RuntimeError(
                f"Trial {trial_id} did not produce a summary file at {summary_path}."
            )

        train_objective_value = get_train_metric_value(summary, task["train_metric"])
        test_objective_value = get_objective_value(summary, task["test_metric"])
        train_target_reached = bool(summary.get("train_target_reached", False))
        test_target_reached = bool(summary.get("test_target_reached", False))
        both_targets_reached = bool(summary.get("both_targets_reached", False))

        if train_target_reached and study_train_reached_time_hours is None:
            study_train_reached_time_hours = (time.time() - study_start_time) / 3600.0
        if test_target_reached and study_test_reached_time_hours is None:
            study_test_reached_time_hours = (time.time() - study_start_time) / 3600.0

        trial_record = {
            "trial_id": trial_id,
            "params": sampled_params,
            "train_objective_value": train_objective_value,
            "test_objective_value": test_objective_value,
            "train_target_reached": train_target_reached,
            "test_target_reached": test_target_reached,
            "both_targets_reached": both_targets_reached,
            "trial_train_reached_time_hours": summary.get("trial_train_reached_time_hours"),
            "trial_test_reached_time_hours": summary.get("trial_test_reached_time_hours"),
            "returncode": returncode,
            "summary_path": summary_path,
            "trial_dir": trial_dir,
            "log_path": log_path,
            "termination_reason": summary.get("termination_reason", ""),
            "pruned": bool(summary.get("termination_reason") == "pruned" or prune_requested),
        }
        append_jsonl(trials_jsonl_path, trial_record)

        for key, value in sampled_params.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("summary_path", summary_path)
        trial.set_user_attr("trial_dir", trial_dir)
        trial.set_user_attr("log_path", log_path)
        trial.set_user_attr("termination_reason", summary.get("termination_reason", ""))
        trial.set_user_attr("train_target_reached", train_target_reached)
        trial.set_user_attr("test_target_reached", test_target_reached)
        trial.set_user_attr("both_targets_reached", both_targets_reached)

        if returncode != 0:
            stop_reason = "trial_failed"
            raise RuntimeError(
                f"Trial {trial_id} failed with return code {returncode}."
            )

        if summary.get("termination_reason") == "pruned" or prune_requested:
            raise optuna.TrialPruned(f"Trial {trial_id} pruned.")

        if both_targets_reached:
            stop_reason = "both_targets_reached"
            study.stop()
        return test_objective_value

    study.optimize(objective, timeout=max_study_time_hours * 3600.0)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
    best_trial = study.best_trial if completed_trials else None
    study_summary = {
        "experiment_name": experiment["name"],
        "experiment_root": experiment_root,
        "target_family": experiment["target_family"],
        "target_dataset": experiment["target_dataset"],
        "target_model_size": experiment["target_model_size"],
        "train_target": float(task["train_target"]),
        "test_target": float(task["test_target"]),
        "study_train_reached_time_hours": study_train_reached_time_hours,
        "study_test_reached_time_hours": study_test_reached_time_hours,
        "both_targets_reached": any(
            bool(trial.user_attrs.get("both_targets_reached", False))
            for trial in study.trials
        ),
        "best_trial_number": best_trial.number if best_trial is not None else None,
        "best_params": best_trial.params if best_trial is not None else None,
        "best_value": best_trial.value if best_trial is not None else None,
        "direction": direction,
        "num_trials": len(study.trials),
        "num_completed_trials": len(completed_trials),
        "num_pruned_trials": len(pruned_trials),
        "max_study_time_hours": max_study_time_hours,
        "stop_reason": stop_reason,
    }
    write_study_summary(study_summary_path, study_summary)

    print("[experiment] completed")
    print(json.dumps(study_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
