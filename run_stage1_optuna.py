import argparse
import json
import math
import os
import random
import shutil
import time
from datetime import datetime

import yaml

import run_optuna_experiment


PROTOCOL_IDENTIFIER = "fixed_tuning_batch_v1"
TOTAL_CANDIDATE_COUNT = 12
BATCH_COUNT = 3
BATCH_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed 12-candidate tuning-batch learning-rate search protocol."
    )
    parser.add_argument("config", help="Path to the experiment YAML config.")
    parser.add_argument(
        "--run-root",
        default="",
        help="Optional shared output root for all batches.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Compatibility flag. When provided it must be exactly 12 for this fixed protocol.",
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=2,
        help="Compatibility flag retained for existing launch scripts. Ignored by this fixed protocol.",
    )
    parser.add_argument(
        "--num-iterations-per-trial",
        type=int,
        default=None,
        help="Optional override for task.num_iterations_per_trial and the corresponding training max_iters.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help=(
            "Optional 1-based batch index to run in isolation. "
            "Use 1 for the first 4 candidates, 2 for the middle 4, and 3 for the last 4."
        ),
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_config_root():
    return os.path.abspath("config")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def controller_state_path(run_root):
    return os.path.join(run_root, "controller_state.json")


def batch_name(batch_index):
    return f"batch_{int(batch_index) + 1:02d}"


def resolve_requested_batch_indices(args):
    batch_index_value = getattr(args, "batch_index", None)
    if batch_index_value is None:
        return [index for index in range(BATCH_COUNT)]
    batch_index = int(batch_index_value)
    if batch_index < 1 or batch_index > BATCH_COUNT:
        raise ValueError(
            f"--batch-index must be between 1 and {BATCH_COUNT}, got {batch_index_value}."
        )
    return [batch_index - 1]


def tuning_iteration_budget(total_iters):
    if int(total_iters) < 1:
        raise ValueError("total_iters must be >= 1.")
    return int(math.ceil(float(total_iters) / float(BATCH_SIZE)))


def require_single_controller():
    if os.environ.get("RANK") not in (None, "0") or os.environ.get("WORLD_SIZE") not in (None, "1"):
        raise ValueError(
            "run_stage1_optuna.py must be launched as a single controller process. "
            "Do not use torchrun or multi-rank launchers for this script."
        )
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    slurm_procid = int(os.environ.get("SLURM_PROCID", "0"))
    if slurm_ntasks > 1 or slurm_procid > 0:
        raise ValueError(
            "run_stage1_optuna.py must be launched with a single SLURM task. "
            "Use one task and let it spawn torchrun for each trial."
        )


def trial_id_from_number(number):
    return f"trial_{number:04d}"


def copy_if_exists(src, dst):
    if src and os.path.exists(src):
        shutil.copyfile(src, dst)
        return True
    return False


def copy_tree_files(file_map):
    for src, dst in file_map:
        parent = os.path.dirname(dst)
        if parent:
            ensure_dir(parent)
        copy_if_exists(src, dst)


def snapshot_trial_artifacts(batch_root, trial_state, phase_name):
    snapshot_dir = ensure_dir(
        os.path.join(batch_root, "trial_snapshots", trial_state["trial_id"], phase_name)
    )
    summary_snapshot = os.path.join(snapshot_dir, "summary.json")
    records_snapshot = os.path.join(snapshot_dir, "records.jsonl")
    log_snapshot = os.path.join(snapshot_dir, "trial.log")
    copy_if_exists(trial_state["summary_path"], summary_snapshot)
    copy_if_exists(trial_state["records_path"], records_snapshot)
    copy_if_exists(trial_state["log_path"], log_snapshot)
    return {
        "summary_path": os.path.abspath(summary_snapshot),
        "records_path": os.path.abspath(records_snapshot),
        "log_path": os.path.abspath(log_snapshot),
    }


def build_discrete_lr_values(spec, default_num_choices=50):
    spec_type = spec["type"]
    if spec_type == "discrete":
        values = list(spec.get("values", []))
        if values:
            return [float(value) for value in values]
        num_choices = int(spec.get("num_choices", default_num_choices))
        scale = spec.get("scale", "linear")
    elif spec_type in {"log_uniform", "uniform"}:
        num_choices = int(spec.get("num_choices", default_num_choices))
        scale = "log" if spec_type == "log_uniform" else "linear"
    else:
        raise ValueError(
            "Serial tuning-batch candidate construction requires a learning-rate hyperparameter "
            f"with type in {{'log_uniform', 'uniform', 'discrete'}}, got {spec_type!r}."
        )

    if num_choices < 1:
        raise ValueError("Discrete learning-rate construction requires num_choices >= 1.")
    low, high = spec["range"]
    if num_choices == 1:
        return [float(low)]
    if scale == "log":
        log_low = math.log(float(low))
        log_high = math.log(float(high))
        return [
            math.exp(log_low + (log_high - log_low) * index / float(num_choices - 1))
            for index in range(num_choices)
        ]
    return [
        float(low) + (float(high) - float(low)) * index / float(num_choices - 1)
        for index in range(num_choices)
    ]


def build_ordered_trial_candidates(hyperparameters):
    tuned_param_name = run_optuna_experiment.resolve_tuned_lr_param_name(hyperparameters)
    lr_values = build_discrete_lr_values(hyperparameters[tuned_param_name], default_num_choices=50)
    scheduler_values = ["cosine_10pct"]
    if "scheduler" in hyperparameters:
        scheduler_spec = hyperparameters["scheduler"]
        if scheduler_spec.get("type") != "categorical":
            raise ValueError("Scheduler hyperparameter must have type 'categorical'.")
        scheduler_values = list(scheduler_spec.get("values", []))
        if not scheduler_values:
            raise ValueError("Scheduler hyperparameter must define at least one categorical value.")

    full_candidates = []
    for lr_value in lr_values:
        for scheduler_value in scheduler_values:
            params = {tuned_param_name: float(lr_value)}
            if "scheduler" in hyperparameters:
                params["scheduler"] = scheduler_value
            full_candidates.append(params)

    sample_size = min(16, len(full_candidates))
    sample_seed = 1337
    sampled_indices = random.Random(sample_seed).sample(range(len(full_candidates)), sample_size)
    ordered_candidates = []
    for candidate_rank, candidate_index in enumerate(sampled_indices):
        ordered_candidates.append(
            {
                "candidate_rank": int(candidate_rank),
                "candidate_index": int(candidate_index),
                "params": dict(full_candidates[candidate_index]),
            }
        )

    return {
        "tuned_param_name": tuned_param_name,
        "lr_values": [float(value) for value in lr_values],
        "scheduler_values": scheduler_values,
        "full_candidate_count": len(full_candidates),
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "ordered_candidates": ordered_candidates,
    }


def validate_candidate_plan(candidate_plan, source_name):
    required_keys = {
        "tuned_param_name",
        "lr_values",
        "scheduler_values",
        "full_candidate_count",
        "sample_size",
        "sample_seed",
        "ordered_candidates",
    }
    missing = sorted(required_keys.difference(candidate_plan.keys()))
    if missing:
        raise ValueError(f"{source_name} is missing required keys: {missing}")
    return candidate_plan


def canonical_fixed_candidate_pool(payload):
    if payload is None:
        return None
    selected_candidate_order = payload.get("selected_candidate_order")
    if selected_candidate_order is None:
        selected_candidate_order = payload.get("ordered_candidates")
    if selected_candidate_order is None:
        raise ValueError(
            "Fixed candidate pool payload must define either "
            "'selected_candidate_order' or 'ordered_candidates'."
        )
    return {
        "tuned_param_name": payload["tuned_param_name"],
        "lr_values": payload["lr_values"],
        "scheduler_values": payload["scheduler_values"],
        "full_candidate_count": int(payload["full_candidate_count"]),
        "sample_size": int(payload["sample_size"]),
        "sample_seed": int(payload["sample_seed"]),
        "selected_candidate_order": selected_candidate_order,
    }


def default_candidate_plan_config_path(config_path):
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join(repo_config_root(), "fixed_candidate_pools", f"{config_stem}.yaml")


def resolve_candidate_plan_config_path(config_path, config):
    serial_halving_cfg = config.get("serial_halving", {})
    configured_path = serial_halving_cfg.get("fixed_candidate_pool_config", "")
    if configured_path:
        if os.path.isabs(configured_path):
            return configured_path
        return os.path.abspath(os.path.join(repo_config_root(), configured_path))
    return os.path.abspath(default_candidate_plan_config_path(config_path))


def load_persisted_candidate_plan(config_path, config):
    candidate_plan_config_path = resolve_candidate_plan_config_path(config_path, config)
    if os.path.exists(candidate_plan_config_path):
        payload = read_yaml(candidate_plan_config_path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Candidate plan config must be a mapping: {candidate_plan_config_path}"
            )
        return validate_candidate_plan(payload, candidate_plan_config_path), candidate_plan_config_path

    serial_halving_cfg = config.get("serial_halving", {})
    embedded_candidate_plan = serial_halving_cfg.get("fixed_candidate_pool")
    if embedded_candidate_plan:
        return (
            validate_candidate_plan(embedded_candidate_plan, "serial_halving.fixed_candidate_pool"),
            candidate_plan_config_path,
        )
    return None, candidate_plan_config_path


def persist_candidate_plan_if_missing(config_path, config, candidate_plan):
    persisted_candidate_plan, candidate_plan_config_path = load_persisted_candidate_plan(config_path, config)
    if persisted_candidate_plan is not None:
        return

    serial_halving_cfg = config.setdefault("serial_halving", {})
    relative_candidate_plan_path = os.path.relpath(candidate_plan_config_path, repo_config_root())
    serial_halving_cfg["fixed_candidate_pool_config"] = relative_candidate_plan_path
    serial_halving_cfg.pop("fixed_candidate_pool", None)
    write_yaml(config_path, config)
    ensure_dir(os.path.dirname(candidate_plan_config_path))
    write_yaml(candidate_plan_config_path, candidate_plan)
    print(f"[tuning-batch] wrote fixed candidate pool config: {candidate_plan_config_path}")


def get_or_create_persisted_candidate_plan(config_path, config):
    persisted, _ = load_persisted_candidate_plan(config_path, config)
    if persisted is not None:
        return persisted
    candidate_plan = build_ordered_trial_candidates(config["hyperparameters"])
    persist_candidate_plan_if_missing(config_path, config, candidate_plan)
    return candidate_plan


def objective_value_from_summary(task, summary):
    train_value = run_optuna_experiment.get_metric_value(summary, task["train_metric"])
    test_value = run_optuna_experiment.get_metric_value(summary, task["test_metric"])
    return train_value, test_value


def metric_sort_key(metric_mode, value):
    return value if metric_mode == "min" else -value


def choose_best_trial(trials, metric_mode, value_key):
    if not trials:
        raise ValueError("Expected at least one trial to choose from.")
    eligible_trials = [trial for trial in trials if trial.get(value_key) is not None]
    if not eligible_trials:
        raise ValueError("No completed trials are available for selection.")
    ranked = sorted(
        eligible_trials,
        key=lambda trial: (
            metric_sort_key(metric_mode, trial[value_key]),
            trial["trial_number"],
        ),
    )
    return ranked[0]


def state_trials_by_id(state):
    return {trial["trial_id"]: trial for trial in state["trials"]}


def trials_for_batch(state, batch_index):
    return [
        trial
        for trial in sorted(state["trials"], key=lambda entry: entry["trial_number"])
        if int(trial["batch_index"]) == int(batch_index)
    ]


def save_controller_state(run_root, state):
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(controller_state_path(run_root), state)


def update_total_running_time(state, session_start_time):
    base_hours = float(state.get("completed_running_time_hours", 0.0))
    session_hours = max(0.0, (time.time() - session_start_time) / 3600.0)
    total_hours = base_hours + session_hours
    state["total_running_time_hours"] = total_hours
    return total_hours


def current_best_completed_winner(state):
    completed = list(state.get("completed_full_runs", []))
    if not completed:
        return None
    metric_mode = state["metric_mode"]
    ranked = sorted(
        completed,
        key=lambda item: (
            metric_sort_key(metric_mode, item["winner_final_metric"]),
            item["batch_index"],
        ),
    )
    return ranked[0]


def public_result_from_state(state):
    next_batch_cursor = int(state.get("next_batch_cursor", 0))
    batch_indices_to_run = [int(index) for index in state.get("batch_indices_to_run", [])]
    next_batch_index = None
    if next_batch_cursor < len(batch_indices_to_run):
        next_batch_index = int(batch_indices_to_run[next_batch_cursor])
    payload = {
        "schema_version": 1,
        "stage": "fixed_tuning_batch",
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
        "total_running_time_hours": float(state.get("total_running_time_hours", 0.0)),
        "config_path": state["config_path"],
        "run_root": state["run_root"],
        "next_batch_index": next_batch_index,
        "next_rung_index": next_batch_index,
        "protocol_identifier": state["protocol_identifier"],
        "tuning_batch": {
            "protocol_identifier": state["protocol_identifier"],
            "total_candidate_count": int(state["total_candidate_count"]),
            "requested_candidate_count": int(state.get("requested_candidate_count", state["total_candidate_count"])),
            "batch_count": int(state["batch_count"]),
            "batch_size": int(state["batch_size"]),
            "tuning_iters": int(state["tuning_iters"]),
            "total_training_iters": int(state["total_training_iters"]),
            "requested_batch_indices": state.get("requested_batch_indices", [1, 2, 3]),
            "fixed_candidate_pool": state["fixed_candidate_pool"],
        },
        "completed_batches": state["completed_batches"],
        "completed_rungs": state["completed_batches"],
        "batch_winner_trial_ids": state["batch_winner_trial_ids"],
        "completed_full_runs": state["completed_full_runs"],
        "current_best_completed_winner": state.get("current_best_completed_winner"),
    }
    if state.get("final_results"):
        payload["results"] = state["final_results"]
    return payload


def write_public_result(run_root, state):
    write_json(os.path.join(run_root, "serial_halving_result.json"), public_result_from_state(state))


def validate_protocol_args(args):
    num_trials_value = getattr(args, "num_trials", None)
    if num_trials_value is not None and int(num_trials_value) != TOTAL_CANDIDATE_COUNT:
        raise ValueError(
            f"This fixed protocol requires exactly {TOTAL_CANDIDATE_COUNT} trials, got {num_trials_value}."
        )
    resolve_requested_batch_indices(args)


def initialize_controller_state(run_root, args, config, tuning_iters, total_training_iters):
    run_root = os.path.abspath(run_root)
    candidate_plan = get_or_create_persisted_candidate_plan(args.config, config)
    ordered_candidates = list(candidate_plan["ordered_candidates"])
    selected_batch_indices = resolve_requested_batch_indices(args)
    if len(ordered_candidates) < TOTAL_CANDIDATE_COUNT:
        raise ValueError(
            f"The fixed tuning-batch workflow requires at least {TOTAL_CANDIDATE_COUNT} ordered candidates, "
            f"but only {len(ordered_candidates)} are available."
        )
    shared_trials_root = ensure_dir(os.path.join(run_root, "shared_trials"))
    trials = []
    for pool_position, candidate in enumerate(ordered_candidates[:TOTAL_CANDIDATE_COUNT]):
        sampled_params = dict(candidate["params"])
        trial_number = int(candidate["candidate_rank"])
        trial_id = trial_id_from_number(trial_number)
        batch_index = pool_position // BATCH_SIZE
        if batch_index not in selected_batch_indices:
            continue
        trial_dir = ensure_dir(os.path.join(shared_trials_root, trial_id))
        batch_candidate_index = pool_position % BATCH_SIZE
        trials.append(
            {
                "trial_number": trial_number,
                "candidate_index": int(candidate["candidate_index"]),
                "candidate_pool_position": int(pool_position),
                "trial_id": trial_id,
                "params": sampled_params,
                "trial_dir": trial_dir,
                "summary_path": os.path.join(trial_dir, "summary.json"),
                "records_path": os.path.join(trial_dir, "records.jsonl"),
                "log_path": os.path.join(trial_dir, "trial.log"),
                "prune_signal_path": os.path.join(trial_dir, "PRUNE"),
                "batch_index": int(batch_index),
                "batch_candidate_index": int(batch_candidate_index),
                "tuning_completed": False,
                "full_completed": False,
                "completed_iters": 0,
                "tuning_objective_value": None,
                "tuning_test_value": None,
                "final_objective_value": None,
                "final_test_value": None,
                "last_summary": None,
                "tuning_record": None,
                "full_record": None,
            }
        )
    timestamp = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
        "config_path": os.path.abspath(args.config),
        "run_root": run_root,
        "completed_running_time_hours": 0.0,
        "total_running_time_hours": 0.0,
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "total_candidate_count": int(TOTAL_CANDIDATE_COUNT),
        "requested_candidate_count": int(len(selected_batch_indices) * BATCH_SIZE),
        "batch_count": int(BATCH_COUNT),
        "batch_size": int(BATCH_SIZE),
        "tuning_iters": int(tuning_iters),
        "total_training_iters": int(total_training_iters),
        "metric_mode": config["task"]["metric_mode"],
        "selection_metric": config["task"]["train_metric"],
        "fixed_candidate_pool": canonical_fixed_candidate_pool(candidate_plan),
        "requested_batch_indices": [int(index) + 1 for index in selected_batch_indices],
        "batch_indices_to_run": [int(index) for index in selected_batch_indices],
        "next_batch_cursor": 0,
        "trials": trials,
        "completed_batches": [],
        "batch_winner_trial_ids": [],
        "completed_full_runs": [],
        "current_best_completed_winner": None,
        "final_results": [],
    }


def load_or_initialize_controller_state(run_root, args, config, tuning_iters, total_training_iters):
    candidate_plan = get_or_create_persisted_candidate_plan(args.config, config)
    selected_batch_indices = resolve_requested_batch_indices(args)
    if len(candidate_plan["ordered_candidates"]) < TOTAL_CANDIDATE_COUNT:
        raise ValueError(
            f"The fixed tuning-batch workflow requires at least {TOTAL_CANDIDATE_COUNT} ordered candidates, "
            f"but only {len(candidate_plan['ordered_candidates'])} are available."
        )
    state_path = controller_state_path(run_root)
    if os.path.exists(state_path):
        state = read_json(state_path)
        if os.path.abspath(args.config) != state["config_path"]:
            raise ValueError(
                f"Config path mismatch for resumed run: current={os.path.abspath(args.config)!r} "
                f"saved={state['config_path']!r}"
            )
        if state.get("protocol_identifier") != PROTOCOL_IDENTIFIER:
            raise ValueError("Saved controller state does not use the fixed tuning-batch protocol.")
        if int(state["total_candidate_count"]) != TOTAL_CANDIDATE_COUNT:
            raise ValueError("Saved controller state has an invalid total candidate count.")
        if int(state["batch_count"]) != BATCH_COUNT or int(state["batch_size"]) != BATCH_SIZE:
            raise ValueError("Saved controller state has incompatible batch dimensions.")
        if int(state["tuning_iters"]) != int(tuning_iters):
            raise ValueError("Configured tuning iteration budget does not match saved controller state.")
        if int(state["total_training_iters"]) != int(total_training_iters):
            raise ValueError("Configured total training iterations do not match saved controller state.")
        state_batch_indices = state.get("batch_indices_to_run")
        if state_batch_indices is None:
            state_batch_indices = [index for index in range(BATCH_COUNT)]
            state["batch_indices_to_run"] = state_batch_indices
        if [int(index) for index in state_batch_indices] != [int(index) for index in selected_batch_indices]:
            raise ValueError("Requested batch indices do not match saved controller state.")
        state.setdefault(
            "requested_batch_indices",
            [int(index) + 1 for index in state["batch_indices_to_run"]],
        )
        state.setdefault(
            "requested_candidate_count",
            int(len(state["batch_indices_to_run"]) * BATCH_SIZE),
        )
        state.setdefault("next_batch_cursor", 0)
        if canonical_fixed_candidate_pool(state.get("fixed_candidate_pool")) != canonical_fixed_candidate_pool(candidate_plan):
            raise ValueError(
                "Persisted fixed candidate pool in the config does not match the saved "
                "controller state for this run."
            )
        print(
            f"[tuning-batch] resuming run from {run_root}: "
            f"next_batch_cursor={state['next_batch_cursor']}"
        )
        return state

    if os.path.exists(run_root) and os.listdir(run_root):
        raise RuntimeError(
            f"Run root {run_root!r} is non-empty but has no controller_state.json. "
            "Cannot safely resume this directory."
        )
    print(
        f"[tuning-batch] initializing new run at {run_root}: "
        f"total_candidates={TOTAL_CANDIDATE_COUNT} requested_batches="
        f"{[int(index) + 1 for index in selected_batch_indices]}"
    )
    return initialize_controller_state(
        run_root=run_root,
        args=args,
        config=config,
        tuning_iters=tuning_iters,
        total_training_iters=total_training_iters,
    )


def resolve_resume_checkpoint_path(trial_state):
    summary = trial_state.get("last_summary") or {}
    candidate_paths = [
        summary.get("best_checkpoint_path", ""),
        summary.get("last_checkpoint_path", ""),
        os.path.join(trial_state["trial_dir"], "ckpt.pt"),
        os.path.join(trial_state["trial_dir"], "ckpt_last.pt"),
    ]
    seen = set()
    for path in candidate_paths:
        normalized = os.path.abspath(path) if path else ""
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized
    raise FileNotFoundError(
        f"Selected batch winner {trial_state['trial_id']} has no resumable checkpoint in "
        f"{trial_state['trial_dir']}."
    )


def run_trial_phase(config, phase_name, batch_index, target_iters, trial_state, batch_root, all_records_path):
    summary_path = trial_state["summary_path"]
    prune_signal_path = trial_state["prune_signal_path"]
    tuned_param_name = run_optuna_experiment.resolve_tuned_lr_param_name(config["hyperparameters"])
    if os.path.exists(prune_signal_path):
        os.remove(prune_signal_path)

    init_from = "resume" if phase_name == "full_training" else "scratch"
    if init_from == "resume":
        resolve_resume_checkpoint_path(trial_state)
    command = run_optuna_experiment.build_command(
        config=config,
        trial_dir=trial_state["trial_dir"],
        trial_id=trial_state["trial_id"],
        sampled_params=trial_state["params"],
        summary_path=summary_path,
        prune_signal_path=prune_signal_path,
        num_iterations_override=target_iters,
        init_from=init_from,
        stop_at_eval_boundary=True,
    )

    returncode, _ = run_optuna_experiment.stream_process(
        command=command,
        log_path=trial_state["log_path"],
        record_paths=[trial_state["records_path"], all_records_path],
        record_context={
            "phase": phase_name,
            "trial_id": trial_state["trial_id"],
            "hyperparameter_name": tuned_param_name,
            "hyperparameter_value": trial_state["params"][tuned_param_name],
        },
        trial=None,
        prune_signal_path=prune_signal_path,
    )
    summary = run_optuna_experiment.read_summary(summary_path)
    if summary is None:
        raise RuntimeError(
            f"Trial {trial_state['trial_id']} did not produce a summary at {summary_path}."
        )
    if returncode != 0:
        raise RuntimeError(
            f"Trial {trial_state['trial_id']} failed during {phase_name} of batch {batch_index} "
            f"with return code {returncode}."
        )

    train_value, test_value = objective_value_from_summary(config["task"], summary)
    snapshot_paths = snapshot_trial_artifacts(batch_root, trial_state, phase_name)
    record = {
        "trial_id": trial_state["trial_id"],
        "trial_number": int(trial_state["trial_number"]),
        "batch_index": int(batch_index),
        "params": trial_state["params"],
        "selection_metric": config["task"]["train_metric"],
        "train_objective_value": train_value,
        "test_objective_value": test_value,
        "returncode": returncode,
        "summary_path": snapshot_paths["summary_path"],
        "trial_dir": os.path.abspath(trial_state["trial_dir"]),
        "log_path": snapshot_paths["log_path"],
        "records_path": snapshot_paths["records_path"],
        "termination_reason": summary.get("termination_reason", ""),
        "phase": phase_name,
        "target_iters": int(target_iters),
        "completed_iters": int(summary.get("iter_num", 0)),
        "init_from": init_from,
    }

    trial_state["last_summary"] = summary
    trial_state["completed_iters"] = int(summary.get("iter_num", 0))
    if phase_name == "tuning":
        trial_state["tuning_completed"] = True
        trial_state["tuning_objective_value"] = train_value
        trial_state["tuning_test_value"] = test_value
        trial_state["tuning_record"] = record
    else:
        trial_state["full_completed"] = True
        trial_state["final_objective_value"] = train_value
        trial_state["final_test_value"] = test_value
        trial_state["full_record"] = record
    return record


def write_batch_trial_records(batch_root, trial_records):
    trials_jsonl_path = os.path.join(batch_root, "trials.jsonl")
    trial_records_path = os.path.join(batch_root, "trial_records.json")
    write_jsonl(trials_jsonl_path, trial_records)
    write_json(trial_records_path, trial_records)
    return os.path.abspath(trials_jsonl_path), os.path.abspath(trial_records_path)


def write_selected_trial_artifacts(batch_root, selected_record, selected_root_name):
    selected_root = ensure_dir(os.path.join(batch_root, selected_root_name))
    selected_summary_path = os.path.join(selected_root, "summary.json")
    selected_records_path = os.path.join(selected_root, "records.jsonl")
    selected_log_path = os.path.join(selected_root, "trial.log")
    copy_if_exists(selected_record["summary_path"], selected_summary_path)
    copy_if_exists(selected_record["records_path"], selected_records_path)
    copy_if_exists(selected_record["log_path"], selected_log_path)
    return {
        "summary_path": os.path.abspath(selected_summary_path),
        "records_path": os.path.abspath(selected_records_path),
        "log_path": os.path.abspath(selected_log_path),
        "selected_root": os.path.abspath(selected_root),
    }


def build_stage1_result_payload(
    config,
    config_path,
    batch_root,
    tuning_summary_path,
    tuning_records_paths,
    tuning_selected_paths,
    winner_trial,
    tuning_records,
    total_running_time_hours,
    tuning_iters,
):
    experiment = config["experiment"]
    task = config["task"]
    tuned_param_name = run_optuna_experiment.resolve_tuned_lr_param_name(config["hyperparameters"])
    winner_record = winner_trial["tuning_record"]
    return {
        "schema_version": 1,
        "stage": "stage1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_running_time_hours": float(total_running_time_hours),
        "config_path": os.path.abspath(config_path),
        "config_snapshot_path": os.path.abspath(os.path.join(batch_root, "resolved_config.yaml")),
        "experiment_name": experiment["name"],
        "experiment_root": os.path.abspath(batch_root),
        "stage1_root": os.path.abspath(batch_root),
        "trials_jsonl_path": tuning_records_paths[0],
        "study_summary_path": os.path.abspath(tuning_summary_path),
        "all_records_path": os.path.abspath(os.path.join(batch_root, "all_records.jsonl")),
        "max_running_time_per_trial_hours": float(task.get("max_running_time_per_trial_hours", 0.0)),
        "max_study_time_hours": 0.0,
        "selection_metric": task["train_metric"],
        "tuned_hyperparameter_name": tuned_param_name,
        "best_params": winner_trial["params"],
        "best_hyperparameter_value": float(winner_trial["params"][tuned_param_name]),
        "best_learning_rate": float(winner_trial["params"][tuned_param_name]),
        "best_trial_number": int(winner_trial["trial_number"]),
        "best_value": winner_record["train_objective_value"],
        "best_train_value": winner_record["train_objective_value"],
        "best_test_value": winner_record["test_objective_value"],
        "selected_trial_id": winner_trial["trial_id"],
        "selected_summary_path": tuning_selected_paths["summary_path"],
        "selected_records_path": tuning_selected_paths["records_path"],
        "selected_log_path": tuning_selected_paths["log_path"],
        "selected_trial_dir": os.path.abspath(winner_trial["trial_dir"]),
        "num_trials": len(tuning_records),
        "num_completed_trials": len(tuning_records),
        "num_pruned_trials": len(tuning_records) - 1,
        "num_iterations_per_trial": int(tuning_iters),
        "stop_reason": "tuning_batch_completed",
    }


def build_stage2_result_payload(
    stage1_result_path,
    config_path,
    stage2_root,
    final_dir,
    final_summary,
    selected_trial_dir,
    total_training_iters,
    total_running_time_hours,
    max_running_time_per_trial_hours,
):
    return {
        "schema_version": 1,
        "stage": "stage2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_running_time_hours": float(total_running_time_hours),
        "stage1_result_path": os.path.abspath(stage1_result_path),
        "config_path": os.path.abspath(config_path),
        "stage2_root": os.path.abspath(stage2_root),
        "final_dir": os.path.abspath(final_dir),
        "summary_path": os.path.abspath(os.path.join(final_dir, "summary.json")),
        "log_path": os.path.abspath(os.path.join(final_dir, "stage2.log")),
        "records_path": os.path.abspath(os.path.join(final_dir, "records.jsonl")),
        "loaded_learning_rate": run_optuna_experiment.load_learning_rate_from_run(
            summary=final_summary,
            run_dir=selected_trial_dir,
        ),
        "num_iterations_per_trial": int(total_training_iters),
        "max_study_time_hours": 0.0,
        "max_running_time_per_trial_hours": float(max_running_time_per_trial_hours),
        "returncode": 0,
        "best_train_loss": float(final_summary["best_train_loss"]),
        "best_val_loss": float(final_summary["best_val_loss"]),
        "termination_reason": final_summary.get("termination_reason"),
        "stage2_forward_backward_hours": final_summary.get(
            "forward_backward_hours", final_summary.get("wall_clock_hours")
        ),
        "elapsed_wall_clock_hours": final_summary.get("elapsed_wall_clock_hours"),
    }


def write_stage2_manifest(batch_root, stage2_result_payload):
    manifest_path = os.path.join(batch_root, "stage2_manifest.json")
    manifest = {
        "schema_version": 1,
        "stage": "stage2_multilevel",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": [stage2_result_payload],
    }
    write_json(manifest_path, manifest)
    return os.path.abspath(manifest_path)


def write_latest_legacy_outputs(run_root, stage1_payload, stage2_payload, batch_result):
    write_json(os.path.join(run_root, "stage1_result.json"), stage1_payload)
    write_json(os.path.join(run_root, "stage2_result.json"), stage2_payload)
    write_json(os.path.join(run_root, "latest_batch_result.json"), batch_result)

    stage2_root = ensure_dir(os.path.join(run_root, "stage2"))
    final_dir = ensure_dir(os.path.join(stage2_root, "final"))
    copy_tree_files(
        [
            (stage2_payload["summary_path"], os.path.join(final_dir, "summary.json")),
            (stage2_payload["records_path"], os.path.join(final_dir, "records.jsonl")),
            (stage2_payload["log_path"], os.path.join(final_dir, "stage2.log")),
        ]
    )


def build_batch_result(
    config,
    config_path,
    state,
    batch_index,
    batch_root,
    batch_trials,
    winner_trial,
    tuning_selected_paths,
    final_selected_paths,
    stage1_result_path,
    stage2_result_path,
    total_running_time_hours,
):
    experiment = config["experiment"]
    task = config["task"]
    tuned_param_name = run_optuna_experiment.resolve_tuned_lr_param_name(config["hyperparameters"])
    final_summary = run_optuna_experiment.read_summary(final_selected_paths["summary_path"])
    if final_summary is None:
        raise RuntimeError(
            f"Could not read final selected summary from {final_selected_paths['summary_path']!r}."
        )
    cumulative_best = current_best_completed_winner(state)
    candidate_start = int(batch_index * BATCH_SIZE + 1)
    candidate_end = int(candidate_start + BATCH_SIZE - 1)
    winner_tuning_record = winner_trial["tuning_record"]
    return {
        "experiment_name": experiment["name"],
        "schema_version": 1,
        "stage": "fixed_tuning_batch_result",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_running_time_hours": float(total_running_time_hours),
        "config_path": os.path.abspath(config_path),
        "config_snapshot_path": os.path.abspath(os.path.join(batch_root, "resolved_config.yaml")),
        "batch_root": os.path.abspath(batch_root),
        "rung_root": os.path.abspath(batch_root),
        "batch_name": batch_name(batch_index),
        "rung_name": batch_name(batch_index),
        "batch_index": int(batch_index),
        "rung_index": int(batch_index),
        "batch_candidate_range": [candidate_start, candidate_end],
        "batch_trial_ids": [trial["trial_id"] for trial in batch_trials],
        "tuning_iteration_budget": int(state["tuning_iters"]),
        "rung_target_iters": int(state["tuning_iters"]),
        "target_family": experiment["target_family"],
        "target_dataset": experiment["target_dataset"],
        "target_model_size": experiment["target_model_size"],
        "selection_metric": task["train_metric"],
        "selected_trial_number": int(winner_trial["trial_number"]),
        "selected_trial_id": winner_trial["trial_id"],
        "best_params": winner_trial["params"],
        "winner_hyperparameters": winner_trial["params"],
        "tuned_hyperparameter_name": tuned_param_name,
        "best_hyperparameter_value": float(winner_trial["params"][tuned_param_name]),
        "best_learning_rate": float(winner_trial["params"][tuned_param_name]),
        "best_scheduler": winner_trial["params"].get("scheduler", ""),
        "winner_tuning_metric": winner_tuning_record["train_objective_value"],
        "best_value": winner_trial["final_objective_value"],
        "best_train_value": winner_trial["final_objective_value"],
        "best_test_value": winner_trial["final_test_value"],
        "selected_trial_dir": os.path.abspath(winner_trial["trial_dir"]),
        "selected_artifact_root": final_selected_paths["selected_root"],
        "selected_summary_path": final_selected_paths["summary_path"],
        "selected_records_path": final_selected_paths["records_path"],
        "selected_log_path": final_selected_paths["log_path"],
        "tuning_selected_summary_path": tuning_selected_paths["summary_path"],
        "tuning_selected_records_path": tuning_selected_paths["records_path"],
        "tuning_selected_log_path": tuning_selected_paths["log_path"],
        "direction": "minimize" if task["metric_mode"] == "min" else "maximize",
        "num_trials": len(batch_trials),
        "num_completed_trials": len(batch_trials),
        "num_pruned_trials": len(batch_trials) - 1,
        "num_iterations_per_trial": int(state["total_training_iters"]),
        "max_running_time_per_trial_hours": float(task.get("max_running_time_per_trial_hours", 0.0)),
        "all_records_path": os.path.abspath(os.path.join(batch_root, "all_records.jsonl")),
        "trials_jsonl_path": os.path.abspath(os.path.join(batch_root, "trials.jsonl")),
        "trial_records_path": os.path.abspath(os.path.join(batch_root, "trial_records.json")),
        "stage1_result_path": os.path.abspath(stage1_result_path),
        "stage2_result_path": os.path.abspath(stage2_result_path),
        "termination_reason": final_summary.get("termination_reason"),
        "forward_backward_hours": final_summary.get(
            "forward_backward_hours", final_summary.get("wall_clock_hours")
        ),
        "elapsed_wall_clock_hours": final_summary.get("elapsed_wall_clock_hours"),
        "best_train_loss": float(final_summary["best_train_loss"]),
        "best_val_loss": float(final_summary["best_val_loss"]),
        "winner_final_loss": float(final_summary["best_val_loss"]),
        "cumulative_best_completed_winner_so_far": cumulative_best,
        "stop_reason": "fixed_tuning_batch_completed",
    }


def write_batch_outputs(
    config,
    config_path,
    state,
    batch_index,
    batch_root,
    batch_trials,
    winner_trial,
    total_running_time_hours,
):
    config_snapshot_path = os.path.join(batch_root, "resolved_config.yaml")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    tuning_records = []
    for trial in batch_trials:
        if trial.get("tuning_record") is not None:
            record = dict(trial["tuning_record"])
            record["selected_for_full_training"] = trial["trial_id"] == winner_trial["trial_id"]
            tuning_records.append(record)
    tuning_records_paths = write_batch_trial_records(batch_root, tuning_records)

    tuning_selected_paths = write_selected_trial_artifacts(
        batch_root,
        winner_trial["tuning_record"],
        "selected_trial",
    )
    full_selected_paths = write_selected_trial_artifacts(
        batch_root,
        winner_trial["full_record"],
        os.path.join("stage2", "final"),
    )
    final_dir = os.path.join(batch_root, "stage2", "final")
    final_log_path = os.path.join(final_dir, "stage2.log")
    copy_if_exists(full_selected_paths["log_path"], final_log_path)

    tuning_summary_payload = {
        "schema_version": 1,
        "stage": "tuning_batch_stage1_summary",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_index": int(batch_index),
        "batch_name": batch_name(batch_index),
        "batch_trial_ids": [trial["trial_id"] for trial in batch_trials],
        "tuning_iteration_budget": int(state["tuning_iters"]),
        "winner_trial_id": winner_trial["trial_id"],
        "winner_trial_number": int(winner_trial["trial_number"]),
        "winner_tuning_metric": winner_trial["tuning_objective_value"],
        "winner_params": winner_trial["params"],
        "num_trials": len(tuning_records),
        "all_records_path": os.path.abspath(os.path.join(batch_root, "all_records.jsonl")),
    }
    tuning_summary_path = os.path.join(batch_root, "study_summary.json")
    write_json(tuning_summary_path, tuning_summary_payload)

    stage1_payload = build_stage1_result_payload(
        config=config,
        config_path=config_path,
        batch_root=batch_root,
        tuning_summary_path=tuning_summary_path,
        tuning_records_paths=tuning_records_paths,
        tuning_selected_paths=tuning_selected_paths,
        winner_trial=winner_trial,
        tuning_records=tuning_records,
        total_running_time_hours=total_running_time_hours,
        tuning_iters=state["tuning_iters"],
    )
    stage1_result_path = os.path.join(batch_root, "stage1_result.json")
    write_json(stage1_result_path, stage1_payload)

    final_summary = run_optuna_experiment.read_summary(full_selected_paths["summary_path"])
    if final_summary is None:
        raise RuntimeError(
            f"Could not read final winner summary from {full_selected_paths['summary_path']!r}."
        )
    stage2_payload = build_stage2_result_payload(
        stage1_result_path=stage1_result_path,
        config_path=config_path,
        stage2_root=os.path.join(batch_root, "stage2"),
        final_dir=final_dir,
        final_summary=final_summary,
        selected_trial_dir=winner_trial["trial_dir"],
        total_training_iters=state["total_training_iters"],
        total_running_time_hours=total_running_time_hours,
        max_running_time_per_trial_hours=config["task"].get("max_running_time_per_trial_hours", 0.0),
    )
    stage2_result_path = os.path.join(batch_root, "stage2_result.json")
    write_json(stage2_result_path, stage2_payload)
    write_stage2_manifest(batch_root, stage2_payload)

    batch_result = build_batch_result(
        config=config,
        config_path=config_path,
        state=state,
        batch_index=batch_index,
        batch_root=batch_root,
        batch_trials=batch_trials,
        winner_trial=winner_trial,
        tuning_selected_paths=tuning_selected_paths,
        final_selected_paths=full_selected_paths,
        stage1_result_path=stage1_result_path,
        stage2_result_path=stage2_result_path,
        total_running_time_hours=total_running_time_hours,
    )
    result_path = os.path.join(batch_root, "result.json")
    write_json(result_path, batch_result)
    write_latest_legacy_outputs(state["run_root"], stage1_payload, stage2_payload, batch_result)
    return result_path, batch_result


def update_state_with_completed_batch(state, batch_index, batch_root, result_path, batch_result, winner_trial):
    completed_entry = {
        "batch_name": batch_name(batch_index),
        "rung_name": batch_name(batch_index),
        "batch_index": int(batch_index),
        "rung_index": int(batch_index),
        "tuning_iteration_budget": int(state["tuning_iters"]),
        "rung_target_iters": int(state["tuning_iters"]),
        "winner_trial_id": winner_trial["trial_id"],
        "batch_root": os.path.abspath(batch_root),
        "rung_root": os.path.abspath(batch_root),
        "result_path": os.path.abspath(result_path),
    }
    state["completed_batches"] = [
        entry for entry in state["completed_batches"]
        if int(entry["batch_index"]) != int(batch_index)
    ]
    state["completed_batches"].append(completed_entry)
    state["completed_batches"] = sorted(
        state["completed_batches"],
        key=lambda entry: int(entry["batch_index"]),
    )

    winner_summary = run_optuna_experiment.read_summary(batch_result["selected_summary_path"])
    if winner_summary is None:
        raise RuntimeError(
            f"Could not read winner summary from {batch_result['selected_summary_path']!r}."
        )
    completed_full_run = {
        "batch_index": int(batch_index),
        "batch_name": batch_name(batch_index),
        "winner_trial_id": winner_trial["trial_id"],
        "winner_params": winner_trial["params"],
        "winner_final_metric": winner_trial["final_test_value"],
        "winner_final_loss": float(winner_summary["best_val_loss"]),
        "selected_summary_path": batch_result["selected_summary_path"],
        "selected_trial_dir": batch_result["selected_trial_dir"],
    }
    state["completed_full_runs"] = [
        entry for entry in state["completed_full_runs"]
        if int(entry["batch_index"]) != int(batch_index)
    ]
    state["completed_full_runs"].append(completed_full_run)
    state["completed_full_runs"] = sorted(
        state["completed_full_runs"],
        key=lambda entry: int(entry["batch_index"]),
    )

    state["batch_winner_trial_ids"] = [
        trial_id
        for index, trial_id in sorted(
            {
                **{entry["batch_index"]: entry["winner_trial_id"] for entry in state["completed_batches"]},
                int(batch_index): winner_trial["trial_id"],
            }.items()
        )
    ]
    state["final_results"] = [
        result for result in state["final_results"]
        if int(result["batch_index"]) != int(batch_index)
    ]
    state["final_results"].append(batch_result)
    state["final_results"] = sorted(
        state["final_results"],
        key=lambda result: int(result["batch_index"]),
    )
    state["current_best_completed_winner"] = current_best_completed_winner(state)


def run_batch(config, config_path, state, batch_index, session_start_time):
    batch_trials = trials_for_batch(state, batch_index)
    if len(batch_trials) != BATCH_SIZE:
        raise ValueError(
            f"Batch {batch_index} expected exactly {BATCH_SIZE} trials, found {len(batch_trials)}."
        )
    batch_root = ensure_dir(os.path.join(state["run_root"], batch_name(batch_index)))
    all_records_path = os.path.join(batch_root, "all_records.jsonl")

    tuning_records = [
        dict(trial["tuning_record"])
        for trial in batch_trials
        if trial.get("tuning_completed") and trial.get("tuning_record") is not None
    ]
    if tuning_records:
        write_batch_trial_records(batch_root, tuning_records)

    print(
        f"[tuning-batch] batch {batch_index + 1}/{BATCH_COUNT}: "
        f"{len(batch_trials)} trials to {state['tuning_iters']} iterations"
    )
    completed_tuning_ids = {record["trial_id"] for record in tuning_records}
    for trial_state in batch_trials:
        if trial_state["trial_id"] in completed_tuning_ids:
            continue
        print(
            f"[tuning-batch] running {trial_state['trial_id']} from scratch "
            f"to {state['tuning_iters']} iterations"
        )
        tuning_record = run_trial_phase(
            config=config,
            phase_name="tuning",
            batch_index=batch_index,
            target_iters=state["tuning_iters"],
            trial_state=trial_state,
            batch_root=batch_root,
            all_records_path=all_records_path,
        )
        tuning_records.append(dict(tuning_record))
        write_batch_trial_records(batch_root, tuning_records)
        update_total_running_time(state, session_start_time)
        save_controller_state(state["run_root"], state)
        write_public_result(state["run_root"], state)

    winner_trial = choose_best_trial(
        batch_trials,
        metric_mode=config["task"]["metric_mode"],
        value_key="tuning_objective_value",
    )
    resolve_resume_checkpoint_path(winner_trial)
    if not winner_trial.get("full_completed") or winner_trial.get("full_record") is None:
        print(
            f"[tuning-batch] resuming winner {winner_trial['trial_id']} "
            f"to {state['total_training_iters']} iterations"
        )
        run_trial_phase(
            config=config,
            phase_name="full_training",
            batch_index=batch_index,
            target_iters=state["total_training_iters"],
            trial_state=winner_trial,
            batch_root=batch_root,
            all_records_path=all_records_path,
        )
        update_total_running_time(state, session_start_time)
        save_controller_state(state["run_root"], state)
        write_public_result(state["run_root"], state)

    provisional_completed_full_runs = [
        entry for entry in state["completed_full_runs"]
        if int(entry["batch_index"]) != int(batch_index)
    ]
    provisional_completed_full_runs.append(
        {
            "batch_index": int(batch_index),
            "batch_name": batch_name(batch_index),
            "winner_trial_id": winner_trial["trial_id"],
            "winner_params": winner_trial["params"],
            "winner_final_metric": winner_trial["final_test_value"],
            "winner_final_loss": float(winner_trial["last_summary"]["best_val_loss"]),
            "selected_summary_path": winner_trial["full_record"]["summary_path"],
            "selected_trial_dir": winner_trial["trial_dir"],
        }
    )
    original_completed_full_runs = state["completed_full_runs"]
    state["completed_full_runs"] = provisional_completed_full_runs
    state["current_best_completed_winner"] = current_best_completed_winner(state)

    total_running_time_hours = update_total_running_time(state, session_start_time)
    result_path, batch_result = write_batch_outputs(
        config=config,
        config_path=config_path,
        state=state,
        batch_index=batch_index,
        batch_root=batch_root,
        batch_trials=batch_trials,
        winner_trial=winner_trial,
        total_running_time_hours=total_running_time_hours,
    )

    state["completed_full_runs"] = original_completed_full_runs
    update_state_with_completed_batch(
        state=state,
        batch_index=batch_index,
        batch_root=batch_root,
        result_path=result_path,
        batch_result=batch_result,
        winner_trial=winner_trial,
    )
    return batch_result


def main():
    require_single_controller()

    session_start_time = time.time()
    args = parse_args()
    validate_protocol_args(args)

    config = load_config(args.config)
    experiment = config["experiment"]
    task = config["task"]
    hyperparameters = config["hyperparameters"]

    if args.num_iterations_per_trial is not None:
        if int(args.num_iterations_per_trial) < 1:
            raise ValueError("--num-iterations-per-trial must be >= 1.")
        task["num_iterations_per_trial"] = int(args.num_iterations_per_trial)
        fixed_args = config.setdefault("fixed_args", {})
        fixed_args["max_iters"] = int(args.num_iterations_per_trial)
        fixed_args["lr_decay_iters"] = int(args.num_iterations_per_trial)

    run_optuna_experiment.resolve_tuned_lr_param_name(hyperparameters)

    checkpoint = config.setdefault("checkpoint", {})
    checkpoint["save_last"] = True

    if args.run_root:
        run_root = ensure_dir(args.run_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested_batch_indices = resolve_requested_batch_indices(args)
        if len(requested_batch_indices) == 1:
            run_suffix = f"{experiment['name']}_tuning_batch_level_{requested_batch_indices[0] + 1}_{timestamp}"
        else:
            run_suffix = f"{experiment['name']}_tuning_batch_{timestamp}"
        run_root = ensure_dir(
            os.path.join(
                experiment["output_root"],
                run_suffix,
            )
        )

    total_training_iters = int(task["num_iterations_per_trial"])
    tuning_iters = tuning_iteration_budget(total_training_iters)
    state = load_or_initialize_controller_state(
        run_root=run_root,
        args=args,
        config=config,
        tuning_iters=tuning_iters,
        total_training_iters=total_training_iters,
    )
    update_total_running_time(state, session_start_time)
    save_controller_state(run_root, state)
    write_public_result(run_root, state)

    while int(state["next_batch_cursor"]) < len(state["batch_indices_to_run"]):
        batch_cursor = int(state["next_batch_cursor"])
        batch_index = int(state["batch_indices_to_run"][batch_cursor])
        run_batch(
            config=config,
            config_path=args.config,
            state=state,
            batch_index=batch_index,
            session_start_time=session_start_time,
        )
        state["next_batch_cursor"] = batch_cursor + 1
        state["completed_running_time_hours"] = float(update_total_running_time(state, session_start_time))
        state["current_best_completed_winner"] = current_best_completed_winner(state)
        save_controller_state(run_root, state)
        write_public_result(run_root, state)

    print(f"[tuning-batch] completed fixed tuning-batch run at {run_root}")


if __name__ == "__main__":
    main()
