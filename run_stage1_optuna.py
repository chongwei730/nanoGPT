import argparse
import json
import math
import os
import shutil
import time
from datetime import datetime

import yaml

import run_optuna_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run serial successive halving learning-rate search across configured rungs."
    )
    parser.add_argument("config", help="Path to the experiment YAML config.")
    parser.add_argument(
        "--run-root",
        default="",
        help="Optional shared output root for all levels.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Optional initial number of trials for serial successive halving. Rung count is inferred automatically.",
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=4,
        help="Successive halving reduction factor. Defaults to 4.",
    )
    parser.add_argument(
        "--num-iterations-per-trial",
        type=int,
        default=None,
        help="Optional override for task.num_iterations_per_trial and the corresponding training max_iters.",
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


def inferred_rung_count(initial_trial_count, reduction_factor):
    if initial_trial_count < 1:
        raise ValueError("initial_trial_count must be >= 1.")
    if reduction_factor < 2:
        raise ValueError("reduction_factor must be >= 2.")

    num_rungs = 1
    active_trials = int(initial_trial_count)
    while active_trials > 1:
        active_trials = int(math.ceil(float(active_trials) / float(reduction_factor)))
        num_rungs += 1
    return num_rungs


def rung_name(rung_index):
    return f"rung_{int(rung_index):02d}"


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


def rung_iteration_budgets(total_iters, num_levels, reduction_factor):
    budgets = []
    for index in range(num_levels):
        if index == num_levels - 1:
            budget = int(total_iters)
        else:
            power = num_levels - 1 - index
            budget = int(math.ceil(float(total_iters) / (reduction_factor ** power)))
        if budgets and budget <= budgets[-1]:
            budget = min(int(total_iters), budgets[-1] + 1)
        budgets.append(max(1, budget))
    budgets[-1] = int(total_iters)
    return budgets


def copy_if_exists(src, dst):
    if src and os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False


def snapshot_trial_artifacts(rung_root, trial_state):
    snapshot_dir = ensure_dir(os.path.join(rung_root, "trial_snapshots", trial_state["trial_id"]))
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


def sample_trial(study, hyperparameters):
    trial = study.ask()
    params = {
        name: run_optuna_experiment.suggest_value(trial, name, spec)
        for name, spec in hyperparameters.items()
    }
    return trial, params


def objective_value_from_summary(task, summary):
    train_value = run_optuna_experiment.get_metric_value(summary, task["train_metric"])
    test_value = run_optuna_experiment.get_metric_value(summary, task["test_metric"])
    return train_value, test_value


def metric_sort_key(metric_mode, value):
    return value if metric_mode == "min" else -value


def level_records_for_active_trials(active_trials, rung_index):
    records = []
    for trial_state in sorted(active_trials, key=lambda trial: trial["trial_number"]):
        if trial_state["completed_rungs"] > rung_index and trial_state.get("last_level_record"):
            records.append(dict(trial_state["last_level_record"]))
    return records


def write_partial_rung_records(rung_root, trial_records):
    trials_jsonl_path = os.path.join(rung_root, "trials.jsonl")
    trial_records_path = os.path.join(rung_root, "trial_records.json")
    write_jsonl(trials_jsonl_path, trial_records)
    write_json(trial_records_path, trial_records)


def state_trials_by_id(state):
    return {trial["trial_id"]: trial for trial in state["trials"]}


def active_trials_from_state(state):
    trials_by_id = state_trials_by_id(state)
    return [trials_by_id[trial_id] for trial_id in state["active_trial_ids"]]


def save_controller_state(run_root, state):
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(controller_state_path(run_root), state)


def update_total_running_time(state, session_start_time):
    base_hours = float(state.get("completed_running_time_hours", 0.0))
    session_hours = max(0.0, (time.time() - session_start_time) / 3600.0)
    total_hours = base_hours + session_hours
    state["total_running_time_hours"] = total_hours
    return total_hours


def public_result_from_state(state):
    payload = {
        "schema_version": 1,
        "stage": "serial_successive_halving",
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
        "total_running_time_hours": float(state.get("total_running_time_hours", 0.0)),
        "config_path": state["config_path"],
        "run_root": state["run_root"],
        "next_rung_index": int(state["next_rung_index"]),
        "serial_halving": {
            "reduction_factor": int(state["reduction_factor"]),
            "num_rungs": int(state["num_rungs"]),
            "initial_trial_count": int(state["initial_trial_count"]),
            "requested_num_trials": int(state["requested_num_trials"]),
            "rung_target_iters": state["rung_budgets"],
        },
        "completed_rungs": state["completed_rungs"],
    }
    if state.get("final_results"):
        payload["results"] = state["final_results"]
    return payload


def write_public_result(run_root, state):
    write_json(os.path.join(run_root, "serial_halving_result.json"), public_result_from_state(state))


def initialize_controller_state(run_root, args, config, num_rungs, rung_budgets, initial_trial_count, reduction_factor):
    run_root = os.path.abspath(run_root)
    direction = "minimize" if config["task"]["metric_mode"] == "min" else "maximize"
    study = run_optuna_experiment.optuna.create_study(
        direction=direction,
        sampler=run_optuna_experiment.optuna.samplers.TPESampler(),
        pruner=run_optuna_experiment.optuna.pruners.NopPruner(),
    )
    shared_trials_root = ensure_dir(os.path.join(run_root, "shared_trials"))
    trials = []
    for _ in range(initial_trial_count):
        sampled_trial, sampled_params = sample_trial(study, config["hyperparameters"])
        trial_number = int(sampled_trial.number)
        trial_id = trial_id_from_number(trial_number)
        trial_dir = ensure_dir(os.path.join(shared_trials_root, trial_id))
        trials.append(
            {
                "trial_number": trial_number,
                "trial_id": trial_id,
                "params": sampled_params,
                "trial_dir": trial_dir,
                "summary_path": os.path.join(trial_dir, "summary.json"),
                "records_path": os.path.join(trial_dir, "records.jsonl"),
                "log_path": os.path.join(trial_dir, "trial.log"),
                "prune_signal_path": os.path.join(trial_dir, "PRUNE"),
                "completed_rungs": 0,
                "completed_iters": 0,
                "train_objective_value": None,
                "test_objective_value": None,
                "last_summary": None,
                "last_level_record": None,
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
        "num_rungs": int(num_rungs),
        "rung_budgets": [int(budget) for budget in rung_budgets],
        "reduction_factor": int(reduction_factor),
        "initial_trial_count": int(initial_trial_count),
        "requested_num_trials": int(initial_trial_count),
        "next_rung_index": 0,
        "active_trial_ids": [trial["trial_id"] for trial in sorted(trials, key=lambda trial: trial["trial_number"])],
        "trials": trials,
        "completed_rungs": [],
        "final_results": [],
    }


def load_or_initialize_controller_state(run_root, args, config, num_rungs, rung_budgets, initial_trial_count, reduction_factor):
    state_path = controller_state_path(run_root)
    if os.path.exists(state_path):
        state = read_json(state_path)
        if os.path.abspath(args.config) != state["config_path"]:
            raise ValueError(
                f"Config path mismatch for resumed run: current={os.path.abspath(args.config)!r} "
                f"saved={state['config_path']!r}"
            )
        if int(num_rungs) != int(state["num_rungs"]):
            raise ValueError("Configured rung count does not match saved controller state.")
        if [int(budget) for budget in rung_budgets] != [int(budget) for budget in state["rung_budgets"]]:
            raise ValueError("Configured rung iteration budgets do not match saved controller state.")
        if int(initial_trial_count) != int(state["initial_trial_count"]):
            raise ValueError(
                f"Initial num_trials mismatch for resumed run: current={initial_trial_count} "
                f"saved={state['initial_trial_count']}"
            )
        if int(reduction_factor) != int(state["reduction_factor"]):
            raise ValueError(
                f"Reduction factor mismatch for resumed run: current={reduction_factor} "
                f"saved={state['reduction_factor']}"
            )
        print(
            f"[halving] resuming run from {run_root}: "
            f"next_rung_index={state['next_rung_index']} "
            f"active_trials={len(state['active_trial_ids'])}"
        )
        return state

    if os.path.exists(run_root) and os.listdir(run_root):
        raise RuntimeError(
            f"Run root {run_root!r} is non-empty but has no controller_state.json. "
            "Cannot safely resume this directory."
        )
    print(
        f"[halving] initializing new run at {run_root}: "
        f"initial_trials={initial_trial_count}"
    )
    return initialize_controller_state(
        run_root=run_root,
        args=args,
        config=config,
        num_rungs=num_rungs,
        rung_budgets=rung_budgets,
        initial_trial_count=initial_trial_count,
        reduction_factor=reduction_factor,
    )


def run_trial_rung(config, rung_index, target_iters, trial_state, rung_root, all_records_path):
    summary_path = trial_state["summary_path"]
    prune_signal_path = trial_state["prune_signal_path"]
    tuned_param_name = run_optuna_experiment.resolve_tuned_lr_param_name(config["hyperparameters"])
    if os.path.exists(prune_signal_path):
        os.remove(prune_signal_path)

    init_from = "resume" if trial_state["completed_rungs"] else "scratch"
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
            "phase": "serial_successive_halving",
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
            f"Trial {trial_state['trial_id']} failed at rung {rung_index} with return code {returncode}."
        )

    train_value, test_value = objective_value_from_summary(config["task"], summary)
    snapshot_paths = snapshot_trial_artifacts(rung_root, trial_state)
    rung_record = {
        "trial_id": trial_state["trial_id"],
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
        "pruned": False,
        "rung_index": int(rung_index),
        "target_iters": int(target_iters),
        "completed_iters": int(summary.get("iter_num", 0)),
    }

    trial_state["train_objective_value"] = train_value
    trial_state["test_objective_value"] = test_value
    trial_state["last_summary"] = summary
    trial_state["last_level_record"] = rung_record
    trial_state["completed_rungs"] += 1
    trial_state["completed_iters"] = int(summary.get("iter_num", 0))
    return rung_record


def choose_selected_trial(active_trials):
    return sorted(active_trials, key=lambda trial: trial["trial_number"])[-1]


def write_selected_trial_artifacts(rung_root, selected_record):
    selected_root = ensure_dir(os.path.join(rung_root, "selected_trial"))
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


def write_rung_result(
    config,
    config_path,
    rung_index,
    rung_iters,
    rung_root,
    trial_records,
    active_trials,
    pruned_count,
    total_running_time_hours,
):
    experiment = config["experiment"]
    task = config["task"]
    selected_trial = choose_selected_trial(active_trials)
    selected_record = selected_trial["last_level_record"]
    selected_params = selected_trial["params"]
    config_snapshot_path = os.path.join(rung_root, "resolved_config.yaml")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    trials_jsonl_path = os.path.join(rung_root, "trials.jsonl")
    trial_records_path = os.path.join(rung_root, "trial_records.json")
    all_records_path = os.path.join(rung_root, "all_records.jsonl")
    selected_paths = write_selected_trial_artifacts(rung_root, selected_record)
    selected_summary = run_optuna_experiment.read_summary(selected_paths["summary_path"])
    if selected_summary is None:
        raise RuntimeError(
            f"Could not read selected trial summary from {selected_paths['summary_path']!r}."
        )
    tuned_param_name = run_optuna_experiment.resolve_tuned_lr_param_name(config["hyperparameters"])

    rung_result = {
        "experiment_name": experiment["name"],
        "schema_version": 1,
        "stage": "serial_successive_halving_rung",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_running_time_hours": float(total_running_time_hours),
        "config_path": os.path.abspath(config_path),
        "config_snapshot_path": os.path.abspath(config_snapshot_path),
        "rung_root": os.path.abspath(rung_root),
        "rung_name": rung_name(rung_index),
        "rung_index": int(rung_index),
        "rung_target_iters": int(rung_iters),
        "target_family": experiment["target_family"],
        "target_dataset": experiment["target_dataset"],
        "target_model_size": experiment["target_model_size"],
        "selection_metric": task["train_metric"],
        "selected_trial_number": int(selected_trial["trial_number"]),
        "best_params": selected_params,
        "tuned_hyperparameter_name": tuned_param_name,
        "best_hyperparameter_value": float(selected_params[tuned_param_name]),
        "best_learning_rate": float(selected_params[tuned_param_name]),
        "best_value": selected_record["train_objective_value"],
        "best_train_value": selected_record["train_objective_value"],
        "best_test_value": selected_record["test_objective_value"],
        "selected_trial_id": selected_trial["trial_id"],
        "selected_trial_dir": selected_record["trial_dir"],
        "selected_artifact_root": selected_paths["selected_root"],
        "selected_summary_path": selected_paths["summary_path"],
        "selected_records_path": selected_paths["records_path"],
        "selected_log_path": selected_paths["log_path"],
        "direction": "minimize" if task["metric_mode"] == "min" else "maximize",
        "num_trials": len(trial_records),
        "num_completed_trials": len(trial_records),
        "num_pruned_trials": int(pruned_count),
        "num_iterations_per_trial": int(task["num_iterations_per_trial"]),
        "max_running_time_per_trial_hours": float(task.get("max_running_time_per_trial_hours", 0.0)),
        "all_records_path": os.path.abspath(all_records_path),
        "trials_jsonl_path": os.path.abspath(trials_jsonl_path),
        "trial_records_path": os.path.abspath(trial_records_path),
        "termination_reason": selected_summary.get("termination_reason"),
        "forward_backward_hours": selected_summary.get(
            "forward_backward_hours", selected_summary.get("wall_clock_hours")
        ),
        "elapsed_wall_clock_hours": selected_summary.get("elapsed_wall_clock_hours"),
        "best_train_loss": float(selected_summary["best_train_loss"]),
        "best_val_loss": float(selected_summary["best_val_loss"]),
        "stop_reason": "serial_successive_halving_rung_completed",
    }
    rung_result_path = os.path.join(rung_root, "result.json")
    write_json(trial_records_path, trial_records)
    write_json(rung_result_path, rung_result)
    return rung_result_path, rung_result


def main():
    require_single_controller()

    session_start_time = time.time()
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    task = config["task"]
    hyperparameters = config["hyperparameters"]
    reduction_factor = int(args.reduction_factor)

    if args.num_iterations_per_trial is not None:
        if int(args.num_iterations_per_trial) < 1:
            raise ValueError("--num-iterations-per-trial must be >= 1.")
        task["num_iterations_per_trial"] = int(args.num_iterations_per_trial)
        fixed_args = config.setdefault("fixed_args", {})
        fixed_args["max_iters"] = int(args.num_iterations_per_trial)
        fixed_args["lr_decay_iters"] = int(args.num_iterations_per_trial)

    run_optuna_experiment.resolve_tuned_lr_param_name(hyperparameters)

    if args.run_root:
        run_root = ensure_dir(args.run_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = ensure_dir(
            os.path.join(
                experiment["output_root"],
                f"{experiment['name']}_serial_halving_{timestamp}",
            )
        )

    initial_trial_count = int(args.num_trials) if args.num_trials is not None else 1
    if initial_trial_count < 1:
        raise ValueError("--num-trials must be >= 1.")
    num_rungs = inferred_rung_count(
        initial_trial_count=initial_trial_count,
        reduction_factor=reduction_factor,
    )
    rung_budgets = rung_iteration_budgets(
        total_iters=int(task["num_iterations_per_trial"]),
        num_levels=num_rungs,
        reduction_factor=reduction_factor,
    )
    state = load_or_initialize_controller_state(
        run_root=run_root,
        args=args,
        config=config,
        num_rungs=num_rungs,
        rung_budgets=rung_budgets,
        initial_trial_count=initial_trial_count,
        reduction_factor=reduction_factor,
    )
    update_total_running_time(state, session_start_time)
    save_controller_state(run_root, state)
    write_public_result(run_root, state)

    while state["next_rung_index"] < num_rungs:
        rung_index = int(state["next_rung_index"])
        target_iters = rung_budgets[rung_index]
        active_trials = active_trials_from_state(state)
        rung_root = ensure_dir(os.path.join(run_root, rung_name(rung_index)))
        all_records_path = os.path.join(rung_root, "all_records.jsonl")
        trial_records = level_records_for_active_trials(active_trials, rung_index)
        completed_ids = {record["trial_id"] for record in trial_records}
        if trial_records:
            write_partial_rung_records(rung_root, trial_records)

        print(
            f"[halving] rung {rung_index + 1}/{num_rungs}: "
            f"{len(active_trials)} active trials to {target_iters} iterations"
        )
        for trial_state in sorted(active_trials, key=lambda trial: trial["trial_number"]):
            if trial_state["trial_id"] in completed_ids:
                continue
            print(
                f"[halving] running {trial_state['trial_id']} "
                f"({'resume' if trial_state['completed_rungs'] else 'scratch'}) "
                f"to {target_iters} iterations"
            )
            rung_record = run_trial_rung(
                config=config,
                rung_index=rung_index,
                target_iters=target_iters,
                trial_state=trial_state,
                rung_root=rung_root,
                all_records_path=all_records_path,
            )
            trial_records.append(rung_record)
            completed_ids.add(trial_state["trial_id"])
            write_partial_rung_records(rung_root, trial_records)
            update_total_running_time(state, session_start_time)
            save_controller_state(run_root, state)
            write_public_result(run_root, state)

        pruned_count = 0
        if rung_index < num_rungs - 1:
            survivor_count = max(1, int(math.ceil(len(active_trials) / reduction_factor)))
            ranked_trials = sorted(
                active_trials,
                key=lambda trial: (
                    metric_sort_key(task["metric_mode"], trial["train_objective_value"]),
                    trial["trial_number"],
                ),
            )
            survivors = ranked_trials[:survivor_count]
            survivor_ids = {trial["trial_id"] for trial in survivors}
            updated_records = []
            for record in trial_records:
                pruned = record["trial_id"] not in survivor_ids
                if pruned:
                    pruned_count += 1
                record["pruned"] = pruned
                updated_records.append(record)
            write_partial_rung_records(rung_root, updated_records)
            write_json(
                os.path.join(rung_root, "pruning_result.json"),
                {
                    "schema_version": 1,
                    "stage": "serial_successive_halving_pruning",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "rung_index": int(rung_index),
                    "target_iters": int(target_iters),
                    "survivor_count": int(survivor_count),
                    "survivor_ids": sorted(survivor_ids),
                },
            )
            state["active_trial_ids"] = [trial["trial_id"] for trial in survivors]
            trial_records = updated_records
        else:
            write_partial_rung_records(rung_root, trial_records)
            state["active_trial_ids"] = [trial["trial_id"] for trial in active_trials]

        total_running_time_hours = update_total_running_time(state, session_start_time)
        rung_result_path, rung_result = write_rung_result(
            config=config,
            config_path=args.config,
            rung_index=rung_index,
            rung_iters=target_iters,
            rung_root=rung_root,
            trial_records=trial_records,
            active_trials=active_trials_from_state(state),
            pruned_count=pruned_count,
            total_running_time_hours=total_running_time_hours,
        )
        completed_rung_entry = {
            "rung_name": rung_name(rung_index),
            "rung_index": int(rung_index),
            "rung_target_iters": int(target_iters),
            "active_trials_after_rung": len(state["active_trial_ids"]),
            "rung_root": os.path.abspath(rung_root),
            "result_path": os.path.abspath(rung_result_path),
        }
        state["completed_rungs"] = [
            entry for entry in state["completed_rungs"]
            if int(entry["rung_index"]) != int(completed_rung_entry["rung_index"])
        ]
        state["completed_rungs"].append(completed_rung_entry)
        state["completed_rungs"] = sorted(
            state["completed_rungs"],
            key=lambda entry: int(entry["rung_index"]),
        )
        state["final_results"] = [
            result for result in state["final_results"]
            if int(result["rung_index"]) != int(rung_result["rung_index"])
        ]
        state["final_results"].append(rung_result)
        state["final_results"] = sorted(
            state["final_results"],
            key=lambda result: int(result["rung_index"]),
        )
        state["next_rung_index"] = rung_index + 1
        state["completed_running_time_hours"] = float(update_total_running_time(state, session_start_time))
        save_controller_state(run_root, state)
        write_public_result(run_root, state)

    print(f"[halving] completed serial successive halving run at {run_root}")


if __name__ == "__main__":
    main()
