import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stage one: Optuna learning-rate search across tuning-time levels."
    )
    parser.add_argument("config", help="Path to the experiment YAML config.")
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=None,
        help="One or more stage-one tuning budgets in hours. Defaults to optuna.max_study_time_hours_levels or scalar value.",
    )
    parser.add_argument(
        "--run-root",
        default="",
        help="Optional shared output root for all levels.",
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


def levels_from_config(config):
    optuna_cfg = config.get("optuna", {})
    levels = optuna_cfg.get("max_study_time_hours_levels")
    if levels is None:
        levels = [optuna_cfg.get("max_study_time_hours", config["task"]["max_running_time_per_trial_hours"])]
    if not isinstance(levels, list):
        raise ValueError("optuna.max_study_time_hours_levels must be a list when set.")
    return [float(level) for level in levels]


def level_name(level):
    text = f"{level:g}".replace(".", "p")
    return f"level_{text}h"


def main():
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

    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    levels = args.levels if args.levels is not None else levels_from_config(config)

    if args.run_root:
        run_root = args.run_root
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = os.path.join(
            experiment["output_root"],
            f"{experiment['name']}_staged_{timestamp}",
        )
    run_root = ensure_dir(run_root)

    manifest = {
        "schema_version": 1,
        "stage": "stage1_multilevel",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": os.path.abspath(args.config),
        "run_root": os.path.abspath(run_root),
        "levels": [],
    }

    for level in levels:
        level_root = ensure_dir(os.path.join(run_root, level_name(level)))
        stage1_root = ensure_dir(os.path.join(level_root, "stage1"))
        stage1_result_path = os.path.join(stage1_root, "stage1_result.json")
        cmd = [
            sys.executable,
            "run_optuna_experiment.py",
            args.config,
            "--experiment-root",
            stage1_root,
            "--max-running-time-per-trial-hours",
            str(config["task"]["max_running_time_per_trial_hours"]),
            "--max-study-time-hours",
            str(level),
            "--stage1-result-path",
            stage1_result_path,
            "--skip-runtime-table-validation",
        ]
        print(f"[stage1] running level {level:g}h")
        subprocess.run(cmd, cwd=os.getcwd(), check=True)
        manifest["levels"].append(
            {
                "max_study_time_hours": float(level),
                "max_running_time_per_trial_hours": float(config["task"]["max_running_time_per_trial_hours"]),
                "level_root": os.path.abspath(level_root),
                "stage1_root": os.path.abspath(stage1_root),
                "stage1_result_path": os.path.abspath(stage1_result_path),
                "stage1_records_path": os.path.abspath(os.path.join(stage1_root, "all_records.jsonl")),
            }
        )

    manifest_path = os.path.join(run_root, "stage1_manifest.json")
    write_json(manifest_path, manifest)
    print(f"[stage1] wrote manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
