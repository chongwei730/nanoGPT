import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import run_optuna_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stage two: final training with the best learning rate from stage one."
    )
    parser.add_argument(
        "stage1_input",
        help="Path to a stage1_result.json or a stage1_manifest.json.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional config override. Defaults to each stage-one resolved config snapshot.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def stage1_results_from_input(path):
    payload = read_json(path)
    if payload.get("stage") == "stage1":
        return [path]
    if payload.get("stage") == "stage1_multilevel":
        return [entry["stage1_result_path"] for entry in payload["levels"]]
    raise ValueError(
        f"Expected a stage1_result.json or stage1_manifest.json, got stage={payload.get('stage')!r}."
    )


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def copy_if_exists(src, dst):
    if src and os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False


def run_stage2(stage1_result_path, config_override):
    start_time = time.time()
    stage1_result = read_json(stage1_result_path)
    best_params = stage1_result.get("best_params") or {}
    if "learning_rate" not in best_params:
        raise ValueError(f"Stage-one result has no best learning_rate: {stage1_result_path}")

    config_path = config_override or stage1_result["config_snapshot_path"]
    stage1_root = stage1_result["stage1_root"]
    level_root = os.path.dirname(stage1_root)
    stage2_root = ensure_dir(os.path.join(level_root, "stage2"))
    final_dir = ensure_dir(os.path.join(stage2_root, "final"))
    summary_path = os.path.join(final_dir, "summary.json")
    log_path = os.path.join(final_dir, "stage2.log")
    records_path = os.path.join(final_dir, "records.jsonl")
    selected_summary_path = stage1_result.get("selected_summary_path", "")
    selected_records_path = stage1_result.get("selected_records_path", "")
    selected_log_path = stage1_result.get("selected_log_path", "")
    selected_trial_id = stage1_result.get("selected_trial_id", "")

    summary = run_optuna_experiment.read_summary(selected_summary_path)
    if summary is None:
        raise RuntimeError(
            f"Could not read selected trial summary from {selected_summary_path!r}."
        )
    promoted_summary = dict(summary)
    promoted_summary["trial_id"] = selected_trial_id or summary.get("trial_id", "stage2_final")
    promoted_summary["out_dir"] = os.path.abspath(final_dir)
    write_json(summary_path, promoted_summary)
    copy_if_exists(selected_records_path, records_path)
    copy_if_exists(selected_log_path, log_path)
    loaded_learning_rate = run_optuna_experiment.load_learning_rate_from_run(
        summary=summary,
        run_dir=stage1_result.get("selected_trial_dir", ""),
    )
    if loaded_learning_rate is None:
        loaded_learning_rate = float(best_params["learning_rate"])

    stage2_result = {
        "schema_version": 1,
        "stage": "stage2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_running_time_hours": max(0.0, (time.time() - start_time) / 3600.0),
        "stage1_result_path": os.path.abspath(stage1_result_path),
        "config_path": os.path.abspath(config_path),
        "stage2_root": os.path.abspath(stage2_root),
        "final_dir": os.path.abspath(final_dir),
        "summary_path": os.path.abspath(summary_path),
        "log_path": os.path.abspath(log_path),
        "records_path": os.path.abspath(records_path),
        "loaded_learning_rate": loaded_learning_rate,
        "num_iterations_per_trial": int(stage1_result["num_iterations_per_trial"]),
        "max_study_time_hours": float(stage1_result["max_study_time_hours"]),
        "max_running_time_per_trial_hours": float(stage1_result["max_running_time_per_trial_hours"]),
        "returncode": 0,
        "best_train_loss": float(promoted_summary["best_train_loss"]),
        "best_val_loss": float(promoted_summary["best_val_loss"]),
        "termination_reason": promoted_summary.get("termination_reason"),
        "stage2_forward_backward_hours": promoted_summary.get("forward_backward_hours", promoted_summary.get("wall_clock_hours")),
        "elapsed_wall_clock_hours": promoted_summary.get("elapsed_wall_clock_hours"),
    }
    stage2_result_path = os.path.join(stage2_root, "stage2_result.json")
    write_json(stage2_result_path, stage2_result)
    print(
        "[stage2] promoted selected completed trial to final result for "
        f"{stage1_result['max_study_time_hours']:g}h tuning budget"
    )
    print(f"[stage2] wrote result: {stage2_result_path}")
    return stage2_result


def main():
    args = parse_args()
    stage1_result_paths = stage1_results_from_input(args.stage1_input)
    results = [run_stage2(path, args.config) for path in stage1_result_paths]

    if len(results) > 1:
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(results[0]["stage2_root"]))),
            "stage2_manifest.json",
        )
        manifest = {
            "schema_version": 1,
            "stage": "stage2_multilevel",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stage1_input": os.path.abspath(args.stage1_input),
            "results": results,
        }
        write_json(manifest_path, manifest)
        print(f"[stage2] wrote manifest: {manifest_path}")
    elif results:
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(results[0]["stage2_root"])),
            "stage2_manifest.json",
        )
        manifest = {
            "schema_version": 1,
            "stage": "stage2_multilevel",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stage1_input": os.path.abspath(args.stage1_input),
            "results": results,
        }
        write_json(manifest_path, manifest)
        print(f"[stage2] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
