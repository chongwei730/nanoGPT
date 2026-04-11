import argparse
import json
import os
import subprocess
import sys
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


def run_stage2(stage1_result_path, config_override):
    stage1_result = read_json(stage1_result_path)
    best_params = stage1_result.get("best_params") or {}
    if "learning_rate" not in best_params:
        raise ValueError(f"Stage-one result has no best learning_rate: {stage1_result_path}")

    config_path = config_override or stage1_result["config_snapshot_path"]
    config = run_optuna_experiment.load_config(config_path)
    stage1_root = stage1_result["stage1_root"]
    level_root = os.path.dirname(stage1_root)
    stage2_root = ensure_dir(os.path.join(level_root, "stage2"))
    final_dir = ensure_dir(os.path.join(stage2_root, "final"))
    summary_path = os.path.join(final_dir, "summary.json")
    log_path = os.path.join(final_dir, "stage2.log")
    records_path = os.path.join(final_dir, "records.jsonl")
    prune_signal_path = os.path.join(final_dir, "PRUNE")
    if os.path.exists(prune_signal_path):
        os.remove(prune_signal_path)

    command = run_optuna_experiment.build_command(
        config=config,
        trial_dir=final_dir,
        trial_id="stage2_final",
        sampled_params={"learning_rate": best_params["learning_rate"]},
        summary_path=summary_path,
        prune_signal_path=prune_signal_path,
        max_running_time_hours=0.0,
    )

    print(
        "[stage2] running final training for "
        f"{stage1_result['max_study_time_hours']:g}h tuning budget "
        f"with learning_rate={best_params['learning_rate']}"
    )
    returncode, _ = run_optuna_experiment.stream_process(
        command,
        log_path,
        record_paths=[records_path],
        record_context={
            "stage": "stage2",
            "level_hours": stage1_result["max_study_time_hours"],
            "trial_id": "stage2_final",
            "learning_rate": best_params["learning_rate"],
        },
    )
    summary = run_optuna_experiment.read_summary(summary_path)
    if summary is None:
        raise RuntimeError(f"Stage two did not produce a summary file at {summary_path}.")

    stage2_result = {
        "schema_version": 1,
        "stage": "stage2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage1_result_path": os.path.abspath(stage1_result_path),
        "config_path": os.path.abspath(config_path),
        "stage2_root": os.path.abspath(stage2_root),
        "final_dir": os.path.abspath(final_dir),
        "summary_path": os.path.abspath(summary_path),
        "log_path": os.path.abspath(log_path),
        "records_path": os.path.abspath(records_path),
        "loaded_learning_rate": float(best_params["learning_rate"]),
        "max_study_time_hours": float(stage1_result["max_study_time_hours"]),
        "max_running_time_per_trial_hours": float(stage1_result["max_running_time_per_trial_hours"]),
        "returncode": int(returncode),
        "best_train_loss": float(summary["best_train_loss"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "termination_reason": summary.get("termination_reason"),
        "stage2_forward_backward_hours": summary.get("forward_backward_hours", summary.get("wall_clock_hours")),
        "elapsed_wall_clock_hours": summary.get("elapsed_wall_clock_hours"),
    }
    stage2_result_path = os.path.join(stage2_root, "stage2_result.json")
    write_json(stage2_result_path, stage2_result)
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
