import argparse
import json
import os
import sys
import time
from datetime import datetime

import run_optuna_experiment


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a line-search training job with stage2-compatible outputs."
    )
    parser.add_argument(
        "--run-root",
        default="/work/nvme/bgop/cchen47/experiment_runs/gpt124m_line_search_stage2",
        help="Root directory for stage2 outputs.",
    )
    parser.add_argument(
        "--train-script",
        default="train_linesearch.py",
        help="Training script to launch.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of processes per node for torchrun.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Optional master port for torchrun.",
    )
    parser.add_argument(
        "--experiment-name",
        default="gpt124m_line_search",
        help="Experiment name recorded in summary.json.",
    )
    parser.add_argument(
        "--trial-id",
        default="stage2_final",
        help="Trial id recorded in summary.json.",
    )
    parser.add_argument(
        "--metric-mode",
        default="min",
        choices=["min", "max"],
        help="Metric mode recorded in summary.json.",
    )
    parser.add_argument(
        "--max-running-time-hours",
        type=float,
        default=0.0,
        help="Max running time for training loop.",
    )
    parser.add_argument(
        "--max-study-time-hours",
        type=float,
        default=0.0,
        help="Budget used in stage2_result.json (can be 0 if not applicable).",
    )
    parser.add_argument(
        "--stage1-result-path",
        default="",
        help="Optional stage1_result.json path to record in stage2_result.json.",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="Optional config path to record in stage2_result.json.",
    )
    parser.add_argument(
        "--save-last-checkpoint",
        type=parse_bool,
        default=False,
        help="Whether to save ckpt_last.pt.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to the training script (use -- to separate).",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def validate_linesearch_train_args(train_args):
    forbidden_prefixes = (
        "--learning_rate=",
        "--muon_lr=",
        "--hyperparameter-name=",
        "--hyperparameter-value=",
        "--learning-rate=",
    )
    forbidden_flags = {
        "--learning_rate",
        "--muon_lr",
        "--hyperparameter-name",
        "--hyperparameter-value",
        "--learning-rate",
    }
    normalized_args = train_args[1:] if train_args and train_args[0] == "--" else train_args
    for arg in normalized_args:
        if arg in forbidden_flags or arg.startswith(forbidden_prefixes):
            raise ValueError(
                "Line-search runs must not accept externally supplied learning-rate "
                f"hyperparameters, but found forbidden argument: {arg}"
            )


def build_command(
    train_script,
    nproc_per_node,
    master_port,
    train_args,
    out_dir,
    experiment_name,
    trial_id,
    metric_mode,
    max_running_time_hours,
    save_last_checkpoint,
    summary_path,
    prune_signal_path,
):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
    ]
    if master_port is not None:
        cmd.append(f"--master_port={int(master_port)}")
    cmd.append(train_script)
    cmd.extend(
        [
            f"--out_dir={out_dir}",
            f"--experiment_name={experiment_name}",
            f"--trial_id={trial_id}",
            f"--experiment_metric_mode={metric_mode}",
            f"--max_running_time_hours={max_running_time_hours}",
            f"--save_last_checkpoint={save_last_checkpoint}",
            f"--experiment_summary_path={summary_path}",
            f"--experiment_records_path={os.path.join(out_dir, 'records.jsonl')}",
            f"--prune_signal_path={prune_signal_path}",
        ]
    )
    if train_args:
        if train_args[0] == "--":
            cmd.extend(train_args[1:])
        else:
            cmd.extend(train_args)
    return cmd


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    start_time = time.time()
    args = parse_args()

    run_root = ensure_dir(args.run_root)
    stage2_root = os.path.abspath(run_root)
    final_dir = ensure_dir(os.path.join(run_root, "final"))
    summary_path = os.path.join(final_dir, "summary.json")
    log_path = os.path.join(final_dir, "stage2.log")
    records_path = os.path.join(final_dir, "records.jsonl")
    prune_signal_path = os.path.join(final_dir, "PRUNE")
    if os.path.exists(prune_signal_path):
        os.remove(prune_signal_path)
    validate_linesearch_train_args(args.train_args)
    open(records_path, "a", encoding="utf-8").close()

    command = build_command(
        train_script=args.train_script,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        train_args=args.train_args,
        out_dir=final_dir,
        experiment_name=args.experiment_name,
        trial_id=args.trial_id,
        metric_mode=args.metric_mode,
        max_running_time_hours=args.max_running_time_hours,
        save_last_checkpoint=args.save_last_checkpoint,
        summary_path=summary_path,
        prune_signal_path=prune_signal_path,
    )

    returncode, _ = run_optuna_experiment.stream_process(
        command,
        log_path,
    )

    summary = run_optuna_experiment.read_summary(summary_path)
    if summary is None:
        raise RuntimeError(f"Line-search run did not produce a summary file at {summary_path}.")
    loaded_learning_rate = run_optuna_experiment.load_learning_rate_from_run(
        summary=summary,
        run_dir=final_dir,
    )
    if loaded_learning_rate is None and summary.get("learning_rate") is not None:
        loaded_learning_rate = float(summary["learning_rate"])

    stage2_result = {
        "schema_version": 1,
        "stage": "stage2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_running_time_hours": max(0.0, (time.time() - start_time) / 3600.0),
        "stage1_result_path": os.path.abspath(args.stage1_result_path) if args.stage1_result_path else "",
        "config_path": os.path.abspath(args.config_path) if args.config_path else "",
        "stage2_root": stage2_root,
        "final_dir": os.path.abspath(final_dir),
        "summary_path": os.path.abspath(summary_path),
        "log_path": os.path.abspath(log_path),
        "records_path": os.path.abspath(records_path),
        "loaded_learning_rate": loaded_learning_rate,
        "max_study_time_hours": float(args.max_study_time_hours),
        "max_running_time_per_trial_hours": float(args.max_running_time_hours),
        "returncode": int(returncode),
        "best_train_loss": float(summary["best_train_loss"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "termination_reason": summary.get("termination_reason"),
        "stage2_forward_backward_hours": summary.get("forward_backward_hours", summary.get("wall_clock_hours")),
        "elapsed_wall_clock_hours": summary.get("elapsed_wall_clock_hours"),
    }
    stage2_result_path = os.path.join(run_root, "stage2_result.json")
    write_json(stage2_result_path, stage2_result)

    manifest_path = os.path.join(run_root, "stage2_manifest.json")
    manifest = {
        "schema_version": 1,
        "stage": "stage2_multilevel",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage1_input": os.path.abspath(args.stage1_result_path) if args.stage1_result_path else "",
        "results": [stage2_result],
    }
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
