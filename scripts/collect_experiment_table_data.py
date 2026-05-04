#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect experiment summaries into a structured table dataset."
    )
    parser.add_argument(
        "--experiment-root",
        default="/scratch.global/chen8596/experiment_runs",
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--size-label",
        action="append",
        default=[],
        help="Override size display label, e.g. GPT:124M=124M/5B",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_model_size(raw_size):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([mMbB])", raw_size.strip())
    if not match:
        return raw_size.strip()
    number, suffix = match.groups()
    return f"{number}{suffix.upper()}"

def parse_size_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(
                f"Invalid --size-label value '{item}'. Expected FAMILY:SIZE=LABEL."
            )
        key, label = item.split("=", 1)
        family, size = key.split(":", 1)
        overrides[(family.strip().upper(), normalize_model_size(size.strip()))] = label.strip()
    return overrides


def infer_family(experiment_name):
    lowered = experiment_name.lower()
    if lowered.startswith("gpt"):
        return "GPT"
    if lowered.startswith("llama"):
        return "LLAMA"
    return "UNKNOWN"


def infer_model_size(experiment_name):
    match = re.search(r"(?:gpt|llama)(\d+(?:\.\d+)?[mb])", experiment_name.lower())
    if not match:
        return "UNKNOWN"
    return normalize_model_size(match.group(1))


def infer_method(experiment_name, train_script):
    lowered_name = experiment_name.lower()
    lowered_script = (train_script or "").lower()
    if "line_search" in lowered_name or "linesearch" in lowered_name:
        if "muon" in lowered_name or "muon" in lowered_script:
            return "linesearch_muon"
        return "linesearch_adam"
    if "schedulefree" in lowered_name:
        return "schedulefree_adam"
    if "muon" in lowered_name or "muon" in lowered_script:
        return "muon"
    if "lr_search" in lowered_name or lowered_script == "train.py":
        return "cosine"
    return experiment_name


def is_linesearch_method(method):
    return method in {"linesearch_adam", "linesearch_muon"}


def parse_timestamp(value):
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except AttributeError:
        pass
    except ValueError:
        pass
    for pattern in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(normalized, pattern)
        except ValueError:
            continue
    return None


def serial_total_time_hours(payload, result):
    total_time = result.get("total_running_time_hours")
    if total_time is None:
        total_time = payload.get("total_running_time_hours")
    if total_time is not None:
        total_time = float(total_time)

    created_at = parse_timestamp(payload.get("created_at"))
    updated_at = parse_timestamp(payload.get("updated_at"))
    if created_at is None or updated_at is None:
        return total_time
    timestamp_hours = max(0.0, (updated_at - created_at).total_seconds() / 3600.0)
    if total_time is None:
        return timestamp_hours
    if timestamp_hours > 0 and total_time > timestamp_hours * 1.01:
        return timestamp_hours
    return total_time


def infer_run_dir_from_source_path(source_path):
    source_path = Path(source_path)
    if source_path.name == "serial_halving_result.json":
        return source_path.parent
    if source_path.name == "summary.json" and source_path.parent.name == "final":
        return source_path.parent.parent
    return source_path.parent


def infer_num_trials_from_run_dir(run_dir):
    match = re.search(r"(?:^|_)num_trials_(\d+)(?:_|$)", Path(run_dir).name)
    if not match:
        return None
    return int(match.group(1))


def candidate_row_sort_key(item):
    num_trials_setting = item.get("num_trials_setting")
    return (
        float("inf") if num_trials_setting is None else int(num_trials_setting),
        float("inf") if item["total_time_hours"] is None else float(item["total_time_hours"]),
        float("inf") if item["loss"] is None else float(item["loss"]),
        item["source_path"],
    )


def candidate_selection_key(item):
    return (
        float("inf") if item["loss"] is None else float(item["loss"]),
        float("inf") if item["total_time_hours"] is None else float(item["total_time_hours"]),
        item["source_path"],
    )


def make_candidate(
    family,
    model_size,
    size_label,
    method,
    total_time_hours,
    loss,
    wall_clock_hours,
    run_dir,
    num_trials_setting,
    source_path,
    metadata,
):
    return {
        "family": family,
        "model_size": model_size,
        "size_label": size_label,
        "method": method,
        "total_time_hours": total_time_hours,
        "loss": loss,
        "wall_clock_hours": wall_clock_hours,
        "run_dir": str(run_dir),
        "num_trials_setting": num_trials_setting,
        "source_path": str(source_path),
        "metadata": metadata,
    }


def collect_serial_halving_entries(experiment_root, size_overrides):
    candidates = []
    for result_path in sorted(experiment_root.glob("*/serial_halving_result.json")):
        run_dir = infer_run_dir_from_source_path(result_path)
        payload = load_json(result_path)
        results = payload.get("results", [])
        if not results:
            continue
        result = results[-1]
        experiment_name = result.get("experiment_name", "")
        train_script = ""
        selected_summary_path = result.get("selected_summary_path")
        if selected_summary_path and Path(selected_summary_path).exists():
            selected_summary = load_json(selected_summary_path)
            train_script = selected_summary.get("train_script", "")
        family = (result.get("target_family") or infer_family(experiment_name)).upper()
        model_size = normalize_model_size(
            result.get("target_model_size") or infer_model_size(experiment_name)
        )
        method = infer_method(experiment_name, train_script)
        size_label = size_overrides.get((family, model_size), model_size)
        total_time_hours = serial_total_time_hours(payload, result)
        candidates.append(
            make_candidate(
                family=family,
                model_size=model_size,
                size_label=size_label,
                method=method,
                total_time_hours=total_time_hours,
                loss=result.get("best_val_loss"),
                wall_clock_hours=result.get("elapsed_wall_clock_hours"),
                run_dir=run_dir,
                num_trials_setting=infer_num_trials_from_run_dir(run_dir),
                source_path=result_path,
                metadata={
                    "kind": "serial_halving",
                    "run_dir": str(run_dir),
                    "result_path": result.get("result_path", ""),
                    "rung_index": result.get("rung_index"),
                    "rung_name": result.get("rung_name", ""),
                    "num_trials": result.get("num_trials"),
                    "rung_target_iters": result.get("rung_target_iters"),
                },
            )
        )
    return candidates


def collect_linesearch_entries(experiment_root, size_overrides):
    candidates = []
    for summary_path in sorted(experiment_root.glob("*/final/summary.json")):
        if "/rung_" in str(summary_path):
            continue

        run_dir = infer_run_dir_from_source_path(summary_path)
        summary = load_json(summary_path)
        experiment_name = summary.get("experiment_name", "")
        method = infer_method(experiment_name, summary.get("train_script", ""))
        if not is_linesearch_method(method):
            continue

        family = infer_family(experiment_name).upper()
        model_size = infer_model_size(experiment_name)
        size_label = size_overrides.get((family, model_size), model_size)

        total_time_hours = summary.get("elapsed_wall_clock_hours")
        candidates.append(
            make_candidate(
                family=family,
                model_size=model_size,
                size_label=size_label,
                method=method,
                total_time_hours=total_time_hours,
                loss=summary.get("best_val_loss"),
                wall_clock_hours=summary.get("elapsed_wall_clock_hours"),
                run_dir=run_dir,
                num_trials_setting=infer_num_trials_from_run_dir(run_dir),
                source_path=summary_path,
                metadata={
                    "kind": "linesearch_final",
                    "run_dir": str(run_dir),
                    "experiment_name": experiment_name,
                },
            )
        )
    return candidates


def aggregate_candidates(candidates):
    grouped = defaultdict(list)
    for candidate in candidates:
        key = (candidate["family"], candidate["model_size"], candidate["method"])
        grouped[key].append(candidate)

    entries = []
    for key in sorted(grouped):
        family, model_size, method = key
        rows = sorted(grouped[key], key=candidate_row_sort_key)
        selected = min(rows, key=candidate_selection_key)
        entries.append(
            {
                "family": family,
                "model_size": model_size,
                "size_label": selected["size_label"],
                "method": method,
                "selection_rule": "min_loss",
                "selected": selected,
                "candidates": rows,
            }
        )
    return entries


def main():
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    if not experiment_root.exists():
        raise FileNotFoundError(
            f"Experiment root does not exist: {experiment_root}"
        )
    size_overrides = parse_size_overrides(args.size_label)

    candidates = []
    candidates.extend(collect_serial_halving_entries(experiment_root, size_overrides))
    candidates.extend(collect_linesearch_entries(experiment_root, size_overrides))
    entries = aggregate_candidates(candidates)

    payload = {
        # "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "experiment_root": str(experiment_root.resolve()),
        "entry_count": len(entries),
        "entries": entries,
    }

    text = json.dumps(payload, indent=2, sort_keys=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
