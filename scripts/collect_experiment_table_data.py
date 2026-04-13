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
        default="experiment_runs",
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def normalize_model_size(raw_size):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([mMbB])", raw_size.strip())
    if not match:
        return raw_size.strip()
    number, suffix = match.groups()
    return f"{number}{suffix.upper()}"


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
        return "Linesearch"
    if "muon" in lowered_name or "muon" in lowered_script:
        return "muon"
    if "lr_search" in lowered_name or lowered_script == "train.py":
        return "cosine"
    return experiment_name


def make_candidate(
    family,
    model_size,
    size_label,
    method,
    tuning_time_hours,
    loss,
    wall_clock_hours,
    source_path,
    metadata,
):
    return {
        "family": family,
        "model_size": model_size,
        "size_label": size_label,
        "method": method,
        "tuning_time_hours": tuning_time_hours,
        "loss": loss,
        "wall_clock_hours": wall_clock_hours,
        "source_path": str(source_path),
        "metadata": metadata,
    }


def collect_staged_searches(experiment_root, size_overrides):
    candidates = []
    pattern = "*/rung_*/stage2/final/summary.json"
    for summary_path in sorted(experiment_root.glob(pattern)):
        summary = load_json(summary_path)
        stage2_result_path = summary_path.parents[1] / "stage2_result.json"
        stage1_summary_path = summary_path.parents[2] / "stage1" / "study_summary.json"
        stage2_result = load_json(stage2_result_path) if stage2_result_path.exists() else {}
        stage1_summary = load_json(stage1_summary_path) if stage1_summary_path.exists() else {}

        experiment_name = summary.get("experiment_name", "")
        train_script = summary.get("train_script", "")
        family = stage1_summary.get("target_family") or infer_family(experiment_name)
        model_size = normalize_model_size(
            stage1_summary.get("target_model_size") or infer_model_size(experiment_name)
        )
        size_label = size_overrides.get((family.upper(), model_size), model_size)
        method = infer_method(experiment_name, train_script)
        tuning_time_hours = stage2_result.get("max_study_time_hours")

        candidates.append(
            make_candidate(
                family=family,
                model_size=model_size,
                size_label=size_label,
                method=method,
                tuning_time_hours=tuning_time_hours,
                loss=summary.get("best_val_loss"),
                wall_clock_hours=summary.get("elapsed_wall_clock_hours"),
                source_path=summary_path,
                metadata={
                    "kind": "staged_search",
                    "experiment_name": experiment_name,
                    "stage2_result_path": str(stage2_result_path) if stage2_result_path.exists() else "",
                    "stage1_summary_path": str(stage1_summary_path) if stage1_summary_path.exists() else "",
                },
            )
        )
    return candidates


def collect_standalone_finals(experiment_root, size_overrides):
    candidates = []
    pattern = "*/final/summary.json"
    for summary_path in sorted(experiment_root.glob(pattern)):
        if "rung_" in str(summary_path):
            continue

        summary = load_json(summary_path)
        stage2_result_path = summary_path.parents[1] / "stage2_result.json"
        stage2_result = load_json(stage2_result_path) if stage2_result_path.exists() else {}
        experiment_name = summary.get("experiment_name", "")
        train_script = summary.get("train_script", "")
        family = infer_family(experiment_name)
        model_size = infer_model_size(experiment_name)
        size_label = size_overrides.get((family.upper(), model_size), model_size)
        method = infer_method(experiment_name, train_script)

        candidates.append(
            make_candidate(
                family=family,
                model_size=model_size,
                size_label=size_label,
                method=method,
                tuning_time_hours=None,
                loss=summary.get("best_val_loss"),
                wall_clock_hours=summary.get("elapsed_wall_clock_hours"),
                source_path=summary_path,
                metadata={
                    "kind": "standalone_final",
                    "experiment_name": experiment_name,
                    "stage2_result_path": str(stage2_result_path) if stage2_result_path.exists() else "",
                    "loaded_learning_rate": stage2_result.get("loaded_learning_rate"),
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
        rows = sorted(
            grouped[key],
            key=lambda item: (
                float("inf") if item["loss"] is None else item["loss"],
                float("inf") if item["tuning_time_hours"] is None else item["tuning_time_hours"],
                item["source_path"],
            ),
        )
        selected = rows[0]
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
    size_overrides = parse_size_overrides(args.size_label)

    candidates = []
    candidates.extend(collect_staged_searches(experiment_root, size_overrides))
    candidates.extend(collect_standalone_finals(experiment_root, size_overrides))
    entries = aggregate_candidates(candidates)

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
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
