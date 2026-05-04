#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a LaTeX summary table from structured experiment table data."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional JSON produced by collect_experiment_table_data.py.",
    )
    parser.add_argument(
        "--experiment-root",
        default="/scratch.global/chen8596/experiment_runs_last
        help="Experiment output root used when --input is omitted.",
    )
    parser.add_argument("--output", default="", help="Optional output path for the LaTeX table")
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Family order to render. Defaults to GPT then LLAMA.",
    )
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        help="Column spec in FAMILY:SIZE=LABEL form, e.g. GPT:124M=124M/5B",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method row label to render. Repeat to keep a fixed row order.",
    )
    parser.add_argument(
        "--method-label",
        action="append",
        default=[],
        help="Display label override in METHOD=LABEL form, e.g. cosine=Method1",
    )
    parser.add_argument(
        "--linesearch-label",
        default="linesearch_adam",
        help="Display label for the Adam line-search row.",
    )
    parser.add_argument(
        "--linesearch-muon-label",
        default="linesearch_muon",
        help="Display label for the Muon line-search row.",
    )
    parser.add_argument(
        "--loss-decimals",
        type=int,
        default=4,
        help="Number of decimals for loss values.",
    )
    parser.add_argument(
        "--rows-per-method",
        type=int,
        default=3,
        help="Number of data rows to render under each method block.",
    )
    parser.add_argument(
        "--size-label",
        action="append",
        default=[],
        help="Display label override in FAMILY:SIZE=LABEL form, e.g. GPT:124M=124M/5B",
    )
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="Plot training curves from experiment records instead of rendering the LaTeX table.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default="observation/training_curves",
        help="Directory for plots written by --plot-curves.",
    )
    parser.add_argument(
        "--curve-metric",
        default="train_loss",
        help="Metric to plot from records.jsonl, e.g. train_loss or val_loss.",
    )
    parser.add_argument(
        "--curve-x",
        choices=["step", "wall_clock_hours", "forward_backward_hours"],
        default="step",
        help="X axis for --plot-curves.",
    )
    parser.add_argument(
        "--plot-format",
        default="png",
        help="Image format for --plot-curves, e.g. png or pdf.",
    )
    parser.add_argument(
        "--plot-max-legend-items",
        type=int,
        default=24,
        help="Maximum number of curves to list in each legend. Curves are still plotted.",
    )
    parser.add_argument(
        "--curve-group-by",
        choices=["method", "method_model", "method_model_trials"],
        default="method_model_trials",
        help=(
            "Grouping for --plot-curves. The default writes one plot for each "
            "method, model family/size, and num_trials setting."
        ),
    )
    return parser.parse_args()


def normalize_model_size(raw_size):
    raw_size = raw_size.strip()
    if raw_size and raw_size[-1].lower() in {"m", "b"}:
        return raw_size[:-1] + raw_size[-1].upper()
    return raw_size


def load_payload(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON on {path}:{line_number}") from exc
    return records


def parse_columns(items):
    parsed = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(
                f"Invalid --column value '{item}'. Expected FAMILY:SIZE=LABEL."
            )
        key, label = item.split("=", 1)
        family, size = key.split(":", 1)
        parsed.setdefault(family.strip().upper(), []).append(
            {"model_size": normalize_model_size(size), "label": label.strip()}
        )
    return parsed


def parse_method_labels(items):
    labels = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --method-label value '{item}'. Expected METHOD=LABEL."
            )
        method, label = item.split("=", 1)
        labels[method.strip()] = label.strip()
    return labels


def parse_size_labels(items):
    labels = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(
                f"Invalid --size-label value '{item}'. Expected FAMILY:SIZE=LABEL."
            )
        key, label = item.split("=", 1)
        family, size = key.split(":", 1)
        labels[(family.strip().upper(), normalize_model_size(size))] = label.strip()
    return labels


def format_hours(value):
    if value is None:
        return ""
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return f"{rounded}h"
    return f"{value:.2f}h"


def format_loss(value, decimals):
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}"


def discover_columns(entries, family_order):
    discovered = {family: [] for family in family_order}
    seen = {family: set() for family in family_order}
    for entry in entries:
        family = entry["family"].upper()
        if family not in discovered:
            continue
        key = entry["model_size"]
        if key in seen[family]:
            continue
        discovered[family].append(
            {"model_size": key, "label": entry.get("size_label", key)}
        )
        seen[family].add(key)
    for family in discovered:
        discovered[family].sort(key=lambda item: model_size_sort_key(item["model_size"]))
    return discovered


def build_entry_map(entries):
    entry_map = {}
    for entry in entries:
        key = (entry["family"].upper(), entry["model_size"], entry["method"])
        entry_map[key] = entry
    return entry_map


def build_candidate_map(entries):
    candidate_map = {}
    for entry in entries:
        key = (entry["family"].upper(), entry["model_size"], entry["method"])
        candidates = list(entry.get("candidates", []))
        candidates.sort(
            key=lambda item: (
                float("inf") if get_total_time_hours(item) is None else float(get_total_time_hours(item)),
                float("inf") if item.get("loss") is None else float(item["loss"]),
            )
        )
        candidate_map[key] = candidates
    return candidate_map


def is_linesearch_method(method):
    return method in {"linesearch_adam", "linesearch_muon"}


def parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
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


def collect_body_methods(entries, explicit_methods):
    if explicit_methods:
        return [method for method in explicit_methods if not is_linesearch_method(method)]
    preferred = ["cosine", "muon", "schedulefree_adam"]
    methods = []
    for entry in entries:
        method = entry["method"]
        if is_linesearch_method(method):
            continue
        if method not in methods:
            methods.append(method)
    methods.sort(key=lambda method: (preferred.index(method) if method in preferred else len(preferred), method))
    return methods


def collect_linesearch_methods(entries, explicit_methods):
    if explicit_methods:
        methods = [method for method in explicit_methods if is_linesearch_method(method)]
    else:
        preferred = ["linesearch_adam", "linesearch_muon"]
        discovered = []
        for entry in entries:
            method = entry["method"]
            if is_linesearch_method(method) and method not in discovered:
                discovered.append(method)
        methods = [method for method in preferred if method in discovered]
        methods.extend([method for method in discovered if method not in methods])
    return methods


def get_total_time_hours(candidate):
    if candidate.get("total_time_hours") is not None:
        return candidate.get("total_time_hours")
    if candidate.get("tuning_time_hours") is not None:
        return candidate.get("tuning_time_hours")
    return candidate.get("wall_clock_hours")


def total_hours_from_records(records, key="wall_clock_hours"):
    total = 0.0
    segment_max = None
    previous = None
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        value = float(value)
        if previous is not None and value < previous:
            if segment_max is not None:
                total += segment_max
            segment_max = value
        else:
            segment_max = value if segment_max is None else max(segment_max, value)
        previous = value
    if segment_max is not None:
        total += segment_max
    return total if total > 0 else None


def trial_time_hours_from_records_path(records_path):
    records_path = Path(records_path)
    if not records_path.exists():
        return None
    return total_hours_from_records(load_jsonl(records_path))


def trial_time_hours_from_trial_dir(trial_dir):
    trial_dir = Path(trial_dir)
    records_time = trial_time_hours_from_records_path(trial_dir / "records.jsonl")
    if records_time is not None:
        return records_time
    summary_path = trial_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    if summary.get("elapsed_wall_clock_hours") is not None:
        return float(summary["elapsed_wall_clock_hours"])
    if summary.get("wall_clock_hours") is not None:
        return float(summary["wall_clock_hours"])
    return None


def sum_shared_trial_times(run_root):
    run_root = Path(run_root)
    shared_trials_root = run_root / "shared_trials"
    if not shared_trials_root.exists():
        return None
    total = 0.0
    found_any = False
    for trial_dir in sorted(shared_trials_root.glob("trial_*")):
        trial_time = trial_time_hours_from_trial_dir(trial_dir)
        if trial_time is None:
            continue
        total += trial_time
        found_any = True
    return total if found_any else None


def build_tabular_spec(column_count):
    return "c|" + "|".join(["c|c"] * column_count)


def family_header_row(family_name, columns, total_columns):
    cells = [f"\\textbf{{{family_name}}}"]
    padded = list(columns) + [{"label": "Size"}] * (total_columns - len(columns))
    for index, column in enumerate(padded):
        suffix = "|" if index < total_columns - 1 else ""
        cells.append(f"\\multicolumn{{2}}{{c{suffix}}}{{{column['label']}}}")
    return " & ".join(cells) + " \\\\"


def method_header_row(method_label, total_columns):
    cells = [f"\\textbf{{{method_label}}}"]
    for _ in range(total_columns):
        cells.extend(["Total time", "Loss"])
    return " & ".join(cells) + " \\\\"


def method_data_row(method, family, columns, total_columns, candidate_map, row_index, loss_decimals):
    cells = [""]
    padded = list(columns) + [{"model_size": None}] * (total_columns - len(columns))
    for column in padded:
        if not column["model_size"]:
            cells.extend(["", ""])
            continue
        candidates = candidate_map.get((family, column["model_size"], method), [])
        if row_index >= len(candidates):
            cells.extend(["", ""])
            continue
        candidate = candidates[row_index]
        cells.extend(
            [
                format_hours(get_total_time_hours(candidate)),
                format_loss(candidate.get("loss"), loss_decimals),
            ]
        )
    return " & ".join(cells) + " \\\\"


def linesearch_row(label, method, family, columns, total_columns, entry_map, loss_decimals):
    cells = [f"\\textbf{{{label}}}"]
    padded = list(columns) + [{"model_size": None}] * (total_columns - len(columns))
    for column in padded:
        if not column["model_size"]:
            cells.extend(["", ""])
            continue
        entry = entry_map.get((family, column["model_size"], method))
        total_time_value = ""
        loss_value = ""
        if entry:
            selected = entry["selected"]
            total_time_value = format_hours(get_total_time_hours(selected))
            loss_value = format_loss(selected.get("loss"), loss_decimals)
        cells.extend([total_time_value, loss_value])
    return " & ".join(cells) + " \\\\"


def model_size_sort_key(model_size):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([MB])", model_size)
    if not match:
        return (float("inf"), model_size)
    number, suffix = match.groups()
    multiplier = 1_000 if suffix == "B" else 1
    return (float(number) * multiplier, model_size)


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


def infer_experiment_name_from_run_dir(run_dir):
    name = run_dir.name
    suffixes = [
        "_serial_halving",
        "_stage2",
        "_maxiters",
        "_num_trials",
    ]
    end = len(name)
    for suffix in suffixes:
        index = name.find(suffix)
        if index != -1:
            end = min(end, index)
    return name[:end]


def slugify(value):
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def make_candidate(
    family,
    model_size,
    size_label,
    method,
    total_time_hours,
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
        "total_time_hours": total_time_hours,
        "loss": loss,
        "wall_clock_hours": wall_clock_hours,
        "source_path": str(source_path),
        "metadata": metadata,
    }


def collect_serial_halving_entries(experiment_root, size_labels):
    candidates = []
    for result_path in sorted(experiment_root.glob("*/serial_halving_result.json")):
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
        size_label = size_labels.get((family, model_size), model_size)
        method = infer_method(experiment_name, train_script)
        total_time_hours = sum_shared_trial_times(result_path.parent)
        if total_time_hours is None:
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
                source_path=result_path,
                metadata={
                    "kind": "serial_halving",
                    "result_path": result.get("result_path", ""),
                    "rung_index": result.get("rung_index"),
                    "rung_name": result.get("rung_name", ""),
                },
            )
        )
    return candidates


def collect_linesearch_entries(experiment_root, size_labels):
    candidates = []
    for summary_path in sorted(experiment_root.glob("*/final/summary.json")):
        if "/rung_" in str(summary_path):
            continue
        summary = load_json(summary_path)
        experiment_name = summary.get("experiment_name", "")
        method = infer_method(experiment_name, summary.get("train_script", ""))
        if not is_linesearch_method(method):
            continue
        family = infer_family(experiment_name).upper()
        model_size = infer_model_size(experiment_name)
        size_label = size_labels.get((family, model_size), model_size)
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
                source_path=summary_path,
                metadata={
                    "kind": "linesearch_final",
                },
            )
        )
    return candidates


def aggregate_candidates(candidates):
    grouped = {}
    for candidate in candidates:
        key = (candidate["family"], candidate["model_size"], candidate["method"])
        grouped.setdefault(key, []).append(candidate)

    entries = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda item: (
                float("inf") if get_total_time_hours(item) is None else float(get_total_time_hours(item)),
                float("inf") if item.get("loss") is None else float(item["loss"]),
                item["source_path"],
            ),
        )
        selected = min(
            rows,
            key=lambda item: (
                float("inf") if item.get("loss") is None else float(item["loss"]),
                float("inf") if get_total_time_hours(item) is None else float(get_total_time_hours(item)),
                item["source_path"],
            ),
        )
        entries.append(
            {
                "family": key[0],
                "model_size": key[1],
                "size_label": selected["size_label"],
                "method": key[2],
                "selection_rule": "min_loss",
                "selected": selected,
                "candidates": rows,
            }
        )
    return entries


def build_payload_from_experiment_root(experiment_root, size_labels):
    candidates = []
    candidates.extend(collect_serial_halving_entries(experiment_root, size_labels))
    candidates.extend(collect_linesearch_entries(experiment_root, size_labels))
    return {
        # "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "experiment_root": str(experiment_root.resolve()),
        "entry_count": len(candidates),
        "entries": aggregate_candidates(candidates),
    }


def render_table(
    payload,
    family_order,
    explicit_columns,
    method_order,
    method_labels,
    loss_decimals,
    rows_per_method,
):
    entries = payload["entries"]
    if not entries:
        experiment_root = payload.get("experiment_root", "")
        hint = ""
        if experiment_root:
            hint = f" Input payload experiment_root={experiment_root!r}."
        raise ValueError(
            "No experiment entries found to render."
            + hint
            + " Re-run collect_experiment_table_data.py with the correct --experiment-root."
        )
    entry_map = build_entry_map(entries)
    candidate_map = build_candidate_map(entries)
    available_families = []
    if family_order:
        for family in family_order:
            if any(entry["family"].upper() == family for entry in entries):
                available_families.append(family)
    else:
        preferred = ["GPT", "LLAMA"]
        discovered = []
        for entry in entries:
            family = entry["family"].upper()
            if family not in discovered:
                discovered.append(family)
        available_families.extend([family for family in preferred if family in discovered])
        available_families.extend([family for family in discovered if family not in available_families])
    discovered_columns = discover_columns(entries, available_families)
    family_columns = {}
    for family in available_families:
        family_columns[family] = explicit_columns.get(family, discovered_columns.get(family, []))

    total_columns = max([len(columns) for columns in family_columns.values()] or [1])
    body_methods = collect_body_methods(entries, method_order)
    linesearch_methods = collect_linesearch_methods(entries, method_order)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{build_tabular_spec(total_columns)}}}")

    for family in available_families:
        lines.append("\\hline")
        lines.append(family_header_row(family, family_columns[family], total_columns))
        lines.append("\\hline")
        lines.append("")
        for method in body_methods:
            display_label = method_labels.get(method, method)
            lines.append(method_header_row(display_label, total_columns))
            lines.append("\\hline")
            for row_index in range(rows_per_method):
                lines.append(
                    method_data_row(
                        method=method,
                        family=family,
                        columns=family_columns[family],
                        total_columns=total_columns,
                        candidate_map=candidate_map,
                        row_index=row_index,
                        loss_decimals=loss_decimals,
                    )
                )
                lines.append("\\hline")
            lines.append("")
        for method in linesearch_methods:
            lines.append(
                linesearch_row(
                    label=method_labels.get(method, method),
                    method=method,
                    family=family,
                    columns=family_columns[family],
                    total_columns=total_columns,
                    entry_map=entry_map,
                    loss_decimals=loss_decimals,
                )
            )
            lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparison with total time and loss}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def curve_points(records, x_key, y_key):
    points = []
    for record in records:
        if x_key not in record or y_key not in record:
            continue
        try:
            x_value = float(record[x_key])
            y_value = float(record[y_key])
        except (TypeError, ValueError):
            continue
        points.append((x_value, y_value))
    points.sort(key=lambda item: item[0])
    deduped = []
    last_x = None
    for x_value, y_value in points:
        if last_x is not None and x_value == last_x:
            deduped[-1] = (x_value, y_value)
        else:
            deduped.append((x_value, y_value))
            last_x = x_value
    return deduped


def summary_for_records_path(records_path):
    summary_path = records_path.with_name("summary.json")
    if summary_path.exists():
        return load_json(summary_path)
    return {}


def run_dir_for_records_path(experiment_root, records_path):
    relative_parts = records_path.relative_to(experiment_root).parts
    if not relative_parts:
        return experiment_root
    return experiment_root / relative_parts[0]


def curve_metadata(experiment_root, records_path):
    run_dir = run_dir_for_records_path(experiment_root, records_path)
    summary = summary_for_records_path(records_path)
    experiment_name = summary.get("experiment_name") or infer_experiment_name_from_run_dir(run_dir)
    train_script = summary.get("train_script", "")
    family = infer_family(experiment_name).upper()
    model_size = infer_model_size(experiment_name)
    method = infer_method(experiment_name, train_script)
    trial_id = summary.get("trial_id") or records_path.parent.name
    if records_path.parent.name == "final":
        trial_id = summary.get("trial_id") or "final"
    trial_count_match = re.search(r"num_trials_(\d+)", run_dir.name)
    max_iters_match = re.search(r"maxiters_(\d+)", run_dir.name)
    total_time_hours = summary.get("elapsed_wall_clock_hours")
    if total_time_hours is None:
        total_time_hours = summary.get("wall_clock_hours")
    return {
        "run_dir": run_dir,
        "experiment_name": experiment_name,
        "family": family,
        "model_size": model_size,
        "method": method,
        "trial_id": trial_id,
        "num_trials": trial_count_match.group(1) if trial_count_match else "",
        "max_iters": max_iters_match.group(1) if max_iters_match else "",
        "total_time_hours": total_time_hours,
        "records_path": records_path,
        "scheduler": summary.get("scheduler", ""),
        "summary_learning_rate": summary.get("learning_rate"),
    }


def collect_training_curves(experiment_root, x_key, y_key):
    curves = []
    candidate_paths = []
    candidate_paths.extend(experiment_root.glob("*/shared_trials/*/records.jsonl"))
    candidate_paths.extend(experiment_root.glob("*/final/records.jsonl"))
    for records_path in sorted(set(candidate_paths)):
        records = load_jsonl(records_path)
        points = curve_points(records, x_key, y_key)
        if not points:
            continue
        metadata = curve_metadata(experiment_root, records_path)
        record_time = total_hours_from_records(records)
        if record_time is not None:
            metadata["total_time_hours"] = record_time
        curves.append(
            {
                "metadata": metadata,
                "points": points,
                "records": records,
            }
        )
    return curves


def curve_label(metadata):
    family = metadata["family"]
    model_size = metadata["model_size"]
    trial_id = metadata["trial_id"]
    run_name = metadata["run_dir"].name
    run_bits = []
    if metadata["num_trials"]:
        run_bits.append(f"n{metadata['num_trials']}")
    if metadata["max_iters"]:
        run_bits.append(f"i{metadata['max_iters']}")
    if metadata["total_time_hours"] is not None:
        run_bits.append(f"{float(metadata['total_time_hours']):.2f}h")
    run_suffix = f" ({', '.join(run_bits)})" if run_bits else ""
    if family != "UNKNOWN" and model_size != "UNKNOWN":
        return f"{family} {model_size} {trial_id}{run_suffix}"
    return f"{run_name} {trial_id}"


def curve_group_key(metadata, group_by):
    method = metadata["method"]
    family = metadata["family"]
    model_size = metadata["model_size"]
    num_trials = metadata["num_trials"]
    if group_by == "method":
        return (method,)
    if group_by == "method_model":
        return (method, family, model_size)
    return (method, family, model_size, num_trials)


def curve_group_title(key, group_by):
    if group_by == "method":
        return f"{key[0]} training curves"
    if group_by == "method_model":
        return f"{key[0]} {key[1]} {key[2]} training curves"
    trial_label = f"n{key[3]}" if key[3] else "single run"
    return f"{key[0]} {key[1]} {key[2]} {trial_label} training curves"


def curve_group_filename(key, y_key):
    parts = [slugify(str(part)) for part in key if str(part)]
    return "_".join(parts + [slugify(y_key), "curves"])


def total_curve_hours(curves):
    total = 0.0
    found_any = False
    for curve in curves:
        value = curve["metadata"].get("total_time_hours")
        if value is None:
            continue
        total += float(value)
        found_any = True
    return total if found_any else None


def format_curve_learning_rate(points):
    if not points:
        return ""
    first_lr = points[0][1]
    last_lr = points[-1][1]
    if first_lr == last_lr:
        return f"lr {first_lr:.3g}"
    return f"lr {first_lr:.3g}->{last_lr:.3g}"


def scheduler_lr_value(step, base_lr, scheduler_name, total_iters, warmup_iters=100, floor_ratio=0.1):
    if base_lr is None or total_iters is None:
        return None
    base_lr = float(base_lr)
    total_iters = int(total_iters)
    scheduler_name = str(scheduler_name or "")
    if step < warmup_iters:
        return base_lr * (step + 1) / float(warmup_iters + 1)
    floor_lr = base_lr * float(floor_ratio)
    if scheduler_name in {"inverse_square_root", "inv_sqrt"}:
        return base_lr * math.sqrt(float(warmup_iters + 1) / float(max(step + 1, warmup_iters + 1)))
    if total_iters <= warmup_iters:
        return floor_lr
    decay_ratio = (step - warmup_iters) / float(total_iters - warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    if scheduler_name in {"cosine", "cosine_10pct"}:
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    elif scheduler_name in {"linear", "linear_10pct"}:
        coeff = 1.0 - decay_ratio
    else:
        return None
    return floor_lr + coeff * (base_lr - floor_lr)


def points_are_effectively_constant(points):
    if len(points) < 2:
        return True
    first_value = float(points[0][1])
    for _, value in points[1:]:
        if not math.isclose(float(value), first_value, rel_tol=1e-9, abs_tol=1e-12):
            return False
    return True


def reconstruct_scheduler_lr_points(curve, x_key):
    metadata = curve["metadata"]
    scheduler_name = metadata.get("scheduler", "")
    if scheduler_name not in {"cosine", "cosine_10pct", "linear", "linear_10pct", "inverse_square_root", "inv_sqrt"}:
        return None
    records = curve.get("records", [])
    if not records:
        return None
    total_iters = metadata.get("max_iters")
    if not total_iters:
        return None
    base_lr = None
    for record in records:
        if record.get("learning_rate") is not None:
            base_lr = float(record["learning_rate"])
            break
    if base_lr is None:
        base_lr = metadata.get("summary_learning_rate")
    if base_lr is None:
        return None

    points = []
    for record in records:
        if record.get("step") is None or record.get(x_key) is None:
            continue
        try:
            step = int(record["step"])
            x_value = float(record[x_key])
        except (TypeError, ValueError):
            continue
        lr_value = scheduler_lr_value(
            step=step,
            base_lr=base_lr,
            scheduler_name=scheduler_name,
            total_iters=total_iters,
        )
        if lr_value is None:
            return None
        points.append((x_value, float(lr_value)))
    if not points:
        return None
    points.sort(key=lambda item: item[0])
    deduped = []
    last_x = None
    for x_value, y_value in points:
        if last_x is not None and x_value == last_x:
            deduped[-1] = (x_value, y_value)
        else:
            deduped.append((x_value, y_value))
            last_x = x_value
    return deduped


def stable_curve_color(metadata):
    trial_id = metadata.get("trial_id") or "unknown"
    digest = hashlib.md5(str(trial_id).encode("utf-8")).hexdigest()
    return f"C{int(digest[:8], 16) % 10}"


def best_point(points, minimize=True):
    if not points:
        return None
    if minimize:
        return min(points, key=lambda item: (item[1], item[0]))
    return max(points, key=lambda item: (item[1], item[0]))


def annotate_curve_extrema(ax, points, color, y_key):
    if not points:
        return
    minimize = "loss" in y_key.lower()
    final_x, final_y = points[-1]
    best_x, best_y = best_point(points, minimize=minimize)

    ax.scatter([final_x], [final_y], color=color, s=18, marker="o", zorder=3)
    ax.annotate(
        f"final {final_y:.4f}",
        xy=(final_x, final_y),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=7,
        color=color,
        alpha=0.9,
    )

    ax.scatter([best_x], [best_y], color=color, s=22, marker="x", zorder=3)
    best_label = "best" if (best_x, best_y) != (final_x, final_y) else "best/final"
    ax.annotate(
        f"{best_label} {best_y:.4f}",
        xy=(best_x, best_y),
        xytext=(5, -9),
        textcoords="offset points",
        fontsize=7,
        color=color,
        alpha=0.9,
    )


def plot_training_curves(
    experiment_root,
    output_dir,
    x_key,
    y_key,
    image_format,
    max_legend_items,
    group_by,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = collect_training_curves(experiment_root, x_key, y_key)
    lr_curves = collect_training_curves(experiment_root, x_key, "learning_rate")
    lr_points_by_records_path = {
        curve["metadata"]["records_path"]: curve["points"]
        for curve in lr_curves
    }
    for curve in lr_curves:
        records_path = curve["metadata"]["records_path"]
        lr_points = curve["points"]
        reconstructed = reconstruct_scheduler_lr_points(curve, x_key)
        if reconstructed and (not lr_points or points_are_effectively_constant(lr_points)):
            lr_points_by_records_path[records_path] = reconstructed
    if not curves:
        raise ValueError(
            f"No curves found under {experiment_root} with x={x_key!r} and y={y_key!r}."
        )

    grouped = {}
    for curve in curves:
        key = curve_group_key(curve["metadata"], group_by)
        grouped.setdefault(key, []).append(curve)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for group_key, method_curves in sorted(grouped.items()):
        method_curves.sort(
            key=lambda curve: (
                curve["metadata"]["family"],
                model_size_sort_key(curve["metadata"]["model_size"]),
                curve["metadata"]["run_dir"].name,
                curve["metadata"]["trial_id"],
            )
        )
        has_lr_points = any(
            curve["metadata"]["records_path"] in lr_points_by_records_path
            for curve in method_curves
        )
        if has_lr_points:
            fig, (ax, ax_lr) = plt.subplots(
                2,
                1,
                figsize=(11, 8.5),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )
        else:
            fig, ax = plt.subplots(figsize=(11, 7))
            ax_lr = None
        for index, curve in enumerate(method_curves):
            points = curve["points"]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            label = curve_label(curve["metadata"]) if index < max_legend_items else "_nolegend_"
            lr_points = lr_points_by_records_path.get(curve["metadata"]["records_path"])
            if label != "_nolegend_":
                lr_label = format_curve_learning_rate(lr_points)
                if lr_label:
                    label = f"{label}, {lr_label}"
            color = stable_curve_color(curve["metadata"])
            ax.plot(xs, ys, linewidth=1.2, alpha=0.72, label=label, color=color)
            annotate_curve_extrema(ax, points, color, y_key)
            if ax_lr is not None and lr_points:
                lr_xs = [point[0] for point in lr_points]
                lr_ys = [point[1] for point in lr_points]
                ax_lr.plot(lr_xs, lr_ys, linewidth=1.0, alpha=0.72, color=color)

        ax.set_title(curve_group_title(group_key, group_by))
        if ax_lr is None:
            ax.set_xlabel(x_key.replace("_", " "))
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_ylabel(y_key.replace("_", " "))
        ax.grid(True, alpha=0.25)
        if ax_lr is not None:
            ax_lr.set_xlabel(x_key.replace("_", " "))
            ax_lr.set_ylabel("learning rate")
            ax_lr.grid(True, alpha=0.25)
            positive_lrs = [
                point[1]
                for curve in method_curves
                for point in lr_points_by_records_path.get(curve["metadata"]["records_path"], [])
                if point[1] > 0
            ]
            if positive_lrs and max(positive_lrs) / min(positive_lrs) >= 10:
                ax_lr.set_yscale("log")
        if method_curves and max_legend_items > 0:
            legend_title = "trials"
            if len(method_curves) > max_legend_items:
                legend_title = f"first {max_legend_items} of {len(method_curves)} trials"
            ax.legend(title=legend_title, fontsize=7, loc="best")
        aggregate_hours = total_curve_hours(method_curves)
        if aggregate_hours is not None:
            ax.text(
                0.01,
                0.01,
                f"total trial time: {aggregate_hours:.2f}h",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
            )
        fig.tight_layout()
        output_path = output_dir / f"{curve_group_filename(group_key, y_key)}.{image_format}"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        written_paths.append(output_path)
    return written_paths


def main():
    args = parse_args()
    size_labels = parse_size_labels(args.size_label)
    if args.plot_curves:
        written_paths = plot_training_curves(
            experiment_root=Path(args.experiment_root),
            output_dir=Path(args.plot_output_dir),
            x_key=args.curve_x,
            y_key=args.curve_metric,
            image_format=args.plot_format,
            max_legend_items=max(0, int(args.plot_max_legend_items)),
            group_by=args.curve_group_by,
        )
        for path in written_paths:
            print(path)
        return

    if args.input:
        payload = load_payload(args.input)
    else:
        payload = build_payload_from_experiment_root(Path(args.experiment_root), size_labels)
    family_order = [item.upper() for item in args.family]
    explicit_columns = parse_columns(args.column)
    method_order = args.method
    method_labels = {
        "cosine": "cosine",
        "muon": "muon",
        "schedulefree_adam": "schedulefree_adam",
        "linesearch_adam": args.linesearch_label,
        "linesearch_muon": args.linesearch_muon_label,
    }
    method_labels.update(parse_method_labels(args.method_label))

    table_text = render_table(
        payload=payload,
        family_order=family_order,
        explicit_columns=explicit_columns,
        method_order=method_order,
        method_labels=method_labels,
        loss_decimals=args.loss_decimals,
        rows_per_method=args.rows_per_method,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table_text, encoding="utf-8")
    else:
        print(table_text)


if __name__ == "__main__":
    main()
