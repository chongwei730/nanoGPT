#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a LaTeX summary table from structured experiment table data."
    )
    parser.add_argument("--input", required=True, help="JSON produced by collect_experiment_table_data.py")
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
        default="Linesearch",
        help="Row label used for the line search row.",
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
    return parser.parse_args()


def normalize_model_size(raw_size):
    raw_size = raw_size.strip()
    if raw_size and raw_size[-1].lower() in {"m", "b"}:
        return raw_size[:-1] + raw_size[-1].upper()
    return raw_size


def load_payload(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
                float("inf") if item.get("tuning_time_hours") is None else float(item["tuning_time_hours"]),
                float("inf") if item.get("loss") is None else float(item["loss"]),
            )
        )
        candidate_map[key] = candidates
    return candidate_map


def collect_body_methods(entries, linesearch_label, explicit_methods):
    if explicit_methods:
        return explicit_methods
    methods = []
    for entry in entries:
        method = entry["method"]
        if method == linesearch_label:
            continue
        if method not in methods:
            methods.append(method)
    return methods or ["Method1", "Method2", "Method3"]


def build_tabular_spec(column_count):
    return "c|" + "|".join(["c|c|c"] * column_count)


def family_header_row(family_name, columns, total_columns):
    cells = [f"\\textbf{{{family_name}}}"]
    padded = list(columns) + [{"label": "Size"}] * (total_columns - len(columns))
    for index, column in enumerate(padded):
        suffix = "|" if index < total_columns - 1 else ""
        cells.append(f"\\multicolumn{{3}}{{c{suffix}}}{{{column['label']}}}")
    return " & ".join(cells) + " \\\\"


def method_header_row(method_label, total_columns):
    cells = [f"\\textbf{{{method_label}}}"]
    for _ in range(total_columns):
        cells.extend(["Tuning time", "Loss", "Spent time"])
    return " & ".join(cells) + " \\\\"


def method_data_row(method, family, columns, total_columns, candidate_map, row_index, loss_decimals):
    cells = [""]
    padded = list(columns) + [{"model_size": None}] * (total_columns - len(columns))
    for column in padded:
        if not column["model_size"]:
            cells.extend(["", "", ""])
            continue
        candidates = candidate_map.get((family, column["model_size"], method), [])
        if row_index >= len(candidates):
            cells.extend(["", "", ""])
            continue
        candidate = candidates[row_index]
        cells.extend(
            [
                format_hours(candidate.get("tuning_time_hours")),
                format_loss(candidate.get("loss"), loss_decimals),
                format_hours(candidate.get("wall_clock_hours")),
            ]
        )
    return " & ".join(cells) + " \\\\"


def linesearch_row(label, family, columns, total_columns, entry_map, loss_decimals):
    cells = [f"\\textbf{{{label}}}"]
    padded = list(columns) + [{"model_size": None}] * (total_columns - len(columns))
    for column in padded:
        if not column["model_size"]:
            cells.extend(["Nah", "", ""])
            continue
        entry = entry_map.get((family, column["model_size"], label))
        loss_value = ""
        spent_time_value = ""
        if entry:
            selected = entry["selected"]
            loss_value = format_loss(selected.get("loss"), loss_decimals)
            spent_time_value = format_hours(selected.get("wall_clock_hours"))
        cells.extend(["Nah", loss_value, spent_time_value])
    return " & ".join(cells) + " \\\\"


def render_table(
    payload,
    family_order,
    explicit_columns,
    method_order,
    method_labels,
    linesearch_label,
    loss_decimals,
    rows_per_method,
):
    entries = payload["entries"]
    entry_map = build_entry_map(entries)
    candidate_map = build_candidate_map(entries)
    discovered_columns = discover_columns(entries, family_order)
    family_columns = {}
    for family in family_order:
        family_columns[family] = explicit_columns.get(family, discovered_columns.get(family, []))

    total_columns = max([len(columns) for columns in family_columns.values()] or [1])
    body_methods = collect_body_methods(entries, linesearch_label, method_order)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{build_tabular_spec(total_columns)}}}")

    for family in family_order:
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
        lines.append(
            linesearch_row(
                label=linesearch_label,
                family=family,
                columns=family_columns[family],
                total_columns=total_columns,
                entry_map=entry_map,
                loss_decimals=loss_decimals,
            )
        )
    lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparison with tuning time, loss, and spent time}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    payload = load_payload(args.input)
    family_order = [item.upper() for item in args.family] if args.family else ["GPT", "LLAMA"]
    explicit_columns = parse_columns(args.column)
    method_order = args.method
    method_labels = parse_method_labels(args.method_label)

    table_text = render_table(
        payload=payload,
        family_order=family_order,
        explicit_columns=explicit_columns,
        method_order=method_order,
        method_labels=method_labels,
        linesearch_label=args.linesearch_label,
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
