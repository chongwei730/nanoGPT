#!/usr/bin/env python3
"""
Parse finetune_*.txt files to extract validation loss and line-search LR,
then plot loss vs step and LR vs step (LR on secondary log axis).

Usage:
  python3 scripts/plot_finetune.py --pattern "finetune_*.txt" --out finetune_loss_lr.png
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_file(path):
    last_step = None
    loss_steps = []
    loss_vals = []
    lr_steps = []
    lr_vals = []

    step_re = re.compile(r"\[step\s+(\d+)\]")
    loss_re = re.compile(r"val loss:\s*([0-9eE+\-.]+)")
    lr_re = re.compile(r"LINESEARCH LR:\s*([0-9eE+\-.]+)")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = step_re.search(line)
            if m:
                try:
                    last_step = int(m.group(1))
                except Exception:
                    last_step = None

            m_loss = loss_re.search(line)
            if m_loss:
                val = float(m_loss.group(1))
                step = last_step
                if step is None:
                    # fallback: try to extract step on the same line if present
                    m2 = step_re.search(line)
                    step = int(m2.group(1)) if m2 else None
                if step is not None:
                    loss_steps.append(step)
                    loss_vals.append(val)

            m_lr = lr_re.search(line)
            if m_lr:
                val = float(m_lr.group(1))
                step = last_step
                if step is None:
                    m2 = step_re.search(line)
                    step = int(m2.group(1)) if m2 else None
                if step is not None:
                    lr_steps.append(step)
                    lr_vals.append(val)

    return {"loss_steps": loss_steps, "loss_vals": loss_vals, "lr_steps": lr_steps, "lr_vals": lr_vals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="../finetune_*.txt", help="glob pattern for finetune logs")
    parser.add_argument("--out", default="finetune_loss_lr.png", help="output image file")
    parser.add_argument("--show", action="store_true", help="show the plot interactively (if available)")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found for pattern: {args.pattern}")
        return

    plt.figure(figsize=(10, 6))

    for path in files:
        d = parse_file(path)
        # sort by step for plotting
        if d["loss_steps"]:
            pairs = sorted(zip(d["loss_steps"], d["loss_vals"]))
            steps, losses = zip(*pairs)
        else:
            steps, losses = [], []

        if d["lr_steps"]:
            lr_pairs = sorted(zip(d["lr_steps"], d["lr_vals"]))
            lr_steps, lrs = zip(*lr_pairs)
        else:
            lr_steps, lrs = [], []

        label = os.path.basename(path)
        ax = plt.gca()
        if steps:
            ax.plot(steps, losses, label=f"loss: {label}")

        if lr_steps:
            ax2 = ax.twinx()
            ax2.plot(lr_steps, lrs, linestyle="--", color="C1", label=f"lr: {label}")
            ax2.set_yscale("log")
            ax2.set_ylabel("LR (log scale)")

    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.set_title("Finetune: Validation Loss and LINESEARCH LR")

    # build combined legend
    lines, labels = ax.get_legend_handles_labels()
    # include lines from ax2 if present
    try:
        ax2_lines, ax2_labels = ax2.get_legend_handles_labels()
        lines += ax2_lines
        labels += ax2_labels
    except Exception:
        pass

    if lines:
        ax.legend(lines, labels, loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")

    if args.show:
        try:
            plt.show()
        except Exception:
            print("Could not show plot interactively. Use --out to save the image.")


if __name__ == "__main__":
    main()
