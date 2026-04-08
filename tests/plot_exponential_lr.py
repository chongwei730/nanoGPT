#!/usr/bin/env python3
"""Plot exponential distributions with different means (lr values).

Generates PDF curves and sample histograms for several mean values so you
can visually compare how the exponential distribution changes when the
mean (used here to represent `lr`) varies.

Usage:
  python tests/plot_exponential_lr.py --save
  python tests/plot_exponential_lr.py --show

Requires: numpy, matplotlib
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_exponentials(lr_values, samples=10000, save_path=None, show=False):
    max_lr = max(lr_values)
    x_max = max(5 * max_lr, 0.1)
    x = np.linspace(0, x_max, 1000)

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

    # Plot PDFs
    ax_pdf = axes[0]
    for lr in lr_values:
        # exponential rate (lambda) is 1/mean
        lam = 1.0 / lr if lr > 0 else 1.0
        pdf = lam * np.exp(-lam * x)
        ax_pdf.plot(x, pdf, label=f"mean={lr}")
    ax_pdf.set_title("Exponential PDF for different means (lr)")
    ax_pdf.set_xlabel("x")
    ax_pdf.set_ylabel("pdf(x)")
    ax_pdf.legend()

    # Plot histograms of samples
    ax_hist = axes[1]
    bins = 80
    for lr in lr_values:
        lam = 1.0 / lr if lr > 0 else 1.0
        samples_arr = np.random.default_rng(seed=int(lr * 1e6) % 2**32).exponential(scale=lr, size=samples)
        ax_hist.hist(samples_arr, bins=bins, density=True, alpha=0.5, label=f"mean={lr}")
    ax_hist.set_title(f"Histogram of {samples} samples (density normalized)")
    ax_hist.set_xlabel("x")
    ax_hist.set_ylabel("density")
    ax_hist.set_xlim(0, x_max)
    ax_hist.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", nargs="*", type=float,
                        default=[0.01, 0.03, 0.05, 0.08, 0.1],
                        help="List of lr values to use as means for the exponential distributions")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--save", action="store_true", help="Save figure to outputs/exponential_lr.png")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    args = parser.parse_args()

    out_path = None
    if args.save:
        out_path = os.path.join("outputs", "exponential_lr.png")

    plot_exponentials(args.lr, samples=args.samples, save_path=out_path, show=args.show)


if __name__ == "__main__":
    main()
