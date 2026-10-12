"""Regenerate compact public result figures from the released CSV tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_benchmark(rows: list[dict[str, str]], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    models = [row["model"] for row in rows]
    complex_acc = [float(row["complex_accuracy"]) for row in rows]
    joint_acc = [float(row["joint_accuracy"]) for row in rows]
    positions = range(len(models))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar([x - 0.2 for x in positions], complex_acc, width=0.4, label="Complex accuracy", color="#566273")
    ax.bar([x + 0.2 for x in positions], joint_acc, width=0.4, label="Joint accuracy", color="#238a77")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(list(positions), models, rotation=35, ha="right")
    ax.set_ylim(0, 85)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot released EndoCA result summaries.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/paper/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    for suite, title in [("core", "EndoCA-Core"), ("diagnostic", "EndoCA-Diagnostic")]:
        rows = read_csv(args.results_dir / f"endoca_{suite}_results.csv")
        plot_benchmark(rows, args.output_dir / f"endoca_{suite}_summary.png", title)


if __name__ == "__main__":
    main()
