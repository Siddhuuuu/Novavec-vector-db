"""
Plot Recall@10 vs QPS and Recall@10 vs p99 Pareto curves
from benchmark results produced by bench_recall.py.

Usage
-----
    python benchmarks/plot_pareto.py \
        [--input benchmarks/results/hnsw_pareto.json] \
        [--output benchmarks/results/pareto_curve.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not data:
        print(f"ERROR: no results in {path}")
        sys.exit(1)
    return data


def plot(data: list[dict], output: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for CI/headless
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        print("Install it with: pip install matplotlib")
        return

    recalls   = [d.get("recall_at_10", 0) for d in data]
    qps_vals  = [d.get("qps", 0)          for d in data]
    p99_vals  = [d.get("p99_ms", 0)       for d in data]
    p50_vals  = [d.get("p50_ms", 0)       for d in data]
    ef_vals   = [d.get("ef_search", 0)    for d in data]

    dataset   = data[0].get("dataset", "unknown")
    n_vectors = data[0].get("n_vectors", 0)
    M_val     = data[0].get("M", 16)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"NovaVec HNSW Pareto Curve — {dataset}  "
        f"({n_vectors:,} vectors, M={M_val})",
        fontsize=13,
        fontweight="bold",
    )

    # ---- Plot 1: Recall vs QPS ----
    ax1.plot(recalls, qps_vals, "b-o", linewidth=2.0,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    for i, ef in enumerate(ef_vals):
        ax1.annotate(
            f"ef={ef}",
            xy=(recalls[i], qps_vals[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )
    ax1.set_xlabel("Recall@10", fontsize=12)
    ax1.set_ylabel("Queries Per Second (QPS)", fontsize=12)
    ax1.set_title("Recall vs Throughput", fontsize=11)
    ax1.set_xlim(max(0, min(recalls) - 0.02), 1.03)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))

    # Shade the area under the curve to emphasize the Pareto frontier
    ax1.fill_between(recalls, qps_vals, alpha=0.08, color="blue")

    # ---- Plot 2: Recall vs p99 latency ----
    ax2.plot(recalls, p99_vals, "r-o", linewidth=2.0,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    for i, ef in enumerate(ef_vals):
        ax2.annotate(
            f"ef={ef}",
            xy=(recalls[i], p99_vals[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )
    ax2.set_xlabel("Recall@10", fontsize=12)
    ax2.set_ylabel("p99 Latency (ms)", fontsize=12)
    ax2.set_title("Recall vs p99 Latency", fontsize=11)
    ax2.set_xlim(max(0, min(recalls) - 0.02), 1.03)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def print_markdown_table(data: list[dict]) -> None:
    """Print results as a GitHub-flavored Markdown table."""
    print("\n## Benchmark Results\n")
    print(
        "| ef_search | Recall@10 | QPS     | p50 (ms) | p95 (ms) | p99 (ms) |"
    )
    print(
        "|-----------|-----------|---------|----------|----------|----------|"
    )
    for d in data:
        print(
            f"| {d.get('ef_search', '?'):>9} "
            f"| {d.get('recall_at_10', 0):>9.4f} "
            f"| {d.get('qps', 0):>7.0f} "
            f"| {d.get('p50_ms', 0):>8.2f} "
            f"| {d.get('p95_ms', 0):>8.2f} "
            f"| {d.get('p99_ms', 0):>8.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot recall/latency Pareto curve from benchmark results."
    )
    parser.add_argument(
        "--input",
        default="benchmarks/results/hnsw_pareto.json",
        help="Input JSON file from bench_recall.py",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/pareto_curve.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating the PNG, only print the markdown table",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: results file not found: {args.input}")
        print("Run 'python benchmarks/bench_recall.py' first.")
        sys.exit(1)

    data = load_results(args.input)

    # Sort by ef_search ascending for clean plot
    data.sort(key=lambda d: d.get("ef_search", 0))

    if not args.no_plot:
        plot(data, args.output)

    print_markdown_table(data)


if __name__ == "__main__":
    main()
