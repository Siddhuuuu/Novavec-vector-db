"""
Sweep HNSW ef_search values and collect recall/latency/QPS.

Calls bench_runner for each ef value, collects JSON output,
saves combined results for plot_pareto.py.

Usage
-----
    python benchmarks/bench_recall.py --dataset sift-128 --queries 1000

Output
------
    benchmarks/results/hnsw_pareto.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ef_search values to sweep — covers the full recall/latency Pareto frontier
EF_VALUES = [10, 20, 50, 100, 200, 500]

# Where bench_runner binary lives after cmake build
BENCH_RUNNER = str(Path(__file__).parent.parent / "build" / "bench_runner")

# Output directory
RESULTS_DIR = Path(__file__).parent / "results"


def run_benchmark(
    dataset: str,
    ef: int,
    queries: int,
    M: int,
    ef_construction: int,
) -> dict:
    """Run bench_runner for one ef value and return parsed JSON result."""
    output_file = f"/tmp/bench_ef{ef}.json"

    cmd = [
        BENCH_RUNNER,
        "--dataset",         dataset,
        "--index",           "hnsw",
        "--M",               str(M),
        "--ef-construction", str(ef_construction),
        "--ef-search",       str(ef),
        "--queries",         str(queries),
        "--output",          output_file,
    ]

    print(f"  Running: ef_search={ef}…", end=" ", flush=True)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        # Print any stdout from the C++ binary for visibility
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            # Print the summary line
            for line in lines:
                if "Recall" in line or "QPS" in line:
                    print(line.strip(), end=" ")
    except subprocess.CalledProcessError as exc:
        print(f"\nERROR: bench_runner failed (exit {exc.returncode})")
        print(exc.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nERROR: bench_runner not found at '{BENCH_RUNNER}'")
        print("Build the project first: cmake -B build && cmake --build build -j$(nproc)")
        sys.exit(1)

    with open(output_file) as f:
        data = json.load(f)

    data["ef_search"] = ef
    print(f"  recall={data.get('recall_at_10', 0):.3f}  "
          f"p99={data.get('p99_ms', 0):.1f}ms  "
          f"qps={data.get('qps', 0):.0f}")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep HNSW ef_search and collect Pareto curve data."
    )
    parser.add_argument(
        "--dataset", default="sift-128",
        help="Dataset name or path (default: sift-128)"
    )
    parser.add_argument(
        "--queries", type=int, default=1000,
        help="Number of query vectors (default: 1000)"
    )
    parser.add_argument(
        "--M", type=int, default=16,
        help="HNSW M parameter (default: 16)"
    )
    parser.add_argument(
        "--ef-construction", type=int, default=200,
        help="HNSW ef_construction (default: 200)"
    )
    parser.add_argument(
        "--ef-values", type=int, nargs="+", default=EF_VALUES,
        help=f"ef_search values to sweep (default: {EF_VALUES})"
    )
    args = parser.parse_args()

    if not os.path.exists(BENCH_RUNNER):
        print(f"ERROR: bench_runner not found at '{BENCH_RUNNER}'")
        print("Build first: cmake -B build && cmake --build build -j$(nproc)")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sweeping ef_search={args.ef_values} on dataset='{args.dataset}'")
    print(f"Index: HNSW M={args.M}, ef_construction={args.ef_construction}")
    print(f"Queries: {args.queries}\n")

    results = []
    for ef in sorted(args.ef_values):
        data = run_benchmark(
            dataset         = args.dataset,
            ef              = ef,
            queries         = args.queries,
            M               = args.M,
            ef_construction = args.ef_construction,
        )
        results.append(data)

    output_path = RESULTS_DIR / "hnsw_pareto.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\nRun 'python benchmarks/plot_pareto.py' to generate the Pareto curve.")

    # Print summary table
    print("\n## Summary Table\n")
    print(f"{'ef_search':>10}  {'Recall@10':>10}  {'QPS':>8}  {'p50 ms':>8}  {'p99 ms':>8}")
    print("-" * 55)
    for d in results:
        print(
            f"{d['ef_search']:>10}  "
            f"{d.get('recall_at_10', 0):>10.4f}  "
            f"{d.get('qps', 0):>8.0f}  "
            f"{d.get('p50_ms', 0):>8.2f}  "
            f"{d.get('p99_ms', 0):>8.2f}"
        )


if __name__ == "__main__":
    main()
