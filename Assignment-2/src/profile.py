"""
profile.py
===========

Performance profiling script for fuzzy search data structure implementations.

This module benchmarks four fuzzy search data structures described in the specification:
    - array (linear brute-force search using `fuzzy_search.levenshtein`)
    - BK_trees (BKTree from `fuzzy_search`)
    - VP_trees (VPTree from `fuzzy_search`)
    - trie (Trie from `fuzzy_search`)

Each structure is tested for build and search (`find_closest`) performance across
various dataset sizes. The script outputs timing statistics and a CSV report.

The generated DataFrame contains:
    ["data_structure", "task", "size", "times"]

Key Features
------------
- Loads string dataset from: `src/string_dataset.npy`
- Uses `timeit.Timer` for accurate timing
- Performs configurable warmup and repetition cycles
- Supports CLI configuration for dataset, sizes, repetitions, and structures
- Outputs results to CSV (`src/profile_results.csv` by default)

Example
-------
    $ python profile.py --sizes 100,1000,5000 --repetitions 5 --warmup 2

Output
------
A CSV file of per-call timing data and a printed summary to stdout.

Dependencies
------------
- numpy
- pandas
- fuzzy_search (with BKTree, VPTree, Trie, levenshtein)
"""

import argparse
import copy
import gc
import os
import random
import statistics
import string
import sys
import timeit
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from fuzzy_search import BKTree, Trie, VPTree, levenshtein

# ---- Configuration / defaults ----
DEFAULT_SIZES = [
    10,
    50,
    100,
    # 500,
    # 1000,
    # 5000,
    # 10000,
    # 50_000,
    # 100_000,
    # 500_000,
    # 1_000_000,
]

DEFAULT_SEED = 42
DATASET_PATH = os.path.join("src", "string_dataset.npy")

# Global variable to hold most recently built structure between timer calls
_BUILT_STRUCT: Any = None


# ---------------------------------------------------------------------------
# Dataset Utilities
# ---------------------------------------------------------------------------


def load_dataset(path: str = DATASET_PATH) -> List[str]:
    """
    Load a string dataset from a .npy file.

    Parameters
    ----------
    path : str, optional
        Path to the .npy dataset file.

    Returns
    -------
    list of str
        List of string entries loaded from the dataset.
    """
    print(f"Loading dataset from {path} ...", file=sys.stderr)
    arr = np.load(path, allow_pickle=True)
    data = arr.tolist()
    print(f"Loaded dataset of length {len(data)}", file=sys.stderr)
    return data


def random_query(length: int = 10, seed: int = DEFAULT_SEED) -> str:
    """
    Generate a reproducible random alphanumeric query string.

    Parameters
    ----------
    length : int, optional
        Length of query string (default 10).
    seed : int, optional
        Random seed for deterministic output (default 42).

    Returns
    -------
    str
        Random query string.
    """
    rng = random.Random(seed)
    return "".join(rng.choices(string.ascii_letters + string.digits, k=length))


# ---------------------------------------------------------------------------
# Callable Generators for timeit
# ---------------------------------------------------------------------------


def make_array_build_callable(sublist: List[str]) -> Callable[[], None]:
    """Return a callable that deep-copies the list (baseline 'build' operation)."""

    def build():
        nonlocal sublist
        global _BUILT_STRUCT
        _BUILT_STRUCT = copy.deepcopy(sublist)

    return build


def make_array_find_callable(
    query: str, count: int = 10
) -> Callable[[], List[Tuple[str, int]]]:
    """Return a callable that performs linear fuzzy search using Levenshtein distance."""

    def find():
        global _BUILT_STRUCT
        assert isinstance(_BUILT_STRUCT, list), "Array not built"
        results = [(w, levenshtein(query, w)) for w in _BUILT_STRUCT]
        results.sort(key=lambda x: x[1])
        return results[:count]

    return find


def make_bktree_build_callable(sublist: List[str]) -> Callable[[], None]:
    """Return a callable that builds a BKTree from the given sublist."""

    def build():
        nonlocal sublist
        global _BUILT_STRUCT
        tree = BKTree()
        tree.build(sublist)
        _BUILT_STRUCT = tree

    return build


def make_bktree_find_callable(
    query: str, count: int = 10
) -> Callable[[], List[Tuple[str, int]]]:
    """Return a callable that performs fuzzy search on the current BKTree."""

    def find():
        global _BUILT_STRUCT
        assert isinstance(_BUILT_STRUCT, BKTree), "BKTree not built"
        return _BUILT_STRUCT.find_closest(query, count=count)

    return find


def make_vptree_build_callable(sublist: List[str]) -> Callable[[], None]:
    """Return a callable that builds a VPTree from the given sublist."""

    def build():
        nonlocal sublist
        global _BUILT_STRUCT
        tree = VPTree()
        tree.build(sublist)
        _BUILT_STRUCT = tree

    return build


def make_vptree_find_callable(
    query: str, count: int = 10
) -> Callable[[], List[Tuple[str, int]]]:
    """Return a callable that performs fuzzy search on the current VPTree."""

    def find():
        global _BUILT_STRUCT
        assert isinstance(_BUILT_STRUCT, VPTree), "VPTree not built"
        return _BUILT_STRUCT.find_closest(query, count=count)

    return find


def make_trie_build_callable(sublist: List[str]) -> Callable[[], None]:
    """Return a callable that builds a Trie from the given sublist."""

    def build():
        nonlocal sublist
        global _BUILT_STRUCT
        t = Trie()
        t.build(sublist)
        _BUILT_STRUCT = t

    return build


def make_trie_find_callable(
    query: str, count: int = 10
) -> Callable[[], List[Tuple[str, int]]]:
    """Return a callable that performs fuzzy search on the current Trie."""

    def find():
        global _BUILT_STRUCT
        assert isinstance(_BUILT_STRUCT, Trie), "Trie not built"
        return _BUILT_STRUCT.find_closest(query, count=count)

    return find


# ---------------------------------------------------------------------------
# Timing and Profiling Helpers
# ---------------------------------------------------------------------------


def _clear_built_struct() -> None:
    """Clear the globally stored built structure and run garbage collection."""
    global _BUILT_STRUCT
    _BUILT_STRUCT = None
    gc.collect()


def run_warmup(
    build_callable: Callable[[], None],
    find_callable: Callable[[], Any],
    warmup_cycles: int,
) -> None:
    """Execute a few unmeasured warmup cycles to stabilize performance."""
    print(f"  Warmup: {warmup_cycles} cycles ...", file=sys.stderr)
    for _ in range(warmup_cycles):
        build_callable()
        find_callable()
        _clear_built_struct()


def time_callable(
    callable_obj: Callable[[], Any],
    per_repetition_stmt_count: int,
    repeat_times: int = 1,
) -> List[float]:
    """
    Measure average execution time per call of a callable using `timeit.Timer`.

    Returns
    -------
    list of float
        Per-call execution times in seconds.
    """
    timer = timeit.Timer(callable_obj)
    raw_times = timer.repeat(repeat=repeat_times, number=per_repetition_stmt_count)
    return [t / per_repetition_stmt_count for t in raw_times]


def summarystats(times: List[float]) -> Tuple[float, float, float, float]:
    """Compute min, mean, max, and standard deviation of timing data."""
    if not times:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(times),
        statistics.mean(times),
        max(times),
        statistics.stdev(times) if len(times) > 1 else 0.0,
    )


def profile_one_datastructure(
    name: str,
    sizes: List[int],
    dataset: List[str],
    warmup_cycles: int,
    repetitions: int,
    per_repetition_stmt_count: int,
) -> List[Dict[str, Any]]:
    """
    Profile build and search times for a given fuzzy search data structure.

    Parameters
    ----------
    name : str
        One of {"array", "BK_trees", "VP_trees", "trie"}.
    sizes : list of int
        Dataset sizes to benchmark.
    dataset : list of str
        Full dataset to slice.
    warmup_cycles : int
        Number of warmup iterations before timing.
    repetitions : int
        Number of measured repetitions.
    per_repetition_stmt_count : int
        Number of operations per repetition (for timeit normalization).

    Returns
    -------
    list of dict
        Each dict contains `data_structure`, `task`, `size`, and `times`.
    """
    results_rows: List[Dict[str, Any]] = []

    print(f"\n===== Profiling {name} =====", file=sys.stderr)
    for n in sizes:
        print(f"\nDatastructure: {name}", file=sys.stderr)
        print(f"At size {n}", file=sys.stderr)
        sublist = dataset[:n]
        query = random_query(10)

        # Choose builder & finder
        builders = {
            "array": make_array_build_callable,
            "BK_trees": make_bktree_build_callable,
            "VP_trees": make_vptree_build_callable,
            "trie": make_trie_build_callable,
        }
        finders = {
            "array": make_array_find_callable,
            "BK_trees": make_bktree_find_callable,
            "VP_trees": make_vptree_find_callable,
            "trie": make_trie_find_callable,
        }
        if name not in builders:
            raise ValueError(f"Unknown data structure: {name}")

        build_callable = builders[name](sublist)
        find_callable = finders[name](query, count=10)

        run_warmup(build_callable, find_callable, warmup_cycles)

        build_times_all, find_times_all = [], []
        for rep in range(repetitions):
            print(f"  repetition {rep + 1}/{repetitions} ...", file=sys.stderr)
            _clear_built_struct()
            build_times_all += time_callable(build_callable, per_repetition_stmt_count)
            find_times_all += time_callable(find_callable, per_repetition_stmt_count)
            _clear_built_struct()

        # Summarize
        b_min, b_mean, b_max, b_sd = summarystats(build_times_all)
        f_min, f_mean, f_max, f_sd = summarystats(find_times_all)
        print(
            f"  Build: min={b_min:.6f}, avg={b_mean:.6f}, max={b_max:.6f}, sd={b_sd:.6f}",
            file=sys.stderr,
        )
        print(
            f"  Find : min={f_min:.6f}, avg={f_mean:.6f}, max={f_max:.6f}, sd={f_sd:.6f}",
            file=sys.stderr,
        )

        results_rows.append(
            {
                "data_structure": name,
                "task": "build",
                "size": n,
                "times": build_times_all,
            }
        )
        results_rows.append(
            {
                "data_structure": name,
                "task": "find_closest",
                "size": n,
                "times": find_times_all,
            }
        )

    return results_rows


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def parse_size_list(arg: str) -> List[int]:
    """Parse comma-separated list of integers for --sizes CLI argument."""
    try:
        return [int(p.strip()) for p in arg.split(",") if p.strip()]
    except Exception as e:
        raise argparse.ArgumentTypeError(
            "sizes must be comma separated integers"
        ) from e


def main(argv: List[str] | None = None) -> None:
    """Command-line entrypoint for profiling fuzzy search data structures."""
    parser = argparse.ArgumentParser(
        description="Profile fuzzy_search data structures (build + find_closest)."
    )
    parser.add_argument(
        "--dataset", default=DATASET_PATH, help="Path to src/string_dataset.npy"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup cycles (default 3)"
    )
    parser.add_argument(
        "--repetitions", type=int, default=10, help="Number of repetitions (default 10)"
    )
    parser.add_argument(
        "--per-repetition-stmt-count",
        type=int,
        default=1,
        help="Number of statements per repetition",
    )
    parser.add_argument(
        "--sizes",
        type=parse_size_list,
        default=None,
        help="Comma-separated list of dataset sizes",
    )
    parser.add_argument(
        "--out-csv", default="src/profile_results.csv", help="CSV output path"
    )
    parser.add_argument(
        "--structures",
        default="array,BK_trees,VP_trees,trie",
        help="Comma-separated list of structures to profile",
    )

    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    sizes = args.sizes if args.sizes else DEFAULT_SIZES
    structures = [s.strip() for s in args.structures.split(",") if s.strip()]

    all_rows = []
    for ds in ["array", "BK_trees", "VP_trees", "trie"]:
        if ds in structures:
            all_rows.extend(
                profile_one_datastructure(
                    ds,
                    sizes,
                    dataset,
                    args.warmup,
                    args.repetitions,
                    args.per_repetition_stmt_count,
                )
            )

    df = pd.DataFrame(all_rows, columns=["data_structure", "task", "size", "times"])
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved profiling results to {args.out_csv}", file=sys.stderr)

    print("\n=== Compact Summary ===")
    grouped = (
        df.groupby(["data_structure", "task", "size"])["times"].first().reset_index()
    )
    for _, row in grouped.iterrows():
        mn, mean, mx, sd = summarystats(row["times"])
        print(
            f"{row['data_structure']:>10} | {row['task']:>11} | size={row['size']:>9} | "
            f"min={mn:.6f} avg={mean:.6f} max={mx:.6f} sd={sd:.6f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
