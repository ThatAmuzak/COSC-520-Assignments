"""
profile.py

Profiles fuzzy search implementations described in the spec:
 - array (linear brute-force using fuzzy_search.levenshtein)
 - BK_trees (BKTree from fuzzy_search)
 - VP_trees (VPTree from fuzzy_search)
 - trie (Trie from fuzzy_search)

Produces a pandas DataFrame with columns:
  ["data_structure", "task", "size", "times"]

Notes:
 - The dataset is loaded from: src/string_dataset.npy
 - The script uses timeit.Timer and repeats as described in the spec.
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
    500,
    1000,
    5000,
    10000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    # 5_000_000,
    # 10_000_000,
]

DEFAULT_SEED = 42

# Load the large dataset once
DATASET_PATH = os.path.join("src", "string_dataset.npy")


def load_dataset(path: str = DATASET_PATH) -> List[str]:
    """Load dataset from .npy file and return as Python list (allow_pickle=True)."""
    print(f"Loading dataset from {path} ...", file=sys.stderr)
    arr = np.load(path, allow_pickle=True)
    # convert to list to allow slicing and to match spec
    data = arr.tolist()
    print(f"Loaded dataset of length {len(data)}", file=sys.stderr)
    return data


def random_query(length: int = 10, seed: int = DEFAULT_SEED) -> str:
    rng = random.Random(seed)
    return "".join(rng.choices(string.ascii_letters + string.digits, k=length))


# We'll use module-level globals to hold the most recently built structure for the timer callables.
# This avoids having to return values in the timed functions (Timer will call them directly).
_BUILT_STRUCT: Any = None  # holds whichever structure was last built


# ----- Maker functions produce callables that timeit.Timer can call -----
def make_array_build_callable(sublist: List[str]) -> Callable[[], None]:
    """
    Build operation for the 'array' baseline is a deepcopy into a new list.
    The callable will set _BUILT_STRUCT.
    """

    def build():
        nonlocal sublist
        global _BUILT_STRUCT
        # According to spec, build time is deepcopy into a new list
        _BUILT_STRUCT = copy.deepcopy(sublist)

    return build


def make_array_find_callable(
    query: str, count: int = 10
) -> Callable[[], List[Tuple[str, int]]]:
    """
    Returns a callable that performs a linear scan using levenshtein on the previously
    built list stored in _BUILT_STRUCT and returns top `count` entries.
    """

    def find():
        global _BUILT_STRUCT
        assert _BUILT_STRUCT is not None and isinstance(_BUILT_STRUCT, list), (
            "array structure not built"
        )
        # compute distances
        results: List[Tuple[str, int]] = []
        for w in _BUILT_STRUCT:
            d = levenshtein(query, w)
            results.append((w, d))
        # sort by distance and return top `count`
        results.sort(key=lambda x: x[1])
        return results[:count]

    return find


def make_bktree_build_callable(sublist: List[str]) -> Callable[[], None]:
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
    def find():
        global _BUILT_STRUCT
        assert _BUILT_STRUCT is not None and isinstance(_BUILT_STRUCT, BKTree), (
            "BKTree not built"
        )
        return _BUILT_STRUCT.find_closest(query, count=count)

    return find


def make_vptree_build_callable(sublist: List[str]) -> Callable[[], None]:
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
    def find():
        global _BUILT_STRUCT
        assert _BUILT_STRUCT is not None and isinstance(_BUILT_STRUCT, VPTree), (
            "VPTree not built"
        )
        return _BUILT_STRUCT.find_closest(query, count=count)

    return find


def make_trie_build_callable(sublist: List[str]) -> Callable[[], None]:
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
    def find():
        global _BUILT_STRUCT
        assert _BUILT_STRUCT is not None and isinstance(_BUILT_STRUCT, Trie), (
            "Trie not built"
        )
        return _BUILT_STRUCT.find_closest(query, count=count)

    return find


# ----- utility to run warmup and timed repeats -----
def run_warmup(
    build_callable: Callable[[], None],
    find_callable: Callable[[], Any],
    warmup_cycles: int,
) -> None:
    print(f"  Warmup: {warmup_cycles} cycles ...", file=sys.stderr)
    for i in range(warmup_cycles):
        # build and find once
        build_callable()
        # ensure not to reuse stale state accidentally
        _ = find_callable()
        # free structure so warmup behaves like new runs
        _clear_built_struct()


def _clear_built_struct():
    global _BUILT_STRUCT
    _BUILT_STRUCT = None
    gc.collect()


def time_callable(
    callable_obj: Callable[[], Any],
    per_repetition_stmt_count: int,
    repeat_times: int = 1,
) -> List[float]:
    """
    Time the callable using timeit.Timer.repeat. Returns list of per-call times (normalized).
    We pass the callable directly to Timer, which will call it number times per repeat; we then
    divide by number to get per-call time.
    """
    timer = timeit.Timer(callable_obj)
    # The call to repeat returns a list of total times for each repetition entry.
    raw_times = timer.repeat(repeat=repeat_times, number=per_repetition_stmt_count)
    # Normalize to per-call times
    per_call_times = [t / per_repetition_stmt_count for t in raw_times]
    return per_call_times


def summarystats(times: List[float]) -> Tuple[float, float, float, float]:
    """Return min, mean, max, std (std calculated with statistics.stdev or 0 if single value)."""
    if not times:
        return (0.0, 0.0, 0.0, 0.0)
    mn = min(times)
    mx = max(times)
    mean = statistics.mean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    return (mn, mean, mx, sd)


def profile_one_datastructure(
    name: str,
    sizes: List[int],
    dataset: List[str],
    warmup_cycles: int,
    repetitions: int,
    per_repetition_stmt_count: int,
) -> List[Dict[str, Any]]:
    """
    Profiles the requested data structure for the passed sizes against `dataset`.
    Returns a list of dicts to be appended to the final DataFrame.
    """
    results_rows: List[Dict[str, Any]] = []

    print(f"\n===== Profiling {name} =====", file=sys.stderr)
    for n in sizes:
        print(f"\nDatastructure: {name}", file=sys.stderr)
        print(f"At size {n}", file=sys.stderr)
        sublist = dataset[:n]

        # prepare query
        query = random_query(10)

        # select builder and finder
        if name == "array":
            build_callable = make_array_build_callable(sublist)
            find_callable = make_array_find_callable(query, count=10)
        elif name == "BK_trees":
            build_callable = make_bktree_build_callable(sublist)
            find_callable = make_bktree_find_callable(query, count=10)
        elif name == "VP_trees":
            build_callable = make_vptree_build_callable(sublist)
            find_callable = make_vptree_find_callable(query, count=10)
        elif name == "trie":
            build_callable = make_trie_build_callable(sublist)
            find_callable = make_trie_find_callable(query, count=10)
        else:
            raise ValueError(f"Unknown data structure: {name}")

        # Warmup
        run_warmup(build_callable, find_callable, warmup_cycles)

        # Now do repetitions: for each repetition, time build then time find (so build exists for find)
        build_times_all: List[float] = []
        find_times_all: List[float] = []

        for rep in range(repetitions):
            print(f"  repetition {rep + 1}/{repetitions} ...", file=sys.stderr)
            # Time build
            # Ensure no existing built struct persists from previous repetition
            _clear_built_struct()
            build_times = time_callable(
                build_callable, per_repetition_stmt_count, repeat_times=1
            )
            # build_times is list length 1 (because repeat=1), but keep as list for aggregate
            build_times_all.extend(build_times)

            # After building, the built struct should exist (the builder sets it). Time find.
            find_times = time_callable(
                find_callable, per_repetition_stmt_count, repeat_times=1
            )
            find_times_all.extend(find_times)

            # Clear built struct before next repetition to match spec semantics
            _clear_built_struct()

        # Compute and print summary stats
        b_min, b_mean, b_max, b_sd = summarystats(build_times_all)
        f_min, f_mean, f_max, f_sd = summarystats(find_times_all)

        print(
            f"  Build times (seconds): min={b_min:.6f}, avg={b_mean:.6f}, max={b_max:.6f}, std={b_sd:.6f}",
            file=sys.stderr,
        )
        print(
            f"  Find times  (seconds): min={f_min:.6f}, avg={f_mean:.6f}, max={f_max:.6f}, std={f_sd:.6f}",
            file=sys.stderr,
        )

        # Save rows for DataFrame; store the entire list of per-repetition per-call times
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


def parse_size_list(arg: str) -> List[int]:
    """Parse a comma-separated list of ints for --sizes CLI flag."""
    try:
        parts = arg.split(",")
        sizes = [int(p.strip()) for p in parts if p.strip() != ""]
        return sizes
    except Exception as e:
        raise argparse.ArgumentTypeError(
            "sizes must be comma separated integers"
        ) from e


def main(argv: List[str] | None = None) -> None:
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
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions for statistics (default 10)",
    )
    parser.add_argument(
        "--per-repetition-stmt-count",
        type=int,
        default=1,
        help="Number of statement executions per repetition (default 1)",
    )
    parser.add_argument(
        "--sizes",
        type=parse_size_list,
        default=None,
        help="Comma-separated list of sizes to test (overrides defaults)",
    )
    parser.add_argument(
        "--out-csv",
        default="src/profile_results.csv",
        help="CSV file to write DataFrame results to",
    )
    parser.add_argument(
        "--structures",
        default="array,BK_trees,VP_trees,trie",
        help="Comma-separated list of structures to profile (default all)",
    )
    args = parser.parse_args(argv)

    # load dataset
    dataset = load_dataset(args.dataset)

    sizes = args.sizes if args.sizes is not None else DEFAULT_SIZES
    structures = [s.strip() for s in args.structures.split(",") if s.strip()]

    all_rows: List[Dict[str, Any]] = []

    # Profile in the order: array, BK_trees, VP_trees, trie (if present in requested list)
    order = ["array", "BK_trees", "VP_trees", "trie"]
    for ds in order:
        if ds not in structures:
            continue
        rows = profile_one_datastructure(
            ds,
            sizes,
            dataset,
            warmup_cycles=args.warmup,
            repetitions=args.repetitions,
            per_repetition_stmt_count=args.per_repetition_stmt_count,
        )
        all_rows.extend(rows)

    # Build DataFrame
    df = pd.DataFrame(all_rows, columns=["data_structure", "task", "size", "times"])

    # Save to CSV (times as JSON-ish strings)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved profiling results to {args.out_csv}", file=sys.stderr)

    # Also print a compact summary to stdout
    print("\n=== Compact Summary ===")
    grouped = (
        df.groupby(["data_structure", "task", "size"])["times"].first().reset_index()
    )
    for _, row in grouped.iterrows():
        times = row["times"]
        mn, mean, mx, sd = summarystats(times)
        print(
            f"{row['data_structure']:>10} | {row['task']:>11} | size={row['size']:>9} | min={mn:.6f} avg={mean:.6f} max={mx:.6f} sd={sd:.6f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
