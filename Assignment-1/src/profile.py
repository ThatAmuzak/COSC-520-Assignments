import random
import statistics
import timeit

import pandas as pd
from search_algorithms import (
    BloomFilter,
    CuckooFilter,
    binary_search,
    generate_hash_table,
    hash_search,
    linear_search,
)
from tqdm import tqdm
from typeguard import typechecked

# Global Configurations
WARMUP_COUNT = 3  # number of warmup iterations before measuring
REPEAT_COUNT = 10  # number of repetitions
PER_REPETITION_STMT_COUNT = 5  # number of function evals within a repetition
RANDOM_SEED = 42  # random seed

results_df = pd.DataFrame(
    {
        "search_func": pd.Series(dtype="string"),
        "size": pd.Series(dtype="int"),
        "hit": pd.Series(dtype="string"),
        "times": pd.Series(dtype="float"),
    }
)


@typechecked
def generate_unique_usernames(n: int, prefix: str = "user") -> list[str]:
    """
    generates unique logins dataset
    not really representative of actual usernames
    but good enough for testing purposes
    """
    return [f"{prefix}{i}" for i in tqdm(range(n), desc="Generating usernames")]


@typechecked
def profile_linear_search(arr: list[str], sizes: list[int], element_present: bool):
    # fixing randomization
    random.seed(RANDOM_SEED)

    # profiling for different array sizes
    for n in sizes:
        sub_arr = arr[:n]

        # selection of target element
        if element_present:
            target = sub_arr[random.randint(0, n - 1)]
        else:
            target = "not_present"

        # wrapper statement for timeit
        def stmt():
            return linear_search(sub_arr, target)

        # warmup, to stabilize JIT, caching, CPU freq scaling, branch predictions
        # inspired from Unity's unit testing framework, although not entire sure
        # if this works in python, or how effective it is
        for _ in range(WARMUP_COUNT):
            stmt()

        # measure
        timer = timeit.Timer(stmt)
        times = timer.repeat(repeat=REPEAT_COUNT, number=PER_REPETITION_STMT_COUNT)

        print(
            f"n={n:>10} | "
            f"target={'HIT' if element_present else 'MISS'} | "
            f"min={min(times):.6f}s | "
            f"avg={sum(times) / len(times):.6f}s | "
            f"max={max(times):.6f}s | "
            f"stddev={statistics.stdev(times):.6f}s"
        )

        for t in times:
            results_df.loc[len(results_df)] = {
                "search_func": "Linear",
                "size": n,
                "hit": "HIT" if element_present else "MISS",
                "times": t,
            }


@typechecked
def profile_binary_search(arr: list[str], sizes: list[int], element_present: bool):
    random.seed(RANDOM_SEED)
    for n in sizes:
        sub_arr = arr[:n]

        if element_present:
            target = sub_arr[random.randint(0, n - 1)]
        else:
            target = "not_present"

        # wrapper statement for timeit
        def stmt():
            return binary_search(sub_arr, target)

        # warmup
        for _ in range(WARMUP_COUNT):
            stmt()

        # measure
        timer = timeit.Timer(stmt)
        times = timer.repeat(repeat=REPEAT_COUNT, number=PER_REPETITION_STMT_COUNT)

        print(
            f"n={n:>10} | "
            f"target={'HIT' if element_present else 'MISS'} | "
            f"min={min(times):.6f}s | "
            f"avg={sum(times) / len(times):.6f}s | "
            f"max={max(times):.6f}s | "
            f"stddev={statistics.stdev(times):.6f}s"
        )

        for t in times:
            results_df.loc[len(results_df)] = {
                "search_func": "Binary",
                "size": n,
                "hit": "HIT" if element_present else "MISS",
                "times": t,
            }


@typechecked
def profile_hashmap_search(arr: list[str], sizes: list[int], element_present: bool):
    random.seed(RANDOM_SEED)
    print("Setting up Hashmap")
    hash_table = generate_hash_table(arr)
    print("Hashmap setup complete")
    for n in sizes:
        sub_arr = arr[:n]

        if element_present:
            target = sub_arr[random.randint(0, n - 1)]
        else:
            target = "not_present"

        # wrapper statement for timeit
        def stmt():
            return hash_search(hash_table, target)

        # warmup
        for _ in range(WARMUP_COUNT):
            stmt()

        # measure
        timer = timeit.Timer(stmt)
        times = timer.repeat(repeat=REPEAT_COUNT, number=PER_REPETITION_STMT_COUNT)

        print(
            f"n={n:>10} | "
            f"target={'HIT' if element_present else 'MISS'} | "
            f"min={min(times):.6f}s | "
            f"avg={sum(times) / len(times):.6f}s | "
            f"max={max(times):.6f}s | "
            f"stddev={statistics.stdev(times):.6f}s"
        )

        for t in times:
            results_df.loc[len(results_df)] = {
                "search_func": "Hashmap",
                "size": n,
                "hit": "HIT" if element_present else "MISS",
                "times": t,
            }


@typechecked
def profile_bloom_search(arr: list[str], sizes: list[int], element_present: bool):
    random.seed(RANDOM_SEED)
    print("Setting up bitarray")
    filter = BloomFilter()
    for login in arr:
        filter.add(login)
    print("bitarray setup complete")
    for n in sizes:
        sub_arr = arr[:n]

        if element_present:
            target = sub_arr[random.randint(0, n - 1)]
        else:
            target = "not_present"

        # wrapper statement for timeit
        def stmt():
            return filter.get(target)

        # warmup
        for _ in range(WARMUP_COUNT):
            stmt()

        # measure
        timer = timeit.Timer(stmt)
        times = timer.repeat(repeat=REPEAT_COUNT, number=PER_REPETITION_STMT_COUNT)

        print(
            f"n={n:>10} | "
            f"target={'HIT' if element_present else 'MISS'} | "
            f"min={min(times):.6f}s | "
            f"avg={sum(times) / len(times):.6f}s | "
            f"max={max(times):.6f}s | "
            f"stddev={statistics.stdev(times):.6f}s"
        )

        for t in times:
            results_df.loc[len(results_df)] = {
                "search_func": "Bloom",
                "size": n,
                "hit": "HIT" if element_present else "MISS",
                "times": t,
            }


@typechecked
def profile_cuckoo_search(arr: list[str], sizes: list[int], element_present: bool):
    random.seed(RANDOM_SEED)
    print("Setting up bitarray")
    filter = CuckooFilter()
    for login in arr:
        filter.add(login)
    print("bitarray setup complete")
    for n in sizes:
        sub_arr = arr[:n]

        if element_present:
            target = sub_arr[random.randint(0, n - 1)]
        else:
            target = "not_present"

        # wrapper statement for timeit
        def stmt():
            return filter.get(target)

        # warmup
        for _ in range(WARMUP_COUNT):
            stmt()

        # measure
        timer = timeit.Timer(stmt)
        times = timer.repeat(repeat=REPEAT_COUNT, number=PER_REPETITION_STMT_COUNT)

        print(
            f"n={n:>10} | "
            f"target={'HIT' if element_present else 'MISS'} | "
            f"min={min(times):.6f}s | "
            f"avg={sum(times) / len(times):.6f}s | "
            f"max={max(times):.6f}s | "
            f"stddev={statistics.stdev(times):.6f}s"
        )

        for t in times:
            results_df.loc[len(results_df)] = {
                "search_func": "Cuckoo",
                "size": n,
                "hit": "HIT" if element_present else "MISS",
                "times": t,
            }


if __name__ == "__main__":
    sizes = [
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
        5_000_000,
        10_000_000,
    ]

    # generate once at max size
    max_size = max(sizes)
    arr = generate_unique_usernames(max_size)

    print("=" * 60)
    print("📊 Profiling: LINEAR SEARCH".center(60))
    print("Warmup: 3 cycles; Measuring on: 10 x 5 cycles;".center(60))
    print("=" * 60)
    print("When item is present".center(60))
    profile_linear_search(arr, sizes, True)
    print("=" * 60)
    print("When item is NOT present".center(60))
    profile_linear_search(arr, sizes, False)
    print("=" * 60)

    print("=" * 60)
    print("📊 Profiling: BINARY SEARCH".center(60))
    print("Warmup: 3 cycles; Measuring on: 10 x 5 cycles;".center(60))
    print("=" * 60)
    print("When item is present".center(60))
    profile_binary_search(arr, sizes, True)
    print("=" * 60)
    print("When item is NOT present".center(60))
    profile_binary_search(arr, sizes, False)
    print("=" * 60)

    print("=" * 60)
    print("📊 Profiling: HASHMAP SEARCH".center(60))
    print("Warmup: 3 cycles; Measuring on: 10 x 5 cycles;".center(60))
    print("=" * 60)
    print("When item is present".center(60))
    profile_hashmap_search(arr, sizes, True)
    print("=" * 60)
    print("When item is NOT present".center(60))
    profile_hashmap_search(arr, sizes, False)
    print("=" * 60)

    print("=" * 60)
    print("📊 Profiling: BLOOMSEARCH".center(60))
    print("Warmup: 3 cycles; Measuring on: 10 x 5 cycles;".center(60))
    print("=" * 60)
    print("When item is present".center(60))
    profile_bloom_search(arr, sizes, True)
    print("=" * 60)
    print("When item is NOT present".center(60))
    profile_bloom_search(arr, sizes, False)
    print("=" * 60)

    print("=" * 60)
    print("📊 Profiling: CUCKOOSEARCH".center(60))
    print("Warmup: 3 cycles; Measuring on: 10 x 5 cycles;".center(60))
    print("=" * 60)
    print("When item is present".center(60))
    profile_cuckoo_search(arr, sizes, True)
    print("=" * 60)
    print("When item is NOT present".center(60))
    profile_cuckoo_search(arr, sizes, False)
    print("=" * 60)

    results_df.to_csv("src/results.csv", index=False)
    print("Results saved to results.csv")
