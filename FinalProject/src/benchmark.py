import os
import gc
import time
import random
import tracemalloc
from typing import Callable, Dict, List, Tuple
import argparse
import pandas as pd
from tqdm.auto import tqdm

from algorithms import naive_search, kmp_search, rabin_karp, boyer_moore
from analyze_benchmark import BenchmarkAnalyzer
# -----------------------------
# Debug switch
# -----------------------------

DEBUG = False


def debug_log(msg: str) -> None:
    """Simple debug logger controlled by DEBUG flag."""
    if DEBUG:
        print(msg, flush=True)


# -----------------------------
# Config
# -----------------------------

DATA_DIR = "data"
GUTENBERG_PATH = os.path.join(DATA_DIR, "gutenberg.txt")
ADVERSARIAL_PATH = os.path.join(DATA_DIR, "adversarial.txt")
DNA_PATH = os.path.join(DATA_DIR, "Panda_DNA.fa")

# Text sizes: from 100 to 10 million (11 sizes)
TEXT_SIZES = [
    100,
    500,
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
]

def pattern_size_scaling_with_cap(text_size: int, max_pattern: int = 10_000) -> int:
    """
    Pattern size scales with text size but has an upper bound.
    Corrected to be monotonically increasing.
    
    Strategy:
    - < 10K: Linear (10%) -> Ends at 1,000
    - 10K - 1M: Sqrt scaled -> Starts at 1,000, ends at 10,000
    - > 1M: Power 0.4 scaled -> Starts at 10,000
    """
    import math
    
    if text_size < 10_000:
        # Small texts: 10% of text size
        # Range: 10 -> 999
        pattern_size = max(10, text_size // 10)
        
    elif text_size < 1_000_000:
        # Medium texts: square root scaling
        # We multiply by 10 to ensure it connects with the previous segment
        # At 10,000: sqrt(10000)*10 = 100*10 = 1,000 (Matches previous segment)
        # At 1M: sqrt(1M)*10 = 1000*10 = 10,000
        pattern_size = int(math.sqrt(text_size) * 10)
        
    else:
        # Large texts: slow growth (power 0.4)
        # We multiply by 40 to ensure it connects with the previous segment
        # At 1M: (10^6)^0.4 * 40 = 251.18 * 40 ≈ 10,047 (Matches approx 10,000)
        pattern_size = int((text_size ** 0.4) * 40)
    
    # Apply cap and ensure it's strictly less than text
    # strict (< text_size) is crucial for search algos
    pattern_size = min(pattern_size, max_pattern, text_size - 1)
    
    # Final safety for very small texts (e.g. size 100 -> pattern 10)
    return max(1, pattern_size)


def pattern_size_fixed(text_size: int, fixed_size: int = 100) -> int:
    return min(fixed_size, text_size - 1)

def get_pattern_size(text_size: int) -> int:
    return pattern_size_scaling_with_cap(text_size, max_pattern=10_000)


# Benchmark parameters (can be overridden by command line)
N_WARMUP = 3      # 3 warmup runs
N_FORMAL = 10     # 10 formal runs
N_REPEAT = 3      # 3 repetitions per formal run (take min)


# -----------------------------
# Adaptive parameter selection
# -----------------------------

def get_adaptive_params(algo_name: str, text_size: int) -> Tuple[int, int, int]:
    """
    Adaptively select warmup, formal, and repeat parameters based on 
    algorithm type and text size to reduce benchmark time.
    
    Strategy:
    - Naive algorithm: Reduce parameters for text > 10K (it's very slow)
    - Large texts (>100K): Reduce parameters for all algorithms
    - Small texts (≤10K): Use full parameters for statistical significance
    
    Returns:
        (n_warmup, n_formal, n_repeat)
    """
    # For Naive algorithm: it's O(n*m) so very slow on large texts
    if algo_name == "Naive":
        if text_size <= 1_000:
            return (3, 10, 3)      # Small: full testing
        elif text_size <= 10_000:
            return (2, 5, 2)       # Medium: reduce
        elif text_size <= 100_000:
            return (1, 3, 2)       # Large: minimal testing
        else:
            return (0, 1, 1)       # Very large: barely test (or skip entirely)
    
    # For efficient algorithms (KMP, RabinKarp, BoyerMoore)
    else:
        if text_size <= 10_000:
            return (3, 10, 3)      # Small: full testing
        elif text_size <= 100_000:
            return (2, 5, 2)       # Medium: moderate reduction
        elif text_size <= 1_000_000:
            return (1, 3, 2)       # Large: significant reduction
        else:
            return (1, 2, 1)       # Very large: minimal testing


# -----------------------------
# Helpers: load datasets
# -----------------------------

def load_text_file(path: str, binary: bool = False) -> str:
    """Load text file with optional binary mode."""
    if binary:
        with open(path, "rb") as f:
            data = f.read()
        return data.decode("ascii", errors="ignore")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def load_datasets() -> Dict[str, str]:
    datasets = {}
    datasets["Gutenberg"] = load_text_file(GUTENBERG_PATH, binary=False)
    datasets["Adversarial"] = load_text_file(ADVERSARIAL_PATH, binary=True)
    datasets["DNA"] = load_text_file(DNA_PATH, binary=False)

    for name, text in datasets.items():
        debug_log(f"[DEBUG] Loaded dataset '{name}' with length={len(text)}")

    return datasets


def get_text_suffix(full_text: str, target_size: int) -> str:
    if len(full_text) >= target_size:
        return full_text[-target_size:]
    else:
        if 'a' in full_text and 'b' in full_text:
            return 'a' * (target_size - 1) + 'b'
        else:
            print(f"[WARN] Repeating {len(full_text)} text to {target_size}")
            repeat_times = (target_size // len(full_text)) + 1
            repeated = full_text * repeat_times
            return repeated[:target_size]


# -----------------------------
# Helpers: hit / miss pattern generation
# -----------------------------

def ensure_size_not_exceed_text(size: int, text: str) -> int:
    """Ensure pattern size is <= len(text) - 1."""
    return min(size, max(1, len(text) - 1))


def make_gutenberg_hit(text: str, size: int, rng: random.Random) -> str:
    size = ensure_size_not_exceed_text(size, text)
    max_start = len(text) - size
    start = rng.randint(0, max_start)
    return text[start:start + size]


def find_missing_char(text: str) -> str:
    """
    Find a character that does not appear in text.
    Tries multiple categories of characters to maximize chance of finding one.
    """
    # Build a comprehensive list of candidate characters
    candidates = (
        "0123456789"
        + "!@#$%^&*()_+-={}[]|\\;:'\",.<>?/~`"
        + "§±µ¶·¸º»¼½¾¿"
        + "¢£¤¥€"
        + "×÷°²³¹"
        + "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß"
        + "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
        + "αβγδεζηθικλμνξοπρστυφχψω"
        + "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        + "│┤┐└┴┬├─┼"
        + "←↑→↓↔↕"
        + "•◦‣⁃∙⋅●○◘◙"
    )
    
    existing = set(text)
    
    # Try to find a character not in text
    for c in candidates:
        if c not in existing:
            return c
    
    # Fallback 1: Try some rare Unicode characters
    rare_unicode = [
        "\u2022",  # Bullet
        "\u25A0",  # Black square
        "\u25CF",  # Black circle
        "\u2665",  # Heart
        "\u2666",  # Diamond
        "\u2663",  # Club
        "\u2660",  # Spade
        "\u00A0",  # Non-breaking space
        "\u200B",  # Zero-width space
        "\u2020",  # Dagger
        "\u2021",  # Double dagger
        "\uFFFD",  # Replacement character
    ]
    
    for c in rare_unicode:
        if c not in existing:
            return c
    
    # Fallback 2: Use a very high Unicode character (extremely unlikely to be in text)
    return "\uFFFF"


def make_gutenberg_miss(text: str, size: int, rng: random.Random) -> str:
    """
    Generate a miss pattern for Gutenberg text.
    
    Strategy: Replace 10% of characters with a character not in the text.
    This makes it statistically impossible for the pattern to appear in text,
    avoiding expensive O(n*m) verification.
    """
    size = ensure_size_not_exceed_text(size, text)
    base = make_gutenberg_hit(text, size, rng)
    missing_char = find_missing_char(text)
    s_list = list(base)
    
    # Replace 10% of characters at random positions (no duplicates)
    n_replace = max(1, size // 10)
    positions = rng.sample(range(size), n_replace)
    
    for pos in positions:
        s_list[pos] = missing_char
    
    pattern = "".join(s_list)
    return pattern


def make_adversarial_hit(text: str, size: int, rng: random.Random) -> str:
    """
    Generate a hit pattern for adversarial text.
    Since we take text suffix, the pattern 'aaaa...ab' should be at the end.
    """
    size = ensure_size_not_exceed_text(size, text)
    # Take the last 'size' characters - should contain the 'b' at the end
    return text[-size:]


def make_adversarial_miss(text: str, size: int, rng: random.Random) -> str:
    """
    Generate a miss pattern for adversarial text.
    Creates 'aaaa...ac' which has many partial matches but never fully matches.
    """
    size = ensure_size_not_exceed_text(size, text)
    if size <= 1:
        return "c"
    # Pattern: 'aaaa...aaac' - many partial matches, but 'c' never appears in text
    return "a" * (size - 1) + "c"


def make_dna_hit(text: str, size: int, rng: random.Random) -> str:
    """Generate a hit pattern from DNA text."""
    size = ensure_size_not_exceed_text(size, text)
    max_start = len(text) - size
    start = rng.randint(0, max_start)
    return text[start:start + size]


def pick_missing_dna_char(text: str) -> str:
    """
    Find a character not present in DNA sequence.
    DNA typically uses: A, T, C, G, and sometimes N for unknown bases.
    """
    alphabet = set(text)
    
    # Primary candidates - characters commonly NOT in DNA sequences
    candidates = [
        "N", "X", "Y", "R", "W", "S", "K", "M", "D", "V", "H", "B",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "#", "@", "$", "%", "^", "&", "*", "-", "_", "+", "=",
        "x", "y", "z", "q", "j",
        "~", "`", "!", "?", "|", "\\", "/",
    ]
    
    for c in candidates:
        if c not in alphabet:
            return c
    return "N"


def make_dna_miss(text: str, size: int, rng: random.Random) -> str:
    """
    Generate a miss pattern for DNA text.
    
    Strategy: Replace 10% of characters with a character not in the DNA alphabet.
    This makes collision virtually impossible without expensive verification.
    """
    size = ensure_size_not_exceed_text(size, text)
    base = make_dna_hit(text, size, rng)
    missing_char = pick_missing_dna_char(text)
    s_list = list(base)
    
    # Replace 10% of characters at unique random positions
    n_replace = max(1, size // 10)
    positions = rng.sample(range(size), n_replace)  # No duplicates
    
    for pos in positions:
        s_list[pos] = missing_char
    
    pattern = "".join(s_list)
    return pattern


def get_pattern_generators():
    """Get pattern generator functions for each dataset."""
    return {
        "Gutenberg": {
            True: make_gutenberg_hit,
            False: make_gutenberg_miss,
        },
        "Adversarial": {
            True: make_adversarial_hit,
            False: make_adversarial_miss,
        },
        "DNA": {
            True: make_dna_hit,
            False: make_dna_miss,
        },
    }


# -----------------------------
# Measurement helpers
# -----------------------------

def measure_time_and_memory(
    func: Callable[[str, str], List[int]],
    pattern: str,
    text: str,
) -> Tuple[float, int, List[int]]:
    # Simple time-only measurement (fast path)
    t0 = time.perf_counter()
    result = func(pattern, text)
    elapsed = time.perf_counter() - t0
    
    # Return with dummy memory value (will be measured separately when needed)
    return elapsed, 0, result


def measure_time_and_memory_full(
    func: Callable[[str, str], List[int]],
    pattern: str,
    text: str,
) -> Tuple[float, int, List[int]]:
    # Start memory tracing
    tracemalloc.start()
    
    # Measure time
    t0 = time.perf_counter()
    result = func(pattern, text)
    elapsed = time.perf_counter() - t0
    
    # Get peak memory usage during execution
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return elapsed, int(peak), result


# -----------------------------
# Main benchmark
# -----------------------------

def run_benchmark(
    random_seed: int = 42,
    selected_algos: List[str] | None = None,
    selected_datasets: List[str] | None = None,
) -> pd.DataFrame:
    """
    Run the complete benchmark.
    
    New logic:
    - Iterate over TEXT_SIZES (small to large)
    - For each text size, calculate pattern size dynamically
    - Generate patterns on-the-fly for each test
    """
    rng = random.Random(random_seed)

    # 1. Load full datasets
    full_datasets = load_datasets()
    pattern_generators = get_pattern_generators()

    # 2. Define algorithms
    algorithms: Dict[str, Callable[[str, str], List[int]]] = {
        "Naive": naive_search,
        "KMP": kmp_search,
        "RabinKarp": rabin_karp,
        "BoyerMoore": boyer_moore,
    }

    # 3. Filter by selected datasets
    if selected_datasets is not None:
        full_datasets = {
            name: text
            for name, text in full_datasets.items()
            if name in selected_datasets
        }
        pg_full = pattern_generators
        pattern_generators = {name: pg_full[name] for name in full_datasets.keys()}

    # 4. Filter by selected algorithms
    if selected_algos is not None:
        algorithms = {
            name: fn
            for name, fn in algorithms.items()
            if name in selected_algos
        }

    print(f"[INFO] Algorithms: {list(algorithms.keys())}")
    print(f"[INFO] Datasets: {list(full_datasets.keys())}")
    print(f"[INFO] Text sizes: {TEXT_SIZES}")
    print()

    records = []

    # Calculate total steps for progress bar (considering adaptive parameters)
    algo_items = list(algorithms.items())
    dataset_items = list(full_datasets.items())
    hit_flags = [True, False]
    
    # Calculate actual total steps with adaptive parameters
    total_steps = 0
    for algo_name, _ in algo_items:
        for text_size in TEXT_SIZES:
            _, n_formal, _ = get_adaptive_params(algo_name, text_size)
            total_steps += len(dataset_items) * len(hit_flags) * n_formal
    
    print(f"[INFO] Using adaptive parameters to optimize benchmark speed")
    print(f"[INFO] Estimated total formal runs: {total_steps}")
    print(f"[INFO] (Original fixed config would be: {len(algo_items) * len(dataset_items) * len(hit_flags) * len(TEXT_SIZES) * N_FORMAL})")

    progress = tqdm(total=total_steps, desc="Benchmark", dynamic_ncols=True)

    # Main benchmark loop
    for algo_name, algo_func in algo_items:
        debug_log(f"\n=== Algorithm: {algo_name} ===")

        for dataset_name, full_text in dataset_items:
            debug_log(f"  Dataset: {dataset_name} (full length={len(full_text)})")
            gen_map = pattern_generators[dataset_name]

            for hit_flag in hit_flags:
                hit_label = "hit" if hit_flag else "miss"
                debug_log(f"    Case: {hit_label}")
                gen_func = gen_map[hit_flag]

                for text_size in TEXT_SIZES:
                    # Get text of the desired size (from the end)
                    text = get_text_suffix(full_text, text_size)
                    
                    # Calculate pattern size for this text size
                    pattern_size = get_pattern_size(text_size)
                    
                    # Get adaptive parameters for this algorithm and text size
                    n_warmup, n_formal, n_repeat = get_adaptive_params(algo_name, text_size)
                    
                    debug_log(
                        f"[DEBUG] text_size={text_size}, pattern_size={pattern_size}, "
                        f"ratio={pattern_size/text_size:.4f}, "
                        f"params=(warmup={n_warmup}, formal={n_formal}, repeat={n_repeat})"
                    )

                    # ==== OPTIMIZATION: Single GC before warmup ====
                    # Only run GC once per configuration instead of before each measurement
                    gc.collect()
                    
                    # Warmup: generate a pattern and run n_warmup times
                    warmup_pattern = gen_func(text, pattern_size, rng)
                    
                    debug_log(
                        f"[DEBUG] Warmup: algo={algo_name}, dataset={dataset_name}, "
                        f"hit={hit_label}, text_size={text_size}, pattern_size={pattern_size}"
                    )
                    
                    for warm_idx in range(n_warmup):
                        debug_log(f"[DEBUG]   Warmup iter {warm_idx + 1}/{n_warmup}")
                        _ = algo_func(warmup_pattern, text)

                    # Release warmup pattern (no GC call here, already clean)
                    del warmup_pattern

                    # Formal runs: generate fresh pattern for each run
                    for run_idx in range(n_formal):
                        pattern = gen_func(text, pattern_size, rng)
                        
                        debug_log(
                            f"[DEBUG] Formal run {run_idx + 1}/{n_formal}: "
                            f"algo={algo_name}, dataset={dataset_name}, "
                            f"hit={hit_label}, text_size={text_size}, pattern_size={len(pattern)}"
                        )

                        # ==== OPTIMIZATION: Smart memory measurement ====
                        # Only measure memory on first repetition, then use fast time-only measurement
                        times: List[float] = []
                        mems: List[int] = []

                        for rep_idx in range(n_repeat):
                            if rep_idx == 0:
                                # First repetition: measure both time and memory
                                t, m, result = measure_time_and_memory_full(algo_func, pattern, text)
                            else:
                                # Subsequent repetitions: only measure time (much faster)
                                t, m, result = measure_time_and_memory(algo_func, pattern, text)
                                # Use memory from first measurement
                                m = mems[0] if mems else 0
                            
                            debug_log(
                                f"[DEBUG]   Rep {rep_idx + 1}/{n_repeat}: "
                                f"time={t:.6f}s, mem={m} bytes, matches={len(result)}"
                            )
                            times.append(t)
                            mems.append(m)

                        best_time = min(times)
                        # Use memory from first measurement (most accurate)
                        best_mem = mems[0] if mems else 0

                        debug_log(
                            f"[DEBUG]   Best: time={best_time:.6f}s, mem={best_mem} bytes"
                        )

                        # Record result
                        records.append({
                            "Algo": algo_name,
                            "Dataset": dataset_name,
                            "TextSize": text_size,
                            "PatternSize": pattern_size,
                            "Hit": hit_flag,
                            "Run": run_idx + 1,
                            "Time": best_time,
                            "Memory": best_mem,
                        })

                        # Update progress bar
                        progress.update(1)
                        progress.set_postfix_str(
                            f"{algo_name} | {dataset_name} | "
                            f"Text={text_size} | Pattern={pattern_size} | "
                            f"{hit_label} | Run={run_idx+1}/{n_formal}"
                        )

    progress.close()
    print("\n[INFO] Benchmark completed!")

    df = pd.DataFrame.from_records(records)
    return df


# -----------------------------
# Main entry point
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="String Matching Benchmark with Dynamic Text/Pattern Sizing"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup cycles (default: 3)"
    )

    parser.add_argument(
        "--formal",
        type=int,
        default=10,
        help="Number of formal test runs (default: 10)"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repetitions per test (default: 3)"
    )

    parser.add_argument(
        "--algos",
        nargs="+",
        choices=["Naive", "KMP", "RabinKarp", "BoyerMoore"],
        help="Algorithms to benchmark (default: all)"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["Gutenberg", "Adversarial", "DNA"],
        help="Datasets to benchmark (default: all)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file path (default: benchmark_results.csv)"
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Automatically generate analysis plots after benchmark completes"
    )

    args = parser.parse_args()
    
    # Update settings
    DEBUG = args.debug
    N_WARMUP = args.warmup
    N_FORMAL = args.formal
    N_REPEAT = args.repeat

    print("=" * 60)
    print("STRING MATCHING BENCHMARK")
    print("=" * 60)
    print(f"Debug logging:    {DEBUG}")
    print(f"Warmup runs:      {N_WARMUP}")
    print(f"Formal runs:      {N_FORMAL}")
    print(f"Repetitions:      {N_REPEAT}")
    print(f"Selected algos:   {args.algos or 'ALL'}")
    print(f"Selected datasets: {args.datasets or 'ALL'}")
    print(f"Output file:      {args.output}")
    print("=" * 60)
    print()

    # Run benchmark
    df = run_benchmark(
        random_seed=42,
        selected_algos=args.algos,
        selected_datasets=args.datasets,
    )

    # Display summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df.head(10))
    print(f"\nTotal records: {len(df)}")
    print(f"\nText sizes tested: {sorted(df['TextSize'].unique())}")
    print(f"Pattern sizes range: {df['PatternSize'].min()} - {df['PatternSize'].max()}")
    
    # Save
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to: {args.output}")
    print("=" * 60)
    
    # Run analysis if requested
    if args.analyze:
        print("\n" + "=" * 60)
        print("RUNNING ANALYSIS...")
        print("=" * 60)
        
            
        analyzer = BenchmarkAnalyzer(args.output)
        analyzer.generate_summary_statistics(save=True)
        analyzer.generate_all_plots()
        
        print("\nAnalysis complete! Check the 'plots/' directory for visualizations.")