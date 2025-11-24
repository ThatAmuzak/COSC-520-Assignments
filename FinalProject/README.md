# COSC 520 Final Project: String Matching

## Project Setup

- **Installation**:
  This project uses [Poetry](https://python-poetry.org/), which can be installed from [here](https://python-poetry.org/docs/#installation).
  To install all relevant dependencies, please run

  ```sh
  poetry install
  ```

- **Dataset**:
  To generate the dataset, simply run:

  ```sh
  python .\src\generate_dataset.py
  ```

- **Testing**:
  To run all unit tests, simply run

  ```sh
  pytest -s .\tests\test_algorithms.py
  ```
- **Benchmarking**:
  To run comprehensive performance benchmarks across different text and pattern sizes, simply run:
```sh
  python .\benchmark.py
```

  The benchmark script supports several command-line options for customization:
```sh
  python .\benchmark.py --help
```

  Common usage examples:
```sh
  # Run benchmark with default settings
  python .\benchmark.py

  # Test specific algorithms only
  python .\benchmark.py --algos KMP BoyerMoore

  # Test on specific datasets
  python .\benchmark.py --datasets Gutenberg DNA

  # Customize benchmark parameters
  python .\benchmark.py --warmup 5 --formal 15 --repeat 3

  # Save results to custom file and auto-analyze
  python .\benchmark.py --output my_results.csv --analyze

  # Enable debug logging for detailed output
  python .\benchmark.py --debug
```

- **Analysis**:
  To analyze benchmark results and generate visualizations, simply run:
```sh
  python .\analyze_benchmark.py benchmark_results.csv
```

  The analysis script provides various visualization options:
```sh
  # Generate all plots
  python .\analyze_benchmark.py benchmark_results.csv

  # Generate specific plots only
  python .\analyze_benchmark.py benchmark_results.csv --plots time_text memory scaling

  # Save plots without displaying them
  python .\analyze_benchmark.py benchmark_results.csv --no-show
```

  Available plot types:
  - `time_text`: Execution time vs text size
  - `time_pattern`: Execution time vs pattern size
  - `memory`: Memory usage analysis
  - `hit_miss`: Pattern match hit/miss comparison
  - `heatmap`: Algorithm performance heatmaps
  - `scaling`: Time complexity scaling analysis
  - `all`: Generate all visualizations (default)


---

## Overview

The string matching problem is the task of finding all occurrences of a pattern string within a larger text string. It is a fundamental problem in computer science with applications in text search, bioinformatics (sequence alignment), data mining, and more. Efficient algorithms are important because naive approaches can be slow on large texts or many queries.

This project implements and profiles three classic exact string matching algorithms:

- **Knuth–Morris–Pratt (KMP)**: KMP achieves linear-time matching by precomputing a longest-prefix-suffix ("failure") table for the pattern. During search, the failure table allows the algorithm to skip re-examining characters when a mismatch occurs, guaranteeing O(n + m) time, where n is the text length and m is the pattern length.

- **Rabin–Karp**: Rabin–Karp uses a rolling hash to quickly compare the pattern against substring of the text. Hash comparisons are O(1) on average, so Rabin–Karp performs well for single or multiple pattern searches; however, its worst-case can degrade depending on hash collisions. It is especially convenient when searching for many patterns simultaneously.

- **Boyer–Moore**: Boyer–Moore uses two main heuristics (bad-character and good-suffix) to skip ahead in the text, often achieving sublinear performance on typical inputs because it aligns the pattern from right to left and can skip large sections on mismatches. Its average performance is very good in practice.

All three algorithms are implemented in `src\algorithms.py` and can be profiled using the provided profiling script to compare their runtime characteristics on generated datasets.
