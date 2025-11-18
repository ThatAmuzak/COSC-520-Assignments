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

- **Profiling**:
  To run the profiling script, simply run:

  ```sh
  python .\src\profile.py
  ```

- **Testing**:
  To run all unit tests, simply run

  ```sh
  pytest .\tests\test_fuzzy_search.py
  ```

---

## Overview

The string matching problem is the task of finding all occurrences of a pattern string within a larger text string. It is a fundamental problem in computer science with applications in text search, bioinformatics (sequence alignment), data mining, and more. Efficient algorithms are important because naive approaches can be slow on large texts or many queries.

This project implements and profiles three classic exact string matching algorithms:

- **Knuth–Morris–Pratt (KMP)**: KMP achieves linear-time matching by precomputing a longest-prefix-suffix ("failure") table for the pattern. During search, the failure table allows the algorithm to skip re-examining characters when a mismatch occurs, guaranteeing O(n + m) time, where n is the text length and m is the pattern length.

- **Rabin–Karp**: Rabin–Karp uses a rolling hash to quickly compare the pattern against substring of the text. Hash comparisons are O(1) on average, so Rabin–Karp performs well for single or multiple pattern searches; however, its worst-case can degrade depending on hash collisions. It is especially convenient when searching for many patterns simultaneously.

- **Boyer–Moore**: Boyer–Moore uses two main heuristics (bad-character and good-suffix) to skip ahead in the text, often achieving sublinear performance on typical inputs because it aligns the pattern from right to left and can skip large sections on mismatches. Its average performance is very good in practice.

All three algorithms are implemented in `src\algorithms.py` and can be profiled using the provided profiling script to compare their runtime characteristics on generated datasets.
