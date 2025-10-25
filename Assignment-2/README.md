# COSC 520 Assignment 2: Advanced Data Structures

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

**Fuzzy Matching Overview**
Fuzzy matching is the task of finding strings in a dataset that closely resemble a query, even when they differ due to typos or variations. It relies on _edit distance_ metrics—particularly the **Levenshtein distance**—to quantify similarity.

**Data Structures Implemented**
To efficiently support fuzzy search, three index structures were implemented and compared, along with a baseline array implementation:

- **BK-Tree** – organizes words by pairwise distances for metric-based pruning.
- **VP-Tree** – partitions words around vantage points for balanced metric searches.
- **Trie (Prefix Tree)** – stores words by prefixes and integrates dynamic programming to prune searches based on edit distance thresholds.
