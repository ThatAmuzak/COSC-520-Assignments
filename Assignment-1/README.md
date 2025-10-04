# COSC 520 Assignment 1: Login Checker Algorithm Comparison

## Project Setup

- **Installation**:
  This project uses [Poetry](https://python-poetry.org/), which can be installed from [here](https://python-poetry.org/docs/#installation).
  To install all relevant dependencies, please run

  ```sh
  poetry install
  ```

- **Profiling**:
  To run the profiling script, simply run:

  ```sh
  python .\src\profile.py
  ```

- **Testing**:
  To run all unit tests, simply run
  ```sh
  pytest .\tests\test_search_algorithms.py
  ```

---

## Summary

The Login Checker problem involves verifying whether a given username exists in a dataset of registered users.
Efficient solutions are crucial for handling large-scale user authentication systems.

Approaches tested:

- **Linear Search**: Suitable for small datasets; not efficient for large-scale applications.

- **Binary Search**: Efficient for sorted data; not applicable for unsorted datasets.

- **HashMap**: Offers fast lookups and supports deletions; requires hashable keys.

- **Bloom Filter**: Ideal for large datasets with read-heavy operations; allows for false positives but no false negatives.

- **Cuckoo Filter**: Provides better space efficiency and supports deletions; suitable for applications requiring both space efficiency and deletion capabilities.
