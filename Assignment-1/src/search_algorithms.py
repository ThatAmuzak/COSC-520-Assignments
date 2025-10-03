from typeguard import typechecked

"""
Search Algorithms Module
------------------------

This module provides simple implementations of common search algorithms

- Linear Search: Sequentially scans through a list.
- Binary Search: Efficiently searches within a sorted list.
- Hash Search: Uses a precomputed hash table for constant-time lookups.
- Bloom Filter:
- Cuckoo Filter:

Each function is annotated with type hints and documented with details
about time and space complexity.

Functions
---------
linear_search(arr: list[str], target: str) -> int
    Performs a linear search for the target in the list.
binary_search(arr: list[str], target: str) -> int
    Performs a binary search on a sorted list for the target.
generate_hash_table(arr: list[str]) -> dict[str, int]
    Builds a hash table mapping each element to its index.
hash_search(table: dict[str, int], target: str) -> int
    Performs a constant-time search for the target in the hash table.
"""


@typechecked
def linear_search(arr: list[str], target: str) -> int:
    """
    Basic linear search.
    Assumes nothing about provided array.
    Time complexity: O(n)
    Space complexity: O(1)

    Parameters
    ----------
    arr : list
        The list to search through.
    target : string
        The element to search for.

    Returns
    -------
    int
        The index of the target if found, otherwise -1.
    """
    for index, value in enumerate(arr):
        if value == target:
            return index
    return -1


@typechecked
def binary_search(arr: list[str], target: str) -> int:
    """
    Basic implementation of binary search.
    Assumes provided array is sorted.
    Time Complexity: O(log n)
    Space Complexity: O(1)

    Parameters
    ----------
    arr : list
        The sorted list to search through.
    target : string
        The element to search for.

    Returns
    -------
    int
        The index of the target if found, otherwise -1.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


@typechecked
def generate_hash_table(arr: list[str]) -> dict[str, int]:
    """
    Generates hash table using dicts for subsequent searches.

    Parameters
    ----------
    arr : list of strings
        The list of strings to store in the hash table.

    Returns
    -------
    dict
        A dictionary mapping each string to its index in the list.

    See Also
    --------
    :func:`hash_search`.


    """
    table = {string: idx for idx, string in enumerate(arr)}
    return table


# @typechecked
def hash_search(table: dict[str, int], target: str) -> int:
    """
    Searches for a string in the hash table and returns its index.
    Assumes hash table has been generated beforehand.
    Time Complexity: O(1)
    Space Complexity: O(n)

    Parameters
    ----------
    table : dict
        The hash table mapping strings to their indices.
    target : str
        The string to search for.

    Returns
    -------
    int
        The index of the string if found, otherwise -1.

    See Also
    --------
    :func:`generate_hash_table`.

    """
    return table.get(target, -1)


hash_search(1, "aa")
