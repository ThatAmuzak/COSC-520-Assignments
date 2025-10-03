from typeguard import typechecked

"""
Search Algorithms Module
for Login Checker Problem
------------------------

This module provides simple implementations of common search algorithms
All implementations return booleans, indicating existence of membership
within a set.

- Linear Search: Sequentially scans through a list.
- Binary Search: Efficiently searches within a sorted list.
- Hash Search: Uses a precomputed hash table for constant-time lookups.
- Bloom Filter: Utilizes multiple hash functions and a bit array to hash
                entries with low probabilities of collision.
- Cuckoo Filter:

Functions
---------
linear_search(arr: list[str], target: str) -> bool
    Performs a linear search for the target in the list.
binary_search(arr: list[str], target: str) -> bool
    Performs a binary search on a sorted list for the target.
generate_hash_table(arr: list[str]) -> dict[str, int]
    Builds a hash table mapping each element to its index.
hash_search(table: dict[str, int], target: str) -> bool
    Performs a constant-time search for the target in the hash table.
"""


@typechecked
def linear_search(arr: list[str], target: str) -> bool:
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
    bool
        True if the string is found, else False
    """
    for index, value in enumerate(arr):
        if value == target:
            return True
    return False


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
    bool
        True if the string is found, else False
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False


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


@typechecked
def hash_search(table: dict[str, int], target: str) -> bool:
    """
    Searches for a string in the hash table and returns true if present
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
    bool
        True if the string is found, else False

    See Also
    --------
    :func:`generate_hash_table`.
    """
    return table.get(True, False)
