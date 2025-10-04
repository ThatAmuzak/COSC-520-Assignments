import math
import random
from typing import Tuple

import mmh3
from bitarray import bitarray
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
- Cuckoo Filter: Utilizes 2 hash functions 

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
    return True if table.get(target, -1) != -1 else False


@typechecked
class BloomFilter:
    """
    Python implementation of Bloom filters with bitarray
    Implemented following
    https://techtonics.medium.com/implementing-bloom-filters-in-python-and-understanding-its-error-probability-a-step-by-step-guide-13c6cb2e05b7
    and utilizing bitarrays over regular arrays for better space efficiency

    Time Complexity (to check membership): O(k)
    Space Complexity: O(m)
    where k is the number of hash functions utilized, and m is number of bits in our hashmap
    """

    def __init__(self, capacity: int = 150_000_000, error_rate: float = 0.01) -> None:
        """
        Initialize a Bloom filter with a given capacity and false positive error rate.

        Parameters:
            capacity (int): Maximum number of elements the Bloom filter should hold.
                            Must be greater than 0.
            error_rate (float): Desired false positive probability (between 0 and 1).

        Raises:
            ValueError: If capacity <= 0 or error_rate is not between 0 and 1.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be > 0")
        if not (0 < error_rate < 1):
            raise ValueError("Error rate must be between 0 and 1")

        self.capacity = capacity
        self.error_rate = error_rate
        self.num_bits = self._get_num_bits(capacity, error_rate)
        self.num_hashes = self._get_num_hashes(self.num_bits, capacity)
        self.bit_array = bitarray(self.num_bits)
        self.bit_array.setall(0)

    # ----- helpers -----
    def _get_num_bits(self, capacity: int, error_rate: float) -> int:
        num_bits = -(capacity * math.log(error_rate)) / (math.log(2) ** 2)
        return int(num_bits)

    def _get_num_hashes(self, num_bits: int, capacity: int) -> int:
        num_hashes = (num_bits / capacity) * math.log(2)
        return int(num_hashes)

    # ----- public API -----
    def add(self, element: str) -> None:
        """
        Add an element to the Bloom filter.

        Parameters:
            element (str): The element (string) to insert into the filter.
        """
        for i in range(self.num_hashes):
            hash_val = mmh3.hash(element, i) % self.num_bits
            self.bit_array[hash_val] = 1

    def get(self, element: str) -> bool:
        """
        Check whether an element is possibly in the Bloom filter.

        Parameters:
            element (str): The element (string) to check.

        Returns:
            bool:
                - True if the element may be in the set (with false positive probability).
                - False if the element is definitely not in the set.
        """
        for i in range(self.num_hashes):
            hash_val = mmh3.hash(element, i) % self.num_bits
            if not self.bit_array[hash_val]:
                return False
        return True


@typechecked
class CuckooFilter:
    """
    Compact Cuckoo filter using bitarray storage and mmh3 hashing.
    Implemented following
    https://github.com/huydhn/cuckoo-filter

    Time Complexity (to check membership): O(1)
    Space Complexity: O(m)
    where m is number of bits in our hashmap
    """

    def __init__(
        self,
        capacity: int,
        error_rate: float = 0.01,
        bucket_size: int = 4,
        max_kicks: int = 500,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.bucket_size = int(bucket_size)
        self.max_kicks = int(max_kicks)
        self.error_rate = (
            float(error_rate)
            if error_rate is not None
            else CuckooFilter.DEFAULT_ERROR_RATE
        )

        # minimal fingerprint size in bits: ceil(log2(1/e) + log2(2b))
        min_fp = math.log2(1.0 / self.error_rate) + math.log2(2 * self.bucket_size)
        self.fingerprint_size: int = int(math.ceil(min_fp))

        # compact storage: capacity * bucket_size * fingerprint_size bits
        total_bits = self.capacity * self.bucket_size * self.fingerprint_size
        self._bits = bitarray(total_bits)
        self._bits.setall(0)
        self.size = 0

    # ----- helpers -----
    def _ensure_bytes(self, item: str) -> bytes:
        if isinstance(item, bytes):
            return item
        if isinstance(item, str):
            return item.encode("utf-8")
        return str(item).encode("utf-8")

    def _mmh3_bytes(self, b: bytes) -> bytes:
        return mmh3.hash_bytes(b)

    def _index_from_hash(self, hbytes: bytes) -> int:
        return int.from_bytes(hbytes[:8], "big") % self.capacity

    def _fingerprint(self, item_bytes: bytes) -> bitarray:
        digest = self._mmh3_bytes(item_bytes)
        bits = bitarray()
        bits.frombytes(digest)
        fp = bits[: self.fingerprint_size]
        if fp.count(True) == 0:
            fp[-1] = 1
        return fp

    def _bit_index(self, bucket_index: int) -> Tuple[int, int]:
        start = bucket_index * self.bucket_size * self.fingerprint_size
        end = start + (self.bucket_size * self.fingerprint_size)
        return start, end

    def _slot_positions(self, bucket_index: int) -> Tuple[int, ...]:
        s, e = self._bit_index(bucket_index)
        step = self.fingerprint_size
        return tuple(range(s, e, step))

    # bucket operations
    def _include(self, fingerprint: bitarray, bucket_index: int) -> bool:
        for pos in self._slot_positions(bucket_index):
            if self._bits[pos : pos + self.fingerprint_size] == fingerprint:
                return True
        return False

    def _insert_into_bucket(self, fingerprint: bitarray, bucket_index: int) -> bool:
        for pos in self._slot_positions(bucket_index):
            stored = self._bits[pos : pos + self.fingerprint_size]
            if stored.count(True) == 0:  # empty slot
                self._bits[pos : pos + self.fingerprint_size] = fingerprint
                return True
        return False

    def _delete_from_bucket(self, fingerprint: bitarray, bucket_index: int) -> bool:
        for pos in self._slot_positions(bucket_index):
            if self._bits[pos : pos + self.fingerprint_size] == fingerprint:
                empty = bitarray(self.fingerprint_size)
                empty.setall(0)
                self._bits[pos : pos + self.fingerprint_size] = empty
                return True
        return False

    def _swap_random(self, fingerprint: bitarray, bucket_index: int) -> bitarray:
        positions = list(self._slot_positions(bucket_index))
        rpos = random.choice(positions)
        swapped = self._bits[rpos : rpos + self.fingerprint_size]
        self._bits[rpos : rpos + self.fingerprint_size] = fingerprint
        return swapped

    def _indices(self, item_bytes: bytes, fingerprint: bitarray) -> Tuple[int, int]:
        h_item = self._mmh3_bytes(item_bytes)
        i1 = self._index_from_hash(h_item)
        h_fp = self._mmh3_bytes(fingerprint.tobytes())
        i2 = (i1 ^ self._index_from_hash(h_fp)) % self.capacity
        return i1, i2

    # ----- public API -----
    def insert(self, item: str) -> bool:
        """
        Insert item. Returns True on success.
        Raises CapacityException (ValueError) if insertion fails after max_kicks.
        """
        item_b = self._ensure_bytes(item)
        fp = self._fingerprint(item_b)
        i1, i2 = self._indices(item_b, fp)

        # try direct insert
        if self._insert_into_bucket(fp, i1) or self._insert_into_bucket(fp, i2):
            self.size += 1
            return True

        # need to relocate
        cur_index = random.choice((i1, i2))
        cur_fp = fp
        for _ in range(self.max_kicks):
            swapped = self._swap_random(cur_fp, cur_index)
            cur_fp = swapped
            cur_index = (
                cur_index ^ self._index_from_hash(self._mmh3_bytes(cur_fp.tobytes()))
            ) % self.capacity
            if self._insert_into_bucket(cur_fp, cur_index):
                self.size += 1
                return True

        raise ValueError(
            f"Cuckoo filter full after {self.max_kicks} kicks (size={self.size})"
        )

    def contains(self, item: str) -> bool:
        item_b = self._ensure_bytes(item)
        fp = self._fingerprint(item_b)
        i1, i2 = self._indices(item_b, fp)
        return self._include(fp, i1) or self._include(fp, i2)

    def delete(self, item: str) -> bool:
        item_b = self._ensure_bytes(item)
        fp = self._fingerprint(item_b)
        i1, i2 = self._indices(item_b, fp)
        if self._delete_from_bucket(fp, i1) or self._delete_from_bucket(fp, i2):
            self.size -= 1
            return True
        return False

    def load_factor(self) -> float:
        return round(float(self.size) / (self.capacity * self.bucket_size), 6)
