"""
Dataset generation script for creating a large array of unique alphanumeric strings.

This script generates N unique alphanumeric strings by hashing integers from 0 to N-1,
converting the hash to an integer, then encoding that integer into a base defined by
an alphanumeric alphabet. The resulting strings are truncated to a fixed length and
stored in a numpy array, which is saved to disk as a .npy file.

Attributes:
    N (int): Number of unique strings to generate.
    alphabet (str): Characters used for base conversion (letters + digits).
"""

import hashlib
import string

import numpy as np
from tqdm import tqdm

N = 10_000_000
alphabet = string.ascii_letters + string.digits
base = len(alphabet)


def int_to_base(n, base_chars):
    """
    Convert an integer to a string representation using the provided alphabet as digits.

    Args:
        n (int): The integer to convert.
        base_chars (str): The characters representing digits in the target base.

    Returns:
        str: The string representation of the integer in the given base.
    """
    base = len(base_chars)
    s = []
    while n:
        n, r = divmod(n, base)
        s.append(base_chars[r])
    return "".join(reversed(s)) or base_chars[0]


# Pre-allocate numpy array to hold the generated strings
data = np.empty(N, dtype=object)

# Generate unique alphanumeric strings by hashing integers and converting to base
for i in tqdm(range(N), desc="Generating unique alphanumerics"):
    # Compute SHA-1 hash of the integer as a string, convert to integer
    h_int = int(hashlib.sha1(str(i).encode()).hexdigest(), 16)
    # Convert hash integer to alphanumeric string and truncate to length 10
    s = int_to_base(h_int, alphabet)[:10]
    data[i] = s

# Verify all generated strings are unique
assert len(data) == len(set(data))

# Save the dataset to a numpy binary file
np.save("src/string_dataset.npy", data)
