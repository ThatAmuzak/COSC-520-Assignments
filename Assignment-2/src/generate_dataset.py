import hashlib
import string

import numpy as np
from tqdm import tqdm

N = 10_000_000
alphabet = string.ascii_letters + string.digits
base = len(alphabet)


def int_to_base(n, base_chars):
    """Convert int to string using given alphabet."""
    base = len(base_chars)
    s = []
    while n:
        n, r = divmod(n, base)
        s.append(base_chars[r])
    return "".join(reversed(s)) or base_chars[0]


data = np.empty(N, dtype=object)
for i in tqdm(range(N), desc="Generating unique alphanumerics"):
    # hash → int → alphanumeric
    h_int = int(hashlib.sha1(str(i).encode()).hexdigest(), 16)
    s = int_to_base(h_int, alphabet)[:10]  # take first 6 chars
    data[i] = s
assert len(data) == len(set(data))
np.save("src/string_dataset.npy", data)
