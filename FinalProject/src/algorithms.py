"""
algorithms.py

Collection of classical substring search algorithms for profiling

The implementations include:
- naive_search: Naive brute-force string matching algorithm.

    Function Signature:
    naive_search(pattern: str, text: str) -> List[int]:

- kmp_search: Knuth–Morris–Pratt algorithm using an LPS
    (longest proper prefix which is also suffix) table.

    Function Signature:
    kmp_search(pattern: str, text: str) -> List[int]:

- rabin_karp: Rabin–Karp algorithm with a rolling hash (large prime modulus).

    Function Signature:
    def rabin_karp(pattern: str, text: str) -> List[int]:

- boyer_moore: Full Boyer–Moore using bad-character and good-suffix heuristics.

    Function Signature:
    def boyer_moore(pattern: str, text: str) -> List[int]:

Notes
-----
These implementations are intended to be clear and reasonably efficient rather
than highly optimized for every edge case.
"""

from typing import List

from typeguard import typechecked


@typechecked
def naive_search(pattern: str, text: str) -> List[int]:
    """
    Naive brute-force substring search.

    This is the simplest string matching algorithm. It checks every possible
    position in the text by comparing characters one by one.
    
    Time complexity: O(n*m) in the worst case, where n = len(text) and 
    m = len(pattern). This occurs when there are many partial matches.
    
    Space complexity: O(1) beyond the result storage.
    """
    if pattern == "":
        return list(range(len(text) + 1))

    n, m = len(text), len(pattern)
    if m > n:
        return []

    res: List[int] = []
    
    # Try each possible starting position
    for i in range(n - m + 1):
        # Check if pattern matches at position i
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        
        if match:
            res.append(i)
    
    return res


@typechecked
def kmp_search(pattern: str, text: str) -> List[int]:
    """
    Knuth–Morris–Pratt (KMP) string search.

    Parameters
    ----------
    pattern : str
        The substring pattern to search for.
        If an empty string is provided, all valid start positions in `text` are
        returned (i.e. 0..len(text)).
    text : str
        The text in which to search for `pattern`.

    Returns
    -------
    list of int
        Starting indices where `pattern` occurs in `text`. If `pattern` is
        empty, returns a list containing every valid starting index in `text`.

    Examples
    --------
    >>> kmp_search("aba", "ababa")
    [0, 2]

    Notes
    -----
    This implementation first builds the LPS (longest proper prefix which
    is also suffix) table for the pattern in O(m) time, where m = len(pattern).
    The search then proceeds in O(n) time, where n = len(text), yielding
    an overall time complexity of O(n + m). Space complexity is O(m) for
    the LPS table.
    """
    if pattern == "":
        return list(range(len(text) + 1))

    # Build LPS (longest proper prefix which is also suffix) array.
    m = len(pattern)
    lps = [0] * m
    # length of previous longest prefix suffix
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # fall back to the previous possible prefix length
                length = lps[length - 1]
                # do not increment i here
            else:
                lps[i] = 0
                i += 1

    # Search using LPS to skip unnecessary comparisons
    res: List[int] = []
    i = 0  # index in text
    j = 0  # index in pattern
    n = len(text)
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                # match found; append starting index
                res.append(i - j)
                # prepare j for the next potential match
                j = lps[j - 1]
        else:
            if j != 0:
                # roll j back using lps to avoid re-checking characters
                j = lps[j - 1]
            else:
                i += 1
    return res


@typechecked
def rabin_karp(pattern: str, text: str) -> List[int]:
    """
    Rabin–Karp substring search using a rolling hash.

    Parameters
    ----------
    pattern : str
        The substring pattern to search for. If an empty string is provided,
        all valid start positions in `text` are returned.
    text : str
        The text in which to search for `pattern`.

    Returns
    -------
    list of int
        Starting indices where `pattern` occurs in `text`. If `pattern` is
        empty, returns a list containing every valid starting index in `text`.

    Notes
    -----
    This implementation uses a rolling hash with parameters:
    - base = 256 (number of possible byte values / extended ASCII)
    - mod = 2**61 - 1 (a large Mersenne-like prime to reduce collisions)

    When a rolling hash match is found, a full substring comparison is
    performed to avoid false positives due to hash collisions.

    Average-case time complexity is O(n + m), but collisions can degrade
    performance in adversarial cases (worst-case O(n*m) if many spurious
    matches occur). Space complexity is O(1) beyond the input and result
    storage.

    Examples
    --------
    >>> rabin_karp("abc", "abcabc")
    [0, 3]
    """
    if pattern == "":
        return list(range(len(text) + 1))

    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Parameters for rolling hash
    base = 256  # number of possible characters (extended ASCII)
    mod = 2**61 - 1  # large Mersenne-like prime (useful to reduce collisions)

    def modmul(a: int, b: int) -> int:
        # modular multiplication tailored for large prime;
        # Python's big ints handle this fine
        return (a * b) % mod

    # compute base^(m-1) % mod for use when removing leading char
    power = 1
    for _ in range(m - 1):
        power = modmul(power, base)

    # compute initial hashes
    pat_hash = 0
    txt_hash = 0
    for i in range(m):
        pat_hash = (modmul(pat_hash, base) + ord(pattern[i])) % mod
        txt_hash = (modmul(txt_hash, base) + ord(text[i])) % mod

    res: List[int] = []
    for i in range(n - m + 1):
        # if hashes match, do full string comparison to avoid false positives
        if txt_hash == pat_hash:
            if text[i : i + m] == pattern:
                res.append(i)
        if i < n - m:
            # remove leading char and add next char to rolling hash
            leading = modmul(ord(text[i]), power)
            # subtract leading char
            txt_hash = (txt_hash + mod - leading) % mod
            txt_hash = (modmul(txt_hash, base) + ord(text[i + m])) % mod
    return res


@typechecked
def boyer_moore(pattern: str, text: str) -> List[int]:
    """
    Boyer–Moore substring search using bad-character and good-suffix heuristics

    Parameters
    ----------
    pattern : str
        The substring pattern to search for. If an empty string is provided,
        all valid start positions in `text` are returned.
    text : str
        The text in which to search for `pattern`.

    Returns
    -------
    list of int
        Starting indices where `pattern` occurs in `text`. If `pattern` is
        empty, returns a list containing every valid starting index in `text`.

    Notes
    -----
    This implementation constructs:
    - a bad-character table mapping characters to their last index in pattern,
    - a good-suffix shift table that determines how far to shift the pattern
        when a mismatch occurs.

    Boyer–Moore often performs sub-linearly on typical inputs because it skips
    portions of the text, but its worst-case time complexity is O(n + m). Space
    complexity is O(m) for the auxiliary tables.

    Examples
    --------
    >>> boyer_moore("needle", "find a needle in a haystack with needle")
    [7, 33]
    """
    if pattern == "":
        return list(range(len(text) + 1))

    n, m = len(text), len(pattern)
    if m > n:
        return []

    # --- Bad character table ---
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    # --- Good suffix table ---
    # Step 1: suffixes[i] = length of longest suffix of pattern[:i+1]
    # that is also a prefix of pattern
    suffixes = [0] * m
    suffixes[m - 1] = m
    g, f = m - 1, m - 1
    for i in range(m - 2, -1, -1):
        if i > g and suffixes[i + m - 1 - f] < i - g:
            suffixes[i] = suffixes[i + m - 1 - f]
        else:
            if i < g:
                g = i
            f = i
            while g >= 0 and pattern[g] == pattern[g + m - 1 - f]:
                g -= 1
            suffixes[i] = f - g

    # Step 2: build good suffix shift table
    good_suffix = [m] * m
    j = 0
    for i in range(m - 1, -1, -1):
        if suffixes[i] == i + 1:
            while j < m - 1 - i:
                if good_suffix[j] == m:
                    good_suffix[j] = m - 1 - i
                j += 1
    for i in range(m - 1):
        good_suffix[m - 1 - suffixes[i]] = m - 1 - i

    # --- Search ---
    res: List[int] = []
    s = 0  # shift of pattern relative to text
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            # match found
            res.append(s)
            s += good_suffix[0]  # shift by full match good suffix
        else:
            char_shift = j - bad_char.get(text[s + j], -1)
            gs_shift = good_suffix[j]
            s += max(1, char_shift, gs_shift)
    return res