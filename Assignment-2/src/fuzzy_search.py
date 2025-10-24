"""
fuzzy_search.py

Contents
--------
- levenshtein(a: str, b: str) -> int
    Compute the Levenshtein edit distance between two strings (insert/delete/replace)
    using the Wagner Fischer algorithm.

- class Trie
    A Trie (prefix tree) that supports:
      - Trie.build(words: List[str]) -> None
      - Trie.find_closest(query: str, count: int = 10) -> List[Tuple[str, int]]

    Behavior: Maintains a class variable `maxDist` (int). `find_closest` performs an
    approximate search by traversing the trie while keeping the Levenshtein DP row
    at each node. A terminal node is a match if the DP last cell <= maxDist. Branches
    are pruned when min(DP_row) > maxDist. The function returns up to `count` matches
    as (word, distance) pairs sorted by increasing distance.

- class BKTree
    A Burkhard-Keller tree using Levenshtein distance as metric with functions:
      - BKTree.build(words: List[str]) -> None
      - BKTree.find_closest(query: str, count: int = 10) -> List[Tuple[str, int]]

    Behavior: Has a class variable `threshold` (int). `find_closest` searches children
    whose edge distance is in [d - threshold, d + threshold], where d is the distance
    between the query and the node's term. Returns up to `count` matches with distance
    <= threshold, sorted by distance.

- class VPTree
    A vantage-point tree using Levenshtein distance with functions:
      - VPTree.build(words: List[str]) -> None
      - VPTree.find_closest(query: str, count: int = 10) -> List[Tuple[str, int]]

    Behavior: Has a class variable `threshold` (int) used as the +/- window when deciding
    which child subtrees may contain closer points. `find_closest` returns up to `count`
    matches with distance <= threshold, sorted by distance.

Notes
----
- All public callables are annotated with type hints and decorated with
  `typeguard.typechecked` to enforce type safety at runtime.

Example usage
-------------
>>> from fuzzy_search import Trie, BKTree, VPTree, levenshtein
>>> words = ["apple", "apply", "ape", "apricot", "banana"]
>>> t = Trie(); t.build(words)
>>> t.find_closest("appel", count=3)

"""

import heapq
import random
from typing import Dict, List, Optional, Tuple

from typeguard import typechecked


@typechecked
def levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Parameters
    ----------
    a : str
        First input string.
    b : str
        Second input string.

    Returns
    -------
    int
        The edit distance (number of single-character insertions, deletions, or substitutions)
        required to transform `a` into `b`.
    """
    # Use the dynamic programming approach with O(len(a)*len(b)) time and
    # O(min(len(a), len(b))) space optimization.
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure b is the longer one to use less space
    if len(a) > len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = previous_row[j] + 1
            delete_cost = current_row[j - 1] + 1
            replace_cost = previous_row[j - 1] + (0 if ca == cb else 1)
            current_row[j] = min(insert_cost, delete_cost, replace_cost)
        previous_row = current_row
    return previous_row[-1]


@typechecked
class Trie:
    """A Trie supporting approximate (Levenshtein) search.

    Class variables
    ---------------
    maxDist : int
        The maximum distance allowed for a match during `find_closest`.

    Methods
    -------
    build(words: List[str]) -> None
        Add a list of words to the trie.

    find_closest(query: str, count: int = 10) -> List[Tuple[str, int]]
        Return up to `count` closest words from the trie whose Levenshtein distance
        to `query` is <= `maxDist`. Results are sorted by distance (ascending).
    """

    maxDist: int = 5

    class _Node:
        __slots__ = ("children", "is_word")

        def __init__(self) -> None:
            self.children: Dict[str, "Trie._Node"] = {}
            self.is_word: bool = False

    def __init__(self) -> None:
        self.root = Trie._Node()

    def build(self, words: List[str]) -> None:
        """Insert multiple words into the trie.

        Parameters
        ----------
        words : List[str]
            List of strings to insert into the trie.
        """
        for w in words:
            node = self.root
            for ch in w:
                if ch not in node.children:
                    node.children[ch] = Trie._Node()
                node = node.children[ch]
            node.is_word = True

    def _search_recursive(
        self,
        node: "Trie._Node",
        prefix: str,
        previous_row: List[int],
        query: str,
        results: List[Tuple[str, int]],
    ) -> None:
        # previous_row has length len(query) + 1
        cols = len(query) + 1
        for ch, child in node.children.items():
            # compute current_row for prefix + ch
            current_row = [previous_row[0] + 1]
            # fill current_row
            for col in range(1, cols):
                insert_cost = current_row[col - 1] + 1
                delete_cost = previous_row[col] + 1
                replace_cost = previous_row[col - 1] + (
                    0 if query[col - 1] == ch else 1
                )
                current_row.append(min(insert_cost, delete_cost, replace_cost))

            # pruning: if the smallest value in current_row > maxDist, skip branch
            if min(current_row) <= self.maxDist:
                new_prefix = prefix + ch
                if child.is_word and current_row[-1] <= self.maxDist:
                    results.append((new_prefix, current_row[-1]))
                # recurse
                self._search_recursive(child, new_prefix, current_row, query, results)

    def find_closest(self, query: str, count: int = 10) -> List[Tuple[str, int]]:
        """Find up to `count` closest words to `query` using trie + Levenshtein DP.

        Parameters
        ----------
        query : str
            Query string to match.
        count : int, optional
            Maximum number of results to return (default 10).

        Returns
        -------
        List[Tuple[str, int]]
            List of (word, distance) sorted by increasing distance. If fewer than
            `count` words are within `maxDist`, fewer results are returned.
        """
        if not query:
            # Special-case: empty query -> return words whose length <= maxDist
            results: List[Tuple[str, int]] = []
            # We'll still use the trie algorithm: initial row is range
        results = []
        # initial DP row for empty prefix is 0..len(query)
        initial_row = list(range(len(query) + 1))
        # check root children recursively
        if self.root.is_word and initial_row[-1] <= self.maxDist:
            results.append(("", initial_row[-1]))
        self._search_recursive(self.root, "", initial_row, query, results)
        # sort and return up to count
        results.sort(key=lambda x: (x[1], x[0]))
        return results[:count]


@typechecked
class BKTree:
    """Burkhard-Keller (BK) tree for discrete metric spaces using Levenshtein distance.

    Class variables
    ---------------
    threshold : int
        Search radius. `find_closest` will look for words with distance <= threshold.

    Methods
    -------
    build(words: List[str]) -> None
        Build the BK-tree from a list of words (first word becomes root if present).

    find_closest(query: str, count: int = 10) -> List[Tuple[str, int]]
        Return up to `count` words within `threshold` distance of `query`, sorted by
        ascending distance.
    """

    threshold: int = 5

    class _Node:
        __slots__ = ("term", "children")

        def __init__(self, term: str) -> None:
            self.term: str = term
            self.children: Dict[int, "BKTree._Node"] = {}

    def __init__(self) -> None:
        self.root: Optional[BKTree._Node] = None

    def build(self, words: List[str]) -> None:
        """Insert words into the BK-tree.

        Parameters
        ----------
        words : List[str]
            List of strings to insert.
        """
        for w in words:
            if self.root is None:
                self.root = BKTree._Node(w)
                continue
            node = self.root
            while True:
                d = levenshtein(w, node.term)
                if d in node.children:
                    node = node.children[d]
                else:
                    node.children[d] = BKTree._Node(w)
                    break

    def find_closest(self, query: str, count: int = 10) -> List[Tuple[str, int]]:
        """Search for words within `threshold` distance of `query`.

        Parameters
        ----------
        query : str
            Query string.
        count : int, optional
            Maximum number of results to return.

        Returns
        -------
        List[Tuple[str, int]]
            Found words as (word, distance), sorted by ascending distance. Up to
            `count` entries are returned and only words with distance <= `threshold`
            are considered.
        """
        if self.root is None:
            return []

        results: List[Tuple[str, int]] = []
        stack = [self.root]
        t = self.threshold
        while stack:
            node = stack.pop()
            d = levenshtein(query, node.term)
            if d <= t:
                results.append((node.term, d))
            # children to consider are in [d - t, d + t]
            lower = d - t
            upper = d + t
            for dist_k, child in node.children.items():
                if lower <= dist_k <= upper:
                    stack.append(child)
        results.sort(key=lambda x: (x[1], x[0]))
        return results[:count]


@typechecked
class VPTree:
    """Vantage-Point Tree (VP-tree) for approximate nearest neighbor search using
    Levenshtein distance.

    Class variables
    ---------------
    threshold : int
        Search radius / pruning threshold used during `find_closest`.

    Methods
    -------
    build(words: List[str]) -> None
        Build the VP-tree from a list of words.

    find_closest(query: str, count: int = 10) -> List[Tuple[str, int]]
        Return up to `count` closest words within `threshold`.
    """

    threshold: int = 5

    class _Node:
        __slots__ = ("point", "mu", "left", "right")

        def __init__(self, point: str) -> None:
            self.point: str = point
            self.mu: Optional[int] = None
            self.left: Optional["VPTree._Node"] = None
            self.right: Optional["VPTree._Node"] = None

    def __init__(self) -> None:
        self.root: Optional[VPTree._Node] = None

    def build(self, words: List[str]) -> None:
        """Build a VP-tree. This chooses vantage points randomly to avoid worst-case
        degenerate trees.

        Parameters
        ----------
        words : List[str]
            Items (strings) to insert into the tree.
        """
        items = list(words)

        def _build_recursive(items_list: List[str]) -> Optional[VPTree._Node]:
            if not items_list:
                return None
            # choose vantage point randomly for balance
            vp = items_list.pop(random.randrange(len(items_list)))
            node = VPTree._Node(vp)
            if not items_list:
                return node
            # compute distances to vantage point
            dists = [(levenshtein(vp, it), it) for it in items_list]
            # median
            dists.sort(key=lambda x: x[0])
            median_index = len(dists) // 2
            node.mu = dists[median_index][0]
            left_items = [it for dist, it in dists[:median_index]]
            right_items = [it for dist, it in dists[median_index:]]
            node.left = _build_recursive(left_items)
            node.right = _build_recursive(right_items)
            return node

        # build from copy
        self.root = _build_recursive(items)

    def find_closest(self, query: str, count: int = 10) -> List[Tuple[str, int]]:
        """Find up to `count` closest items to `query` with distance <= `threshold`.

        Parameters
        ----------
        query : str
            Query string.
        count : int, optional
            Maximum number of results to return.

        Returns
        -------
        List[Tuple[str, int]]
            Found items as (word, distance), sorted by ascending distance.
        """
        if self.root is None:
            return []

        # Use a max-heap to store best candidates (distance negative for max-heap using heapq)
        heap: List[Tuple[int, str]] = []  # (-distance, word)
        tau = self.threshold

        def _maybe_add(point: str, dist: int) -> None:
            if dist <= tau:
                item = (-dist, point)
                if len(heap) < count:
                    heapq.heappush(heap, item)
                else:
                    # if this is better than worst in heap, replace
                    if item > heap[0]:
                        heapq.heapreplace(heap, item)

        def _search(node: Optional[VPTree._Node]) -> None:
            if node is None:
                return
            d = levenshtein(query, node.point)
            _maybe_add(node.point, d)
            if node.mu is None:
                # leaf node
                return
            # distances to median
            if d < node.mu:
                # check left first (closer)
                if node.left is not None:
                    _search(node.left)
                # Decide whether to search right: points in right subtree may have distance within (d +/- tau)
                if (
                    node.right is not None and (d + tau) >= node.mu - 0
                ):  # conservative check
                    _search(node.right)
            else:
                # check right first
                if node.right is not None:
                    _search(node.right)
                if (
                    node.left is not None and (d - tau) <= node.mu + 0
                ):  # conservative check
                    _search(node.left)

        _search(self.root)
        # convert heap to sorted list
        results = [(w, -d) for (d, w) in heap]
        results.sort(key=lambda x: (x[0], x[1]))
        return results[:count]
