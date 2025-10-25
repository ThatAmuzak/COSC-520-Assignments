"""
Test suite for fuzzy search data structures and algorithms.

This module tests the BKTree, Trie, and VPTree implementations along with the
levenshtein distance function. It verifies correctness, type safety, sorting,
and threshold constraints for fuzzy search results.

The tests use a shared dataset of words and queries to validate expected
behavior of the fuzzy search classes and their methods.
"""

from typing import List, Tuple

import pytest
from src.fuzzy_search import BKTree, Trie, VPTree, levenshtein
from typeguard import TypeCheckError

# Shared dataset and queries used across tests
WORDS = ["apple", "apply", "ape", "apricot", "banana"]
QUERY = "appel"
EXACT = "apple"


def _is_sorted_by_distance(results: List[Tuple[str, int]]) -> bool:
    """
    Helper function to check if the list of (word, distance) tuples is sorted
    by non-decreasing distance values.
    """
    dists = [d for (_w, d) in results]
    return all(d1 <= d2 for d1, d2 in zip(dists, dists[1:]))


def _best_distance(query: str, words: List[str]) -> int:
    """
    Helper function to compute the minimum Levenshtein distance between the
    query and a list of words.
    """
    return min(levenshtein(query, w) for w in words)


def test_levenshtein_symmetry_and_known_distance():
    """
    Test the levenshtein function for known distances and symmetry property.
    """
    assert levenshtein("", "") == 0
    assert levenshtein("kitten", "sitting") == 3
    # symmetry
    assert levenshtein("sitting", "kitten") == levenshtein("kitten", "sitting")


def test_levenshtein_types_enforced():
    """
    Test that levenshtein function enforces input types and raises TypeCheckError
    when called with invalid types.
    """
    with pytest.raises(TypeCheckError):
        # ints instead of strings
        levenshtein(123, 456)  # type: ignore


def test_trie_basic_returns_sorted_and_hits_top_match():
    """
    Test Trie basic functionality: building, finding closest matches, and
    verifying results are sorted and include the best match.
    """
    Trie.maxDist = 3  # ensure some slack
    t = Trie()
    t.build(WORDS)
    res = t.find_closest(QUERY, count=3)
    assert isinstance(res, list)
    # All returned words must be from the original list
    assert all(w in WORDS for w, _ in res)
    # distances non-decreasing
    assert _is_sorted_by_distance(res)
    # best result distance equals global best distance (if any results)
    if res:
        assert res[0][1] == _best_distance(QUERY, WORDS)


def test_trie_count_limit_and_length():
    """
    Test that Trie respects the count limit and returns at most that many results.
    """
    Trie.maxDist = 3
    t = Trie()
    t.build(WORDS)
    res = t.find_closest(QUERY, count=1)
    assert len(res) <= 1
    if res:
        assert res[0][1] == _best_distance(QUERY, WORDS)


def test_trie_exact_match_with_zero_maxdist():
    """
    Test Trie behavior when maxDist is zero: only exact matches should be returned.
    """
    Trie.maxDist = 0
    t = Trie()
    t.build(WORDS)
    res_exact = t.find_closest(EXACT, count=5)
    assert ("apple", 0) in res_exact
    # non-exact query returns nothing
    res_non = t.find_closest(QUERY, count=5)
    assert res_non == []


def test_trie_returns_only_within_maxdist():
    """
    Test that Trie returns only results within the maxDist threshold.
    """
    Trie.maxDist = 1
    t = Trie()
    t.build(WORDS)
    res = t.find_closest(QUERY, count=10)
    # all distances must be <= maxDist
    assert all(d <= Trie.maxDist for (_w, d) in res)


def test_trie_type_errors_on_build():
    """
    Test that Trie.build raises TypeCheckError when called with invalid input types.
    """
    t = Trie()
    with pytest.raises(TypeCheckError):
        # build expects List[str]; passing a str or list-of-non-str should raise
        t.build("not-a-list")  # type: ignore


def test_bktree_basic_sorted_and_threshold_respected():
    """
    Test BKTree basic functionality: building, finding closest matches, and
    verifying results are sorted and within threshold.
    """
    BKTree.threshold = 3
    bk = BKTree()
    bk.build(WORDS)
    res = bk.find_closest(QUERY, count=5)
    assert isinstance(res, list)
    assert _is_sorted_by_distance(res)
    # all returned distances should be <= threshold
    assert all(d <= BKTree.threshold for (_w, d) in res)


def test_bktree_count_and_best_match():
    """
    Test BKTree respects count limit and returns the best match.
    """
    BKTree.threshold = 3
    bk = BKTree()
    bk.build(WORDS)
    res = bk.find_closest(QUERY, count=1)
    assert len(res) <= 1
    if res:
        assert res[0][1] == _best_distance(QUERY, WORDS)


def test_bktree_exact_with_zero_threshold():
    """
    Test BKTree behavior with zero threshold: only exact matches returned.
    """
    BKTree.threshold = 0
    bk = BKTree()
    bk.build(WORDS)
    res_exact = bk.find_closest(EXACT, count=5)
    assert ("apple", 0) in res_exact
    # non-exact should give empty (threshold 0)
    res_non = bk.find_closest(QUERY, count=5)
    assert res_non == []


def test_bktree_returns_subset_of_words_and_sorted():
    """
    Test BKTree returns only known words and results are sorted by distance.
    """
    BKTree.threshold = 2
    bk = BKTree()
    bk.build(WORDS)
    res = bk.find_closest(QUERY, count=10)
    assert all(w in WORDS for (w, _d) in res)
    assert _is_sorted_by_distance(res)


def test_bktree_type_errors_on_find():
    """
    Test BKTree.find_closest raises TypeCheckError when called with invalid query type.
    """
    BKTree.threshold = 2
    bk = BKTree()
    bk.build(WORDS)
    with pytest.raises(TypeCheckError):
        # find_closest expects str query; passing int should raise via typeguard
        bk.find_closest(123, count=3)  # type: ignore


def test_vptree_basic_sorted_and_threshold_respected():
    """
    Test VPTree basic functionality: building, finding closest matches, and
    verifying results are sorted and within threshold.
    """
    VPTree.threshold = 3
    vp = VPTree()
    vp.build(WORDS)
    res = vp.find_closest(QUERY, count=5)
    assert isinstance(res, list)
    assert _is_sorted_by_distance(res)
    assert all(d <= VPTree.threshold for (_w, d) in res)


def test_vptree_count_and_best_match():
    """
    Test VPTree respects count limit and returns the best match.
    """
    VPTree.threshold = 3
    vp = VPTree()
    vp.build(WORDS)
    res = vp.find_closest(QUERY, count=1)
    assert len(res) <= 1
    if res:
        assert res[0][1] == _best_distance(QUERY, WORDS)


def test_vptree_exact_with_zero_threshold():
    """
    Test VPTree behavior with zero threshold: only exact matches returned.
    """
    VPTree.threshold = 0
    vp = VPTree()
    vp.build(WORDS)
    res_exact = vp.find_closest(EXACT, count=5)
    assert ("apple", 0) in res_exact
    # non-exact should give empty (threshold 0)
    res_non = vp.find_closest(QUERY, count=5)
    assert res_non == []


def test_vptree_returns_only_known_words_and_sorted():
    """
    Test VPTree returns only known words and results are sorted by distance.
    """
    VPTree.threshold = 2
    vp = VPTree()
    vp.build(WORDS)
    res = vp.find_closest(QUERY, count=10)
    assert all(w in WORDS for (w, _d) in res)
    assert _is_sorted_by_distance(res)


def test_vptree_type_errors_on_build():
    """
    Test VPTree.build raises TypeCheckError when called with invalid input types.
    """
    vp = VPTree()
    with pytest.raises(TypeCheckError):
        # build expects List[str]
        vp.build([1, 2, 3])  # type: ignore
