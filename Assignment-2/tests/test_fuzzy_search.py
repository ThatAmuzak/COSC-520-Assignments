# test_fuzzy_search.py
from typing import List, Tuple

import pytest
from src.fuzzy_search import BKTree, Trie, VPTree, levenshtein
from typeguard import TypeCheckError

# Shared dataset and queries used across tests
WORDS = ["apple", "apply", "ape", "apricot", "banana"]
QUERY = "appel"
EXACT = "apple"


def _is_sorted_by_distance(results: List[Tuple[str, int]]) -> bool:
    """Helper: check non-decreasing distances in results."""
    dists = [d for (_w, d) in results]
    return all(d1 <= d2 for d1, d2 in zip(dists, dists[1:]))


def _best_distance(query: str, words: List[str]) -> int:
    return min(levenshtein(query, w) for w in words)


def test_levenshtein_symmetry_and_known_distance():
    assert levenshtein("", "") == 0
    assert levenshtein("kitten", "sitting") == 3
    # symmetry
    assert levenshtein("sitting", "kitten") == levenshtein("kitten", "sitting")


def test_levenshtein_types_enforced():
    # Expect TypeCheckError from typeguard-decorated function if wrong types passed
    with pytest.raises(TypeCheckError):
        # ints instead of strings
        levenshtein(123, 456)  # type: ignore


def test_trie_basic_returns_sorted_and_hits_top_match():
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
    Trie.maxDist = 3
    t = Trie()
    t.build(WORDS)
    res = t.find_closest(QUERY, count=1)
    assert len(res) <= 1
    if res:
        assert res[0][1] == _best_distance(QUERY, WORDS)


def test_trie_exact_match_with_zero_maxdist():
    # If maxDist == 0, only exact matches should be returned
    Trie.maxDist = 0
    t = Trie()
    t.build(WORDS)
    res_exact = t.find_closest(EXACT, count=5)
    assert ("apple", 0) in res_exact
    # non-exact query returns nothing
    res_non = t.find_closest(QUERY, count=5)
    assert res_non == []


def test_trie_returns_only_within_maxdist():
    Trie.maxDist = 1
    t = Trie()
    t.build(WORDS)
    res = t.find_closest(QUERY, count=10)
    # all distances must be <= maxDist
    assert all(d <= Trie.maxDist for (_w, d) in res)


def test_trie_type_errors_on_build():
    t = Trie()
    with pytest.raises(TypeCheckError):
        # build expects List[str]; passing a str or list-of-non-str should raise
        t.build("not-a-list")  # type: ignore


def test_bktree_basic_sorted_and_threshold_respected():
    BKTree.threshold = 3
    bk = BKTree()
    bk.build(WORDS)
    res = bk.find_closest(QUERY, count=5)
    assert isinstance(res, list)
    assert _is_sorted_by_distance(res)
    # all returned distances should be <= threshold
    assert all(d <= BKTree.threshold for (_w, d) in res)


def test_bktree_count_and_best_match():
    BKTree.threshold = 3
    bk = BKTree()
    bk.build(WORDS)
    res = bk.find_closest(QUERY, count=1)
    assert len(res) <= 1
    if res:
        assert res[0][1] == _best_distance(QUERY, WORDS)


def test_bktree_exact_with_zero_threshold():
    BKTree.threshold = 0
    bk = BKTree()
    bk.build(WORDS)
    res_exact = bk.find_closest(EXACT, count=5)
    assert ("apple", 0) in res_exact
    # non-exact should give empty (threshold 0)
    res_non = bk.find_closest(QUERY, count=5)
    assert res_non == []


def test_bktree_returns_subset_of_words_and_sorted():
    BKTree.threshold = 2
    bk = BKTree()
    bk.build(WORDS)
    res = bk.find_closest(QUERY, count=10)
    assert all(w in WORDS for (w, _d) in res)
    assert _is_sorted_by_distance(res)


def test_bktree_type_errors_on_find():
    BKTree.threshold = 2
    bk = BKTree()
    bk.build(WORDS)
    with pytest.raises(TypeCheckError):
        # find_closest expects str query; passing int should raise via typeguard
        bk.find_closest(123, count=3)  # type: ignore


def test_vptree_basic_sorted_and_threshold_respected():
    VPTree.threshold = 3
    vp = VPTree()
    vp.build(WORDS)
    res = vp.find_closest(QUERY, count=5)
    assert isinstance(res, list)
    assert _is_sorted_by_distance(res)
    assert all(d <= VPTree.threshold for (_w, d) in res)


def test_vptree_count_and_best_match():
    VPTree.threshold = 3
    vp = VPTree()
    vp.build(WORDS)
    res = vp.find_closest(QUERY, count=1)
    assert len(res) <= 1
    if res:
        assert res[0][1] == _best_distance(QUERY, WORDS)


def test_vptree_exact_with_zero_threshold():
    VPTree.threshold = 0
    vp = VPTree()
    vp.build(WORDS)
    res_exact = vp.find_closest(EXACT, count=5)
    assert ("apple", 0) in res_exact
    # non-exact should give empty (threshold 0)
    res_non = vp.find_closest(QUERY, count=5)
    assert res_non == []


def test_vptree_returns_only_known_words_and_sorted():
    VPTree.threshold = 2
    vp = VPTree()
    vp.build(WORDS)
    res = vp.find_closest(QUERY, count=10)
    assert all(w in WORDS for (w, _d) in res)
    assert _is_sorted_by_distance(res)


def test_vptree_type_errors_on_build():
    vp = VPTree()
    with pytest.raises(TypeCheckError):
        # build expects List[str]
        vp.build([1, 2, 3])  # type: ignore
