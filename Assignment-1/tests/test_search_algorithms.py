import pytest
from src.search_algorithms import (
    BloomFilter,
    binary_search,
    generate_hash_table,
    hash_search,
    linear_search,
)
from typeguard import TypeCheckError


def test_linear_search():
    """
    Tests the linear search function
    We only test for existence, since
    that's the login checker's goal.
    """
    assert linear_search(["aa"], "aa")  # should find
    assert not linear_search([], "aa")  # should not find
    assert not linear_search(["aa"], "")  # should not find
    assert linear_search(["a", "b", "c"], "b")  # should find
    assert linear_search(["c", "b", "a"], "b")  # should find
    assert not linear_search(["a", "b", "c"], "d")  # should not find
    with pytest.raises(TypeCheckError):
        linear_search(1, "aa")  # should raise type error
    with pytest.raises(TypeCheckError):
        linear_search(["aa"], 1)  # should raise type error


def test_binary_search():
    """
    Tests the binary search function
    We only test for existence, since
    that's the login checker's goal.
    """
    assert binary_search(["aa"], "aa")  # should find
    assert not binary_search([], "aa")  # should not find
    assert not binary_search(["aa"], "")  # should not find
    assert binary_search(["a", "b", "c"], "b")  # should find
    assert not binary_search(["a", "b", "c"], "d")  # should not find
    assert binary_search(["a", "b", "c", "d"], "c")  # should find
    with pytest.raises(TypeCheckError):
        binary_search(1, "a")  # should raise type error
    with pytest.raises(TypeCheckError):
        binary_search(["a"], 1)  # should raise type error


def test_hashmap_search():
    """
    Tests the hashmap search function
    We only test for existence, since
    that's the login checker's goal.
    """
    assert hash_search(generate_hash_table(["aa"]), "aa")  # should find
    assert not hash_search(generate_hash_table([]), "aa")  # should not find
    assert not hash_search(generate_hash_table(["aa"]), "")  # should not find
    assert hash_search(generate_hash_table(["a", "b", "c"]), "b")  # should find
    assert not hash_search(generate_hash_table(["a", "b", "c"]), "d")  # should not find
    assert hash_search(generate_hash_table(["d", "c", "b", "a"]), "c")  # should find
    with pytest.raises(TypeCheckError):
        hash_search(generate_hash_table(1), "a")  # should raise type error
    with pytest.raises(TypeCheckError):
        hash_search(generate_hash_table(["a"]), 1)  # should raise type error


def test_bloom_search():
    """
    Tests the binary search function
    We only test for existence, since
    that's the login checker's goal.
    """
    filter = BloomFilter()
    filter.add("aa")
    assert filter.get("aa")  # should find
    # assert not hash_search(generate_hash_table([]), "aa")  # should not find
    # assert not hash_search(generate_hash_table(["aa"]), "")  # should not find
    # assert hash_search(generate_hash_table(["a", "b", "c"]), "b")  # should find
    # assert not hash_search(generate_hash_table(["a", "b", "c"]), "d")  # should not find
    # assert hash_search(generate_hash_table(["d", "c", "b", "a"]), "c")  # should find
    # with pytest.raises(TypeCheckError):
    #     hash_search(generate_hash_table(1), "a")  # should raise type error
    # with pytest.raises(TypeCheckError):
    #     hash_search(generate_hash_table(["a"]), 1)  # should raise type error
