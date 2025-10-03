import pytest
from src.search_algorithms import binary_search, linear_search
from typeguard import TypeCheckError


def test_linear_search():
    """
    Tests the linear search function
    We only test for existence, since
    that's the login checker's goal.
    -1 indicates the item was not found
    """
    assert linear_search(["aa"], "aa")  # should find
    assert not linear_search([], "aa")  # should not find
    assert not linear_search(["aa"], "")  # should not find
    assert linear_search(["a", "b", "c"], "b")  # should find
    assert linear_search(["c", "b", "a"], "b")  # should find
    assert not linear_search(["a", "b", "c"], "d")  # should not find
    with pytest.raises(TypeCheckError):
        linear_search(1, 2) == 1  # should raise type error


def test_binary_search():
    """
    Tests the binary search function
    We only test for existence, since
    that's the login checker's goal.
    -1 indicates the item was not found
    """
    assert binary_search(["aa"], "aa")  # should find
    assert not binary_search([], "aa")  # should not find
    assert binary_search(["a", "b", "c"], "b")  # should find
