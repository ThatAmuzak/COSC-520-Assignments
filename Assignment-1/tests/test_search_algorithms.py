import pytest
from src.search_algorithms import linear_search
from typeguard import TypeCheckError


def test_linear_search():
    """
    Tests the linear search function
    We only test for existence, since
    that's the login checker's goal.
    -1 indicates the item was not found
    """
    assert linear_search(["aa"], "aa") != -1  # should find
    assert linear_search([], "aa") == -1  # should not find
    assert linear_search(["aa"], "") == -1  # should not find
    assert linear_search(["a", "b", "c"], "b") != -1  # should find
    assert linear_search(["a", "b", "c"], "d") == -1  # should not find
    with pytest.raises(TypeCheckError):
        linear_search(1, 2) == 1  # should raise type error
