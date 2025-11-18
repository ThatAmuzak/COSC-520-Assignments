# test_algorithms.py

import pytest
from src.algorithms import boyer_moore, kmp_search, rabin_karp
from typeguard import TypeCheckError


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_empty_pattern(func):
    text = "abc"
    assert func("", text) == [0, 1, 2, 3]
    print(f"{func.__name__}: test_empty_pattern passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_empty_text(func):
    pattern = "a"
    assert func(pattern, "") == []
    print(f"{func.__name__}: test_empty_text passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_both_empty(func):
    assert func("", "") == [0]
    print(f"{func.__name__}: test_both_empty passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_no_match(func):
    assert func("x", "abc") == []
    print(f"{func.__name__}: test_no_match passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_single_match(func):
    assert func("a", "abc") == [0]
    assert func("c", "abc") == [2]
    print(f"{func.__name__}: test_single_match passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_multiple_matches(func):
    text = "ababa"
    pattern = "aba"
    assert func(pattern, text) == [0, 2]
    print(f"{func.__name__}: test_multiple_matches passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_pattern_longer_than_text(func):
    assert func("abcdef", "abc") == []
    print(f"{func.__name__}: test_pattern_longer_than_text passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_full_text_match(func):
    text = "hello"
    pattern = "hello"
    assert func(pattern, text) == [0]
    print(f"{func.__name__}: test_full_text_match passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_overlapping_matches(func):
    text = "aaaa"
    pattern = "aa"
    assert func(pattern, text) == [0, 1, 2]
    print(f"{func.__name__}: test_overlapping_matches passed")


@pytest.mark.parametrize("func", [kmp_search, rabin_karp, boyer_moore])
def test_type_errors(func):
    import pytest

    with pytest.raises(TypeCheckError):
        func(123, "abc")
    with pytest.raises(TypeCheckError):
        func("abc", 456)
    with pytest.raises(TypeCheckError):
        func(None, None)
    print(f"{func.__name__}: test_type_errors passed")
