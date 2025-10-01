def linear_search(arr, target):
    """
    Basic linear searching
    Time complexity: O(n)
    Space complexity: O(1)
    """
    for index, value in enumerate(arr):
        if value == target:
            return index
    return -1
