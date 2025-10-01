def generate_hash_table(strings):
    """
    Generates hash table using dicts
    for subsequent searches
    """
    table = {}
    for idx, string in enumerate(strings):
        table[string] = idx
    return table


def hash_search(table, target):
    """
    Searches for a string in the hash table and returns its index
    Assumes hash table has been generated beforehand with :py:func:`hash_table_search.generate_hash_table`.
    Time Complexity: O(1)
    Space Complexity: O(n)
    """
    return table.get(target, -1)
