# from src.algorithms import lin_search, bin_search, hash_table_search, bloom_filter_search, cuckoo_filter_search,

from tqdm import tqdm


def generate_unique_usernames(n, prefix="user"):
    """
    generates unique logins dataset
    not really representative of actual usernames
    but good enough for testing purposes
    """
    return [f"{prefix}{i}" for i in tqdm(range(n), desc="Generating usernames")]


if __name__ == "__main__":
    logins = generate_unique_usernames(10000000)
    print("generated logins")
    # TODO: profile the algorithms, and document their performance for different dataset sizes
