# import search_algorithms

from tqdm import tqdm


def generate_unique_usernames(n, prefix="user"):
    """
    generates unique logins dataset
    not really representative of actual usernames
    but good enough for testing purposes
    """
    return [f"{prefix}{i}" for i in tqdm(range(n), desc="Generating usernames")]


if __name__ == "__main__":
    logins = generate_unique_usernames(1000000)
    print("generated logins")
    # TODO: profile the algorithms, and document their performance for different dataset sizes
