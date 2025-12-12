import numpy as np

from primePy import primes
from collections import OrderedDict


def build_minhash_signatures(bit_vectors: OrderedDict, reduction: float):

    print("Starting MinHash signature generation...")

    keys = list(bit_vectors.keys())
    total_items = len(keys)
    if total_items == 0:
        return np.array([[]])

    vec_length = len(bit_vectors[keys[0]])

    num_hashes = int(round(reduction * vec_length / 500) * 500)
    if num_hashes <= 0:
        num_hashes = 500

    print(f"Using {num_hashes} hash functions.")

    sig = np.full((num_hashes, total_items), np.inf)

    primes_list = primes.between(num_hashes + 1, num_hashes + 100000)
    p = primes_list[np.random.randint(len(primes_list))]

    a = np.random.randint(0, p, size=num_hashes)
    b = np.random.randint(0, p, size=num_hashes)

    for row_idx in range(vec_length):

        hvals = (a + b * row_idx) % p

        for col_idx in range(total_items):
            if bit_vectors[keys[col_idx]][row_idx] == 1:
                sig[:, col_idx] = np.minimum(sig[:, col_idx], hvals)

    print("MinHash signature generation done.\n")
    return sig

def main(args):
    bit_vectors: OrderedDict = args.binary_vectors
    ratio: float = args.reduction
    return build_minhash_signatures(bit_vectors, ratio)


if __name__ == "__main__":
    pass
