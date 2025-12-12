import argparse
import pandas as pd
import numpy as np
import mmh3
import sys

from collections import defaultdict, OrderedDict


def run_lsh(signatures: np.ndarray, items: OrderedDict, bands: int):

    bucket_map = defaultdict(list)

    num_hashes = signatures.shape[0]
    total_cols = signatures.shape[1]

    rows_per_band = num_hashes // bands

    if rows_per_band == 0:
        print(f"Error: Too many bands ({bands}) for {num_hashes} rows. rows_per_band must be > 0.", file=sys.stderr)
        return defaultdict(list), 0

    used_rows = bands * rows_per_band
    print(f"... (Using {used_rows} of {num_hashes} rows) ...")

    sim_threshold = (1 / bands) ** (1 / rows_per_band)
    print(
        f"LSH settings: bands={bands}, rows/band={rows_per_band} "
        f"→ expected similarity threshold ≈ {sim_threshold:.4f}"
    )

    item_keys = list(items.keys())

    for col_index in range(total_cols):
        for b_idx in range(bands):
            row_start = b_idx * rows_per_band
            band = signatures[row_start:row_start + rows_per_band, col_index]

            segment_hash = mmh3.hash(band.tobytes(), seed=10)

            bucket_map[segment_hash].append(item_keys[col_index])

    return bucket_map, sim_threshold


def format_candidate_pairs(bucket_map: defaultdict, items: OrderedDict):
    item_list = list(items.keys())

    matrix = pd.DataFrame(
        np.ones((len(item_list), len(item_list)), dtype=int),
        index=item_list,
        columns=item_list
    )

    for product in bucket_map.values():
        if len(product) > 1:
            for i in range(len(product)):
                for j in range(i + 1, len(product)):
                    a = product[i]
                    b = product[j]
                    matrix.loc[a, b] = 0
                    matrix.loc[b, a] = 0
    return matrix


def main(args):
    items: OrderedDict = args.data
    signatures: np.ndarray = args.sig_matrix
    bands: int = args.b

    raw_pairs, thresh = run_lsh(signatures=signatures, items=items, bands=bands)
    pair_matrix = format_candidate_pairs(bucket_map=raw_pairs, items=items)

    return pair_matrix, thresh


if __name__ == "__main__":
    pass

