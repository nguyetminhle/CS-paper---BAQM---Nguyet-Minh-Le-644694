import re
import numpy as np

from collections import OrderedDict
from ordered_set import OrderedSet


def harvest_tokens(data):
    title_token_pattern = r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)'
    feature_num_pattern = r'(ˆ\d+(\.\d+)?[a-zA-Z]+$|ˆ\d+(\.\d+)?$)'

    vocab = OrderedSet()

    for pid in data:
        entry = data.get(pid, {})
        title_txt = (entry.get("title") or "")

        for match in re.findall(title_token_pattern, title_txt):
            vocab.add(match[0])

        feats = entry.get("featuresMap") or {}
        for val in feats.values():
            if not val:
                continue

            for m in re.findall(feature_num_pattern, val):
                vocab.add(m)
    return list(vocab)


def binary_vector(vocab_list, records):

    print("Generating binary signature vectors...")

    embeddings = OrderedDict()
    total_items = len(records)
    total_features = len(vocab_list)

    progress = 0

    for rid in records:
        vec = np.zeros(total_features, dtype=int)
        entry = records.get(rid, {})

        title_txt = entry.get("title") or ""
        feats = entry.get("featuresMap") or {}

        # Same substring presence logic as reference
        for idx, token in enumerate(vocab_list):
            if not token:
                continue

            if token in title_txt:
                vec[idx] = 1
            else:
                for v in feats.values():
                    if v and token in v:
                        vec[idx] = 1
                        break

        embeddings[rid] = vec

        if progress % 500 == 0:
            print(f"Processed {progress}/{total_items} entries")

        progress += 1

    print("...Binary signature generation complete.\n")
    return embeddings


def main(args):
    catalogue: OrderedDict = args.data

    model_word = harvest_tokens(catalogue)
    vectors = binary_vector(model_word, catalogue)

    return vectors, model_word


if __name__ == "__main__":
    pass
