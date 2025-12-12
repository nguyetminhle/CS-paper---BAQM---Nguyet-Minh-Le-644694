import argparse
import re
from collections import OrderedDict
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

# brand noise
ALT_BRAND_MAP = {
    "lg electronics": "lg",
    "jvc tv": "jvc",
    "sceptre inc.": "sceptre",
    "pansonic": "panasonic",
}

def standardize_brand(input_brand: str) -> str:
    if not input_brand:
        return ""
    cleaned = input_brand.lower().strip()
    return ALT_BRAND_MAP.get(cleaned, cleaned)


def regex_replace(map_dict, text):
    compiled_rules = {
        re.compile(rf"{pattern}", re.IGNORECASE): replacement
        for replacement, patterns in map_dict.items()
        for pattern in patterns
    }
    for patt, repl in compiled_rules.items():
        text = patt.sub(repl, text)
    return text


def process_dataset(raw_block: Dict[str, Any]) -> Tuple[OrderedDict, OrderedDict]:
    revised_multi = OrderedDict()
    revised_single = OrderedDict()

    mw_clean_rules = {
        "inch ": ["Inch", "inches", "\"", "'", "”", "-inch", " inch", "inch"],
        "hz ": ["Hertz", "hertz", "Hz", "HZ", " hz", "-hz", "hz"],
        "lbs ": ["lb", "lb.", " lbs", "lbs", "pound", "pounds"], # adding weights
    }

    title_clean_rules = mw_clean_rules

    for identifier in raw_block:
        items = raw_block.get(identifier)
        count = 0

        for entry in items:
            # extract shop
            shop_val = entry.get("shop") or entry.get("Shop")
            if shop_val:
                entry["shop"] = shop_val.lower().strip()

            # extract brand
            spec_map = entry.get("featuresMap", {})
            detected_brand = None
            for key_option in ("Brand", "brand", "Brand Name", "brand name"):
                if key_option in spec_map and spec_map.get(key_option):
                    detected_brand = spec_map[key_option]
                    break

            if detected_brand is not None:
                spec_map["Brand"] = standardize_brand(detected_brand)

            # title
            entry["title"] = regex_replace(
                title_clean_rules,
                (entry.get("title") or "").lower()
            )

            # feature
            for spec, val in spec_map.items():
                spec_map[spec] = regex_replace(mw_clean_rules, (val or "").lower())

            entry["featuresMap"] = spec_map
            count += 1

        if count > 1:
            revised_multi[identifier] = items
        else:
            revised_single[identifier] = items

    return revised_multi, revised_single


def split_sets(multi_dict, single_dict):
    multi_keys = list(multi_dict.keys())
    single_keys = list(single_dict.keys())

    m_train, m_test = train_test_split(multi_keys, train_size=0.67)
    s_train, s_test = train_test_split(single_keys, train_size=0.67)

    multi_train = {k: multi_dict[k] for k in m_train}
    multi_test = {k: multi_dict[k] for k in m_test}
    single_train = {k: single_dict[k] for k in s_train}
    single_test = {k: single_dict[k] for k in s_test}

    train_bundle = unfold_entries(multi_train)
    train_bundle.update(unfold_entries(single_train))

    test_bundle = unfold_entries(multi_test)
    test_bundle.update(unfold_entries(single_test))

    return train_bundle, test_bundle


def unfold_entries(block):
    flat = OrderedDict()
    for tag in block:
        grp = block.get(tag)
        for item in grp:
            src = item.get("shop")
            flat[f"{tag}_{src}"] = item
    return flat


def extract_brand_list(dataset):
    collected = OrderedDict()
    for key in dataset:
        mark = dataset.get(key).get("featuresMap", {}).get("Brand")
        if mark is not None:
            collected[mark.lower()] = ""
    return collected


def main(args):
    data_arg: dict = args.data

    print("Clean the data...\n")

    cleaned_dup, cleaned_single = process_dataset(data_arg)
    train, test = split_sets(cleaned_dup, cleaned_single)
    brands_train, brands_test = extract_brand_list(train), extract_brand_list(test)

    print("Data is cleaned.\n")

    return train, test, brands_train, brands_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=dict)
    args = parser.parse_args()
    main(args)
