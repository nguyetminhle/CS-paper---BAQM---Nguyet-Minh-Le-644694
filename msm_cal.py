import pandas as pd
import numpy as np
import re

from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering
from ordered_set import OrderedSet
from sim_cal import qgram_jaccard, avg_levenshtein_sim, separate_digits, cosine_words, normalized_lev

def get_shop(product: dict) -> str | None:
    for key, val in product.items():
        if "shop" in key.lower() and isinstance(val, str):
            return val.strip().lower()
    return None


def same_shop(item_a: dict, item_b: dict) -> bool:
    shop_a = get_shop(item_a)
    shop_b = get_shop(item_b)
    return shop_a == shop_b and shop_a is not None


def brand_match(item_a: dict, item_b: dict, brand_dict: OrderedDict) -> bool:
    ALT_BRAND_MAP = {
        "lg electronics": "lg",
        "jvc tv": "jvc",
        "sceptre inc.": "sceptre",
        "pansonic": "panasonic",
    }

    def _normalized_brand(item: dict) -> str | None:
        feats = item.get("featuresMap") or {}
        for key, val in feats.items():
            if "brand" in key.lower() and isinstance(val, str) and val.strip():
                cleaned = val.strip().lower()
                return ALT_BRAND_MAP.get(cleaned, cleaned)
        return None

    brand_a = _normalized_brand(item_a)
    brand_b = _normalized_brand(item_b)
    if brand_a and brand_b:
        return brand_a == brand_b
    return True

RESOLUTION_1080_PATTERNS = ["1080", "full hd", "fhd"]
RESOLUTION_4K_PATTERNS = ["4k", "uhd", "ultra hd", "2160", "3840"]
NUMERIC_EXTRACT_PATTERN = re.compile(r"\d+(?:\.\d+)?")
TITLE_MODEL_REGEX = re.compile(r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)')
MODEL_WORD_REGEX = re.compile(r'(ˆ\d+(\.\d+)?[a-zA-Z]+$|ˆ\d+(\.\d+)?$)')


def _normalize_resolution(value: str) -> str | None:
    if not isinstance(value, str):
        return None

    val = value.lower()
    if any(pat in val for pat in RESOLUTION_4K_PATTERNS):
        return "4k"
    if any(pat in val for pat in RESOLUTION_1080_PATTERNS):
        return "1080p"
    return None


def get_resolution(product: dict) -> str | None:
    feats = product.get("featuresMap") or {}
    for key, val in feats.items():
        if "resolution" in key.lower():
            normalized = _normalize_resolution(val)
            if normalized:
                return normalized
    return None


def _extract_first_number(val: str) -> float | None:
    if not isinstance(val, str):
        return None
    match = NUMERIC_EXTRACT_PATTERN.search(val)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def get_size_value(product: dict) -> float | None:
    feats = product.get("featuresMap") or {}
    for key, val in feats.items():
        if "size" in key.lower():
            num = _extract_first_number(val)
            if num is not None:
                return num
    return None


def collect_model_words(product: dict, remaining_feature_keys: list[str], brands: OrderedDict) -> OrderedSet:
    mw = OrderedSet()

    feats = product.get("featuresMap") or {}
    title = product.get("title") or ""

    for k in remaining_feature_keys:
        val = feats.get(k, "")
        if val:
            mw.update(MODEL_WORD_REGEX.findall(val))

    mw.update(x[0] for x in TITLE_MODEL_REGEX.findall(title))

    brand_val = None
    for key, val in feats.items():
        if "brand" in key.lower() and isinstance(val, str) and val.strip():
            brand_val = val.strip().lower()
            break

    title_lower = title.lower()
    for brand in brands or []:
        if brand in title_lower:
            mw.add(brand)
    if brand_val:
        mw.add(brand_val)

    return mw


def title_comp(title_1: str, title_2: str, alpha: float, beta: float, delta: float, approx: float) -> float:

    name_cosine_sim = cosine_words(title_1, title_2)
    if name_cosine_sim > alpha:
        return 1.0

    model_words_1 = OrderedSet(x[0] for x in TITLE_MODEL_REGEX.findall(title_1))
    model_words_2 = OrderedSet(x[0] for x in TITLE_MODEL_REGEX.findall(title_2))

    similar_model_words = False
    for word_1 in model_words_1:
        non_numeric_1, numeric_1 = separate_digits(word_1)
        for word_2 in model_words_2:
            non_numeric_2, numeric_2 = separate_digits(word_2)

            approx_sim = normalized_lev(non_numeric_1, non_numeric_2)
            if approx_sim > approx and numeric_1 != numeric_2:
                return -1
            elif approx_sim > approx and numeric_1 == numeric_2:
                similar_model_words = True

    final_name_sim = beta * name_cosine_sim + (1 - beta) * avg_levenshtein_sim(list(model_words_1), list(model_words_2),
                                                                               False)
    if similar_model_words:
        final_name_sim = delta * avg_levenshtein_sim(list(model_words_1), list(model_words_2), True) + (
                    1 - delta) * final_name_sim

    return final_name_sim


def compute_msm(
    items: OrderedDict,
    candidate_mask: pd.DataFrame,
    brand_list: OrderedDict,
    gamma: float,
    mu: float,
    epsilon: float
):


    product_ids = list(items.keys())

    dissim = pd.DataFrame(
        np.ones((len(product_ids), len(product_ids))),
        index=product_ids,
        columns=product_ids,
    )

    print(f"Starting MSM computation with gamma={gamma}, mu={mu}, epsilon={epsilon}...")

    for p1, p2 in candidate_mask.stack().index:
        if candidate_mask.loc[p1, p2] != 0:
            continue

        prod_1 = items[p1]
        prod_2 = items[p2]

        if same_shop(prod_1, prod_2):
            dissim.loc[p1, p2] = 1.0
            dissim.loc[p2, p1] = 1.0
            continue

        if not brand_match(prod_1, prod_2, brand_list):
            dissim.loc[p1, p2] = 1.0
            dissim.loc[p2, p1] = 1.0
            continue

        res_1 = get_resolution(prod_1)
        res_2 = get_resolution(prod_2)
        if res_1 and res_2 and res_1 != res_2:
            dissim.loc[p1, p2] = 1.0
            dissim.loc[p2, p1] = 1.0
            continue

        size_1 = get_size_value(prod_1)
        size_2 = get_size_value(prod_2)
        if size_1 is not None and size_2 is not None:
            if abs(size_1 - size_2) > 1.0:
                dissim.loc[p1, p2] = 1.0
                dissim.loc[p2, p1] = 1.0
                continue

        feats_1 = prod_1["featuresMap"]
        feats_2 = prod_2["featuresMap"]

        no_match_1 = list(feats_1.keys())
        no_match_2 = list(feats_2.keys())

        sim_acc = 0.0
        weight_acc = 0.0
        m = 0

        for k1 in feats_1.keys():

            for k2 in list(no_match_2):

                key_sim = qgram_jaccard(k1, k2, q=3)
                if key_sim > gamma:

                    val_sim = qgram_jaccard(feats_1[k1], feats_2[k2], q=3)

                    sim_acc += key_sim * val_sim
                    weight_acc += key_sim
                    m += 1

                    no_match_1.remove(k1)
                    no_match_2.remove(k2)
                    break

        feature_mean_sim = sim_acc / weight_acc if weight_acc > 0 else 0.0

        mw1 = collect_model_words(prod_1, no_match_1, brand_list)
        mw2 = collect_model_words(prod_2, no_match_2, brand_list)

        union_len = len(mw1.union(mw2))
        mw_percentage = 0 if union_len == 0 else len(mw1.intersection(mw2)) / union_len

        title_sim = title_comp(
            prod_1.get("title"),
            prod_2.get("title"),
            alpha=0.5, beta=0.0, delta=0.5, approx=0.5
        )

        min_feats = min(len(feats_1), len(feats_2))

        if title_sim == -1:
            theta_1 = m / min_feats if min_feats else 0
            theta_2 = 1 - theta_1
            h_sim = theta_1 * feature_mean_sim + theta_2 * mw_percentage

        else:
            theta_1 = (1 - mu) * (m / min_feats if min_feats else 0)
            theta_2 = 1 - mu - theta_1
            h_sim = (
                theta_1 * feature_mean_sim +
                theta_2 * mw_percentage +
                mu * title_sim
            )

        h_sim = np.clip(h_sim, 0.0, 1.0)

        dissim.loc[p1, p2] = 1 - h_sim
        dissim.loc[p2, p1] = 1 - h_sim

    np.fill_diagonal(dissim.values, 0.0)

    print("MSM completed. Running clustering...")

    try:
        from sim_cal import clustering
        cluster_model = clustering(dissimilarity_matrix=dissim, threshold=epsilon)
    except:
        cluster_model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=epsilon
        )
        cluster_model.fit(dissim.values)

    return cluster_model, dissim


def main(args):
    data: OrderedDict = args.data
    candidate_pairs: pd.DataFrame = args.candidate_pairs
    brands: OrderedDict = args.brands
    gamma: float = args.gamma
    mu: float = args.mu
    epsilon: float = args.epsilon

    return compute_msm(data, candidate_pairs, brands, gamma, mu, epsilon)


if __name__ == "__main__":
    pass