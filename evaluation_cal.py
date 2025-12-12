import argparse
import os
import numpy as np
import pandas as pd

_mpl_cfg_dir = os.path.join(os.getcwd(), ".matplotlib_cache")
os.makedirs(_mpl_cfg_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cfg_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import defaultdict, OrderedDict
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, Any, Tuple, Set


def extract_cluster_pairs(model: AgglomerativeClustering) -> Set[Tuple[int, int]]:
    predicted = set()
    lbls = model.labels_

    for c_id in range(model.n_clusters_):
        members = np.where(lbls == c_id)[0]
        if len(members) > 1:
            predicted.update(combinations(members, 2))

    return predicted

def evaluate_duplicates(
    data: OrderedDict,
    candidate_df: pd.DataFrame,
    cluster_model: AgglomerativeClustering
) -> Dict[str, float]:

    products = list(data.keys())
    total_possible = len(products) * (len(products) - 1) / 2

    grouped = defaultdict(list)
    for pid in products:
        grouped[data[pid]["modelID"]].append(pid)

    truth_pairs = set()
    for p_list in grouped.values():
        if len(p_list) > 1:
            idxs = sorted(products.index(p) for p in p_list)
            truth_pairs.update(combinations(idxs, 2))

    predicted_pairs = extract_cluster_pairs(cluster_model)

    lsh_candidates = set()
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            if candidate_df.iat[i, j] == 0:
                lsh_candidates.add((i, j))

    TP = len(truth_pairs & predicted_pairs)
    FP = len(predicted_pairs - truth_pairs)
    FN = len(truth_pairs - predicted_pairs)

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    tp_lsh = len(truth_pairs & lsh_candidates)
    pc = tp_lsh / len(truth_pairs) if truth_pairs else 0.0
    pq = tp_lsh / len(lsh_candidates) if lsh_candidates else 0.0
    f1_star = (2 * pc * pq / (pc + pq)) if (pc + pq) else 0.0

    frac_comp = len(lsh_candidates) / total_possible if total_possible else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PC": pc,
        "PQ": pq,
        "F1*": f1_star,
        "Fraction Comparisons": frac_comp,
        "True Duplicates": len(truth_pairs),
        "Predicted Duplicates": len(predicted_pairs),
        "LSH Candidates": len(lsh_candidates),
    }


def plot_metrics(df: pd.DataFrame, x_metric: str, y_metric: str, save_dir: str):
    to_plot = df[[x_metric, y_metric]].dropna().sort_values(x_metric)

    plt.figure(figsize=(8, 6))
    plt.plot(
        to_plot[x_metric],
        to_plot[y_metric],
        marker=None,
        linestyle="-",
        color="black",
        linewidth=1.6,
        alpha=0.9,
    )

    title = f"{y_metric} vs {x_metric}"
    plt.title(title)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.xlim(0.0, 1.0)
    plt.tight_layout()

    safe_name = title.replace("*", "star").replace(" ", "_")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{safe_name}.png")
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    pass
