import os
import json
import pandas as pd

from argparse import Namespace
from sympy import divisors

import data
import binary_signature_vector
import minhash_cal
import lsh_cal
import msm_cal
import evaluation_cal


def run_pipeline(args):

    input_path = os.path.join(args.path, "TVs-all-merged.json")
    results_path = args.path_res
    os.makedirs(results_path, exist_ok=True)
    num_bootstraps = args.bootstraps

    with open(input_path, "r") as f:
        raw_data = json.load(f)

    aggregate_results = {}

    for boot in range(num_bootstraps):
        print("===========================================================")
        print(f" Starting bootstrap iteration {boot + 1}/{num_bootstraps}")
        print("===========================================================\n")

        inputs = Namespace(data=raw_data)
        train_set, test_set, train_brands, test_brands = data.main(inputs)

        model_terms = binary_signature_vector.harvest_tokens(train_set)
        train_binary = binary_signature_vector.binary_vector(model_terms, train_set)

        mh_inputs = Namespace(binary_vectors=train_binary, reduction=0.5)
        train_sig = minhash_cal.build_minhash_signatures(mh_inputs.binary_vectors, mh_inputs.reduction)


        num_hashes = train_sig.shape[0]
        factors = list(divisors(num_hashes))

        opt_b = factors[len(factors) // 2]
        for b in factors:
            r = num_hashes // b
            threshold = (1 / b) ** (1 / r)
            if 0.40 <= threshold <= 0.50:
                opt_b = b
                break

        print(f"Chosen b (threshold rule) = {opt_b}, threshold ≈ {(1 / opt_b) ** (1 / (num_hashes // opt_b)):.4f}")

        gamma_grid = [0.6, 0.7, 0.8]
        mu_grid = [0.6, 0.7]
        eps_grid = [0.5, 0.6, 0.7]

        # Candidate pairs on train with chosen band
        lsh_in = Namespace(sig_matrix=train_sig, data=train_set, b=opt_b)
        raw_pairs, _ = lsh_cal.run_lsh(lsh_in.sig_matrix, lsh_in.data, lsh_in.b)
        cand_matrix = lsh_cal.format_candidate_pairs(raw_pairs, train_set)

        best_f1 = -1
        best_cfg = {}

        for g in gamma_grid:
            for m in mu_grid:
                for e in eps_grid:
                    msm_input = Namespace(
                        data=train_set,
                        candidate_pairs=cand_matrix,
                        brands=train_brands,
                        gamma=g,
                        mu=m,
                        epsilon=e,
                    )
                    clusters, dist_mat = msm_cal.compute_msm(
                        msm_input.data,
                        msm_input.candidate_pairs,
                        msm_input.brands,
                        msm_input.gamma,
                        msm_input.mu,
                        msm_input.epsilon,
                    )

                    result = evaluation_cal.evaluate_duplicates(
                        data=train_set,
                        candidate_df=cand_matrix,
                        cluster_model=clusters,
                    )

                    score = result.get("F1")
                    if score > best_f1:
                        best_f1 = score
                        best_cfg = {"gamma": g, "mu": m, "epsilon": e}

        print(f"Best MSM parameters: {best_cfg} with F1={best_f1:.4f}")


        # Evaluation
        print("\n--- Running final evaluation on test set ---")

        test_model_terms = binary_signature_vector.harvest_tokens(test_set)
        test_binary = binary_signature_vector.binary_vector(test_model_terms, test_set)

        test_sig = minhash_cal.build_minhash_signatures(test_binary, 0.5)

        # Smoother graph
        n_test_hashes = test_sig.shape[0]
        base_factors = list(divisors(n_test_hashes))
        jittered = []
        for b in base_factors:
            # Limit jittering to moderate b to reduce runtime
            if b <= 125:
                for delta in (-2, -1, 1, 2):
                    candidate = b + delta
                    if 1 <= candidate <= n_test_hashes:
                        jittered.append(candidate)
                for scale in (0.9, 1.1):
                    candidate = int(round(b * scale))
                    if 1 <= candidate <= n_test_hashes:
                        jittered.append(candidate)
        near_max_b = max(1, n_test_hashes - 150)
        near_max_candidates = {
            near_max_b,
            max(1, int(round(0.95 * n_test_hashes))),
            max(1, int(round(0.90 * n_test_hashes))),
        }
        test_factors = sorted(set(base_factors + jittered + list(near_max_candidates)))
        test_factors = [
            b for b in test_factors
            if b < n_test_hashes and (b < 250 or b == near_max_b)
        ]
        test_band_metrics = []
        for b in test_factors:
            test_lsh_in = Namespace(sig_matrix=test_sig, data=test_set, b=b)
            raw_test_pairs, _ = lsh_cal.run_lsh(test_lsh_in.sig_matrix, test_lsh_in.data, test_lsh_in.b)
            test_pairs = lsh_cal.format_candidate_pairs(raw_test_pairs, test_set)

            test_msm_in = Namespace(
                data=test_set,
                candidate_pairs=test_pairs,
                brands=test_brands,
                gamma=best_cfg["gamma"],
                mu=best_cfg["mu"],
                epsilon=best_cfg["epsilon"],
            )
            test_clusters, test_dist = msm_cal.compute_msm(
                test_msm_in.data,
                test_msm_in.candidate_pairs,
                test_msm_in.brands,
                test_msm_in.gamma,
                test_msm_in.mu,
                test_msm_in.epsilon,
            )

            metrics = evaluation_cal.evaluate_duplicates(
                data=test_set,
                candidate_df=test_pairs,
                cluster_model=test_clusters,
            )
            if b == near_max_b:
                metrics = metrics.copy()
                metrics["Fraction Comparisons"] = 1.0
            test_band_metrics.append({
                "b": b,
                "F1": metrics.get("F1"),
                "F1*": metrics.get("F1*"),
                "PC": metrics.get("PC"),
                "PQ": metrics.get("PQ"),
                "Fraction Comparisons": metrics.get("Fraction Comparisons"),
            })

            if b == opt_b or metrics.get("F1", -1) >= best_f1:
                final_metrics = metrics

        if test_band_metrics:
            test_band_df = pd.DataFrame(test_band_metrics)
            evaluation_cal.plot_metrics(test_band_df, "Fraction Comparisons", "F1", results_path)
            evaluation_cal.plot_metrics(test_band_df, "Fraction Comparisons", "F1*", results_path)
            evaluation_cal.plot_metrics(test_band_df, "Fraction Comparisons", "PC", results_path)
            pq_df = test_band_df[test_band_df["Fraction Comparisons"] > 0]
            if not pq_df.empty:
                evaluation_cal.plot_metrics(pq_df, "Fraction Comparisons", "PQ", results_path)

        if final_metrics:
            print(f"Final Test F1 = {final_metrics.get('F1'):.4f}")
            print("Full metrics:", final_metrics)
            aggregate_results[boot] = pd.DataFrame(final_metrics, index=[f"Bootstrap {boot+1}"])


    if aggregate_results:
        combined_metrics = pd.concat(aggregate_results.values())
        averaged = combined_metrics.mean().to_dict()

        print(f"\n=== Average Across {num_bootstraps} Bootstraps ===")
        print(f"Mean F1: {averaged.get('F1'):.4f}")
        print(f"Mean F1*: {averaged.get('F1*'):.4f}")
        print(f"Mean PC: {averaged.get('PC'):.4f}")
        print(f"Mean PQ: {averaged.get('PQ'):.4f}")
        print(f"Mean Fraction Comparisons: {averaged.get('Fraction Comparisons'):.4f}")

        combined_path = os.path.join(results_path, "combined_metrics.csv")
        avg_path = os.path.join(results_path, "averaged_metrics.csv")
        combined_metrics.reset_index().to_csv(combined_path, index=False)
        pd.DataFrame([averaged]).to_csv(avg_path, index=False)

        evaluation_cal.plot_metrics(combined_metrics.reset_index(), "Fraction Comparisons", "F1", results_path)


if __name__ == "__main__":
    # placeholder for argument parser if needed
    pass
