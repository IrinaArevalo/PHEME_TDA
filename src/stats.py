import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu # nonparametric test of the distributions are the same
from statsmodels.stats.multitest import multipletests # test results and p-value correction for multiple tests, including FDR


def summarize_group(x):
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


def compare_groups(df, label_col, feature_cols, group_a="rumour", group_b="non-rumour"):
    results = []

    for feat in feature_cols:
        x = df.loc[df[label_col] == group_a, feat].dropna().to_numpy()
        y = df.loc[df[label_col] == group_b, feat].dropna().to_numpy()

        if len(x) == 0 or len(y) == 0:
            continue

        u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")

        sx = summarize_group(x)
        sy = summarize_group(y)

        results.append({
            "feature": feat,
            f"{group_a}_n": sx["n"],
            f"{group_b}_n": sy["n"],
            f"{group_a}_mean": sx["mean"],
            f"{group_b}_mean": sy["mean"],
            f"{group_a}_median": sx["median"],
            f"{group_b}_median": sy["median"],
            "u_stat": float(u_stat),
            "p_value": float(p_value),
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        reject, p_adj, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
        results_df["p_value_fdr"] = p_adj
        results_df["significant_fdr_0.05"] = reject

    return results_df.sort_values("p_value")


def compare_groups_by_event(
    df,
    event_col,
    label_col,
    feature_cols,
    group_a="rumour",
    group_b="non-rumour",
    adjust_within_event=True,
):
    all_results = []

    for event_name, df_event in df.groupby(event_col):
        event_results = []

        for feat in feature_cols:
            x = df_event.loc[df_event[label_col] == group_a, feat].dropna().to_numpy()
            y = df_event.loc[df_event[label_col] == group_b, feat].dropna().to_numpy()

            if len(x) == 0 or len(y) == 0:
                continue

            u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")

            sx = summarize_group(x)
            sy = summarize_group(y)

            event_results.append({
                "event": event_name,
                "feature": feat,
                f"{group_a}_n": sx["n"],
                f"{group_b}_n": sy["n"],
                f"{group_a}_mean": sx["mean"],
                f"{group_b}_mean": sy["mean"],
                f"{group_a}_median": sx["median"],
                f"{group_b}_median": sy["median"],
                "u_stat": float(u_stat),
                "p_value": float(p_value)
            })

        event_results_df = pd.DataFrame(event_results)

        if not event_results_df.empty and adjust_within_event:
            reject, p_adj, _, _ = multipletests(event_results_df["p_value"], method="fdr_bh")
            event_results_df["p_value_fdr"] = p_adj
            event_results_df["significant_fdr_0.05"] = reject

        all_results.append(event_results_df)

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)