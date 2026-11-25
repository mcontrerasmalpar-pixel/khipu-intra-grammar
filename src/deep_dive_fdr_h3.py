"""
deep_dive_fdr_h3.py — Deep-dive for khipu-intra-grammar:
  1) N_PERM = 1,000 for high-resolution p-values + Benjamini-Hochberg FDR (q < 0.05)
  2) Co-occurrence of H3a (color homology) and H3b (numeric R^2 homology): 2x2 Fisher test,
     Spearman correlation, and top-10 strongest tabular khipus
  3) Sensitivity to strict 100% clean decimal cords (clean_dec_frac == 1.0 vs < 1.0)
  4) Extended H4 invariance by cataloger, CREATED_BY, and top museums.
"""

import pathlib
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control
from sklearn.metrics import normalized_mutual_info_score

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from controls_h1_h2_h3 import DEFAULT_DB, RESULTS_DIR, load_pendants, circular_shifts


def bh_fdr_mask(p_series: pd.Series, alpha: float = 0.05) -> tuple[int, int, float]:
    """Return (n_sig_fdr, n_total, pct_sig_fdr) using Benjamini-Hochberg FDR."""
    valid = p_series.dropna().values.astype(float)
    if len(valid) == 0:
        return 0, 0, np.nan
    q = false_discovery_control(valid, method="bh")
    n_sig = int(np.sum(q < alpha))
    return n_sig, len(valid), round(100.0 * n_sig / len(valid), 1)


def col_hom_score(M: np.ndarray) -> float:
    G, m = M.shape
    if G < 2:
        return np.nan
    matches = 0
    total = 0
    for a in range(G):
        for b in range(a + 1, G):
            matches += int(np.sum(M[a] == M[b]))
            total += m
    return matches / total


def pos_r2(M: np.ndarray) -> float:
    tot_var = np.var(M)
    if tot_var <= 1e-12:
        return np.nan
    between = np.mean((M.mean(axis=0, keepdims=True) - M.mean()) ** 2)
    return float(between / tot_var)


def run_deep_dive(df: pd.DataFrame, n_perm: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for kid, g in df.groupby("KHIPU_ID", sort=False):
        n = len(g)
        if n < 10:
            continue

        colors = g["color_key"].fillna("__MISSING__").values
        clusters = g["CLUSTER_ID"].fillna(-1).values
        log_val = np.log1p(g["cord_val"].values.astype(float))
        clean_frac = float(g["is_clean_decimal"].mean())

        n_colors = len(set(colors) - {"__MISSING__"})
        n_clusters = len(set(clusters) - {-1})
        shifts = circular_shifts(n, n_perm, rng)

        # H1a circular
        h1_exc_c = h1_p_c = np.nan
        if n_colors >= 2 and n_clusters >= 2:
            obs = float(normalized_mutual_info_score(clusters, colors))
            null_c = [normalized_mutual_info_score(clusters, np.roll(colors, int(s))) for s in shifts]
            h1_exc_c = obs - float(np.mean(null_c))
            h1_p_c = float((1 + sum(v >= obs for v in null_c)) / (len(null_c) + 1))

        col_ch = (colors[:-1] != colors[1:])
        cl_ch = (clusters[:-1] != clusters[1:])

        def jump_circ(y, mask):
            if mask.sum() < 3 or (~mask).sum() < 3 or np.std(y) == 0:
                return np.nan, np.nan
            j = np.abs(np.diff(y))
            obs = float(j[mask].mean() - j[~mask].mean())
            cnt = 0
            for s in shifts:
                jp = np.abs(np.diff(np.roll(y, int(s))))
                if (jp[mask].mean() - jp[~mask].mean()) >= obs:
                    cnt += 1
            return obs, float((1 + cnt) / (len(shifts) + 1))

        h2a_diff, h2a_pc = jump_circ(log_val, col_ch)
        h2b_diff, h2b_pc = jump_circ(log_val, cl_ch)

        # 2x2 decoupled: pure color inside cluster
        same_cl = ~cl_ch
        pure_col_diff = pure_col_p = np.nan
        if (same_cl & col_ch).sum() >= 3 and (same_cl & (~col_ch)).sum() >= 3 and np.std(log_val) > 0:
            j = np.abs(np.diff(log_val))
            pure_col_diff = float(j[same_cl & col_ch].mean() - j[same_cl & (~col_ch)].mean())
            cl_idxs = [np.where(clusters == c)[0] for c in np.unique(clusters)]
            cnt = 0
            for _ in range(n_perm):
                yp = log_val.copy()
                for idxs in cl_idxs:
                    if len(idxs) > 1:
                        yp[idxs] = rng.permutation(yp[idxs])
                jp = np.abs(np.diff(yp))
                if (jp[same_cl & col_ch].mean() - jp[same_cl & (~col_ch)].mean()) >= pure_col_diff:
                    cnt += 1
            pure_col_p = float((1 + cnt) / (n_perm + 1))

        # 2x2 decoupled: pure cluster with same color
        same_col = ~col_ch
        pure_cl_diff = pure_cl_p = np.nan
        if (same_col & cl_ch).sum() >= 3 and (same_col & (~cl_ch)).sum() >= 3 and np.std(log_val) > 0:
            j = np.abs(np.diff(log_val))
            pure_cl_diff = float(j[same_col & cl_ch].mean() - j[same_col & (~cl_ch)].mean())
            col_idxs = [np.where(colors == c)[0] for c in np.unique(colors)]
            cnt = 0
            for _ in range(n_perm):
                yp = log_val.copy()
                for idxs in col_idxs:
                    if len(idxs) > 1:
                        yp[idxs] = rng.permutation(yp[idxs])
                jp = np.abs(np.diff(yp))
                if (jp[same_col & cl_ch].mean() - jp[same_col & (~cl_ch)].mean()) >= pure_cl_diff:
                    cnt += 1
            pure_cl_p = float((1 + cnt) / (n_perm + 1))

        # H3a & H3b (N_PERM = 1000)
        h3_col_match = h3_col_exc = h3_col_p = np.nan
        h3_val_r2 = h3_val_exc = h3_val_p = np.nan
        h3_m = h3_G = 0
        cl_sizes = g.groupby("CLUSTER_ID", sort=False).size()
        valid_sizes = cl_sizes[cl_sizes >= 3]
        if len(valid_sizes) >= 2:
            sc = valid_sizes.value_counts()
            modal_m = int(sc.index[0])
            if sc.iloc[0] >= 2:
                hom_cids = valid_sizes[valid_sizes == modal_m].index.tolist()
                mat_col = np.array([g[g["CLUSTER_ID"] == c]["color_key"].fillna("__MISSING__").values for c in hom_cids])
                mat_val = np.array([np.log1p(g[g["CLUSTER_ID"] == c]["cord_val"].values.astype(float)) for c in hom_cids])
                h3_G, h3_m = mat_col.shape

                if len(set(mat_col.ravel()) - {"__MISSING__"}) >= 2:
                    obs_cm = col_hom_score(mat_col)
                    null_cm = [col_hom_score(np.array([rng.permutation(mat_col[r]) for r in range(h3_G)])) for _ in range(n_perm)]
                    h3_col_match = float(obs_cm)
                    h3_col_exc = float(obs_cm - np.mean(null_cm))
                    h3_col_p = float((1 + sum(v >= obs_cm for v in null_cm)) / (n_perm + 1))

                obs_r2 = pos_r2(mat_val)
                if not np.isnan(obs_r2):
                    null_r2 = [pos_r2(np.array([rng.permutation(mat_val[r]) for r in range(h3_G)])) for _ in range(n_perm)]
                    h3_val_r2 = float(obs_r2)
                    h3_val_exc = float(obs_r2 - np.mean(null_r2))
                    h3_val_p = float((1 + sum(v >= obs_r2 for v in null_r2)) / (n_perm + 1))

        rows.append({
            "KHIPU_ID": kid, "OKR_NUM": g["OKR_NUM"].iloc[0],
            "cataloger": g["cataloger"].iloc[0],
            "created_by": g["CREATED_BY"].iloc[0],
            "museum": str(g["MUSEUM_NAME"].iloc[0]).strip().strip('"'),
            "n_pendants": n, "clean_dec_frac": clean_frac,
            "h1_exc_circ": h1_exc_c, "h1_p_circ": h1_p_c,
            "h2a_diff": h2a_diff, "h2a_p_circ": h2a_pc,
            "h2b_diff": h2b_diff, "h2b_p_circ": h2b_pc,
            "pure_col_diff": pure_col_diff, "pure_col_p": pure_col_p,
            "pure_cl_diff": pure_cl_diff, "pure_cl_p": pure_cl_p,
            "h3_G": h3_G, "h3_m": h3_m,
            "h3_col_match": h3_col_match, "h3_col_exc": h3_col_exc, "h3_col_p": h3_col_p,
            "h3_val_r2": h3_val_r2, "h3_val_exc": h3_val_exc, "h3_val_p": h3_val_p,
        })

    return pd.DataFrame(rows)


def main():
    df = load_pendants(DEFAULT_DB)
    res = run_deep_dive(df, n_perm=1000, seed=42)
    out = RESULTS_DIR / "deep_dive_fdr_1000.csv"
    res.to_csv(out, index=False)
    print(f"Saved {len(res)} khipus (N_PERM=1000) -> {out}\n")

    print("=== 1. BENJAMINI-HOCHBERG FDR (q < 0.05) AT N_PERM = 1,000 ===")
    tests = [
        ("H1a NMI(cluster, color) [circular]", "h1_exc_circ", "h1_p_circ"),
        ("H2a Color -> jump [circular]",       "h2a_diff",    "h2a_p_circ"),
        ("H2b Cluster -> jump [circular]",     "h2b_diff",    "h2b_p_circ"),
        ("Pure COLOR (inside cluster)",        "pure_col_diff", "pure_col_p"),
        ("Pure CLUSTER (same color)",          "pure_cl_diff",  "pure_cl_p"),
        ("H3a Color positional homology",      "h3_col_exc",  "h3_col_p"),
        ("H3b Numeric R^2 positional homology","h3_val_exc",  "h3_val_p"),
    ]
    fdr_rows = []
    for label, dcol, pcol in tests:
        n_sig, n_tot, pct_fdr = bh_fdr_mask(res[pcol], alpha=0.05)
        raw_pct = round(100.0 * (res[pcol].dropna() < 0.05).mean(), 1)
        med = round(float(res[dcol].dropna().median()), 4)
        wp = stats.wilcoxon(res[dcol].dropna()).pvalue
        fdr_rows.append(dict(test=label, n=n_tot, median=med, wilcoxon_p=f"{wp:.2e}", raw_p05_pct=raw_pct, fdr_q05_n=n_sig, fdr_q05_pct=pct_fdr))
    fdr_df = pd.DataFrame(fdr_rows)
    fdr_df.to_csv(RESULTS_DIR / "table_fdr_summary.csv", index=False)
    print(fdr_df.to_string(index=False))

    print("\n=== 2. CO-OCCURRENCE OF H3a (COLOR) AND H3b (NUMERIC) HOMOLOGY ===")
    both = res.dropna(subset=["h3_col_p", "h3_val_p"]).copy()
    both["q_col"] = false_discovery_control(both["h3_col_p"].values, method="bh")
    both["q_val"] = false_discovery_control(both["h3_val_p"].values, method="bh")
    both["sig_col"] = both["q_col"] < 0.05
    both["sig_val"] = both["q_val"] < 0.05
    ct = pd.crosstab(both["sig_col"], both["sig_val"])
    or_val, p_fish = stats.fisher_exact(ct)
    rho, p_rho = stats.spearmanr(both["h3_col_exc"], both["h3_val_exc"])
    print(f"Khipus testable for BOTH H3a and H3b: n={len(both)}")
    print("2x2 table (FDR q<0.05: rows=H3a color, cols=H3b numeric):\n", ct.to_string())
    print(f"Fisher exact: OR={or_val:.2f}, p={p_fish:.2e} | Spearman(h3_col_exc, h3_val_exc) = {rho:.3f} (p={p_rho:.2e})")

    print("\nTop 10 strongest tabular khipus (both H3a & H3b FDR q<0.05, ranked by h3_val_r2 + h3_col_match):")
    top_tab = both[both["sig_col"] & both["sig_val"]].assign(score=lambda d: d["h3_val_r2"] + d["h3_col_match"]).sort_values("score", ascending=False).head(10)
    cols_show = ["OKR_NUM", "cataloger", "museum", "h3_G", "h3_m", "h3_col_match", "h3_val_r2", "clean_dec_frac"]
    print(top_tab[cols_show].round(3).to_string(index=False))

    print("\n=== 3. SENSITIVITY: 100% CLEAN DECIMAL (clean_dec_frac==1.0) vs NON-CLEAN (<1.0) ===")
    for tag, mask in [("100% clean decimal", res["clean_dec_frac"] == 1.0), ("Contains non-decimal (<1.0)", res["clean_dec_frac"] < 1.0)]:
        sub = res[mask]
        _, n_h3v, pct_h3v = bh_fdr_mask(sub["h3_val_p"])
        _, n_h3c, pct_h3c = bh_fdr_mask(sub["h3_col_p"])
        _, n_pc, pct_pc = bh_fdr_mask(sub["pure_col_p"])
        _, n_pl, pct_pl = bh_fdr_mask(sub["pure_cl_p"])
        print(f"  {tag:28s} (n={len(sub):3d}): H3b_val FDR={pct_h3v:4.1f}% (n={n_h3v}) | H3a_col FDR={pct_h3c:4.1f}% (n={n_h3c}) | pure_col med={sub['pure_col_diff'].median():+.3f} | pure_cl med={sub['pure_cl_diff'].median():+.3f}")

    print("\n=== 4. H4 INVARIANCE BY TOP MUSEUMS (n >= 15 khipus) ===")
    top_m = res["museum"].value_counts()
    top_m = top_m[(top_m >= 15) & (top_m.index != "")].index.tolist()
    inv_m = res[res["museum"].isin(top_m)].groupby("museum").agg(
        n=("KHIPU_ID", "count"),
        h1_circ_raw=("h1_p_circ", lambda x: round(100 * (x.dropna() < 0.05).mean(), 1)),
        h3_col_raw=("h3_col_p", lambda x: round(100 * (x.dropna() < 0.05).mean(), 1)),
        h3_val_raw=("h3_val_p", lambda x: round(100 * (x.dropna() < 0.05).mean(), 1)),
        pure_cl_med=("pure_cl_diff", "median"),
    ).round(3)
    print(inv_m.to_string())


if __name__ == "__main__":
    main()
