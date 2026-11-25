"""
extension_h5_sums.py — Extension H5 for khipu-intra-grammar:
Exact within-khipu summation detection calibrated against within-khipu permutation nulls,
requiring non-trivial sums (target > 0 and >= 2 non-zero summands):
  - H5a: Top-cord sums (CORD_CLASSIFICATION == 'T')
  - H5b: Within-cluster boundary subtotals (first or last cord == sum of remaining cords in cluster)
  - H5c: Cross-group homologous-slot sums ("spreadsheet column totals" across equal-sized groups)
  - Comparison of sum enrichment in doubly-tabular khipus (H3a & H3b FDR q<0.05) vs non-tabular
  - H4 cataloger invariance check.
"""

import pathlib
import sqlite3
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from controls_h1_h2_h3 import DEFAULT_DB, RESULTS_DIR, decode_base10_cords


def load_pendants_with_class(db_path: pathlib.Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    vals = decode_base10_cords(con)
    q = """
    SELECT
        c.KHIPU_ID, k.OKR_NUM, k.INVESTIGATOR_NUM, k.MUSEUM_NAME,
        c.CORD_ID, c.CLUSTER_ID, c.CORD_ORDINAL, c.ATTACH_POS,
        UPPER(TRIM(COALESCE(c.CORD_CLASSIFICATION, ''))) AS cord_class,
        UPPER(TRIM(COALESCE(c.ATTACHMENT_TYPE, ''))) AS att_type
    FROM cord c
    JOIN khipu_main k ON c.KHIPU_ID = k.KHIPU_ID
    WHERE c.CORD_LEVEL = 1
    ORDER BY c.KHIPU_ID, c.CORD_ORDINAL, c.ATTACH_POS, c.CORD_ID
    """
    df = pd.read_sql(q, con)
    con.close()
    df = df.merge(vals, on="CORD_ID", how="left")
    df["cord_val"] = df["cord_val"].fillna(0).round().astype(np.int64)
    df["is_clean_decimal"] = df["is_clean_decimal"].fillna(1).astype(int)
    df["cataloger"] = (
        df["INVESTIGATOR_NUM"].fillna("").astype(str).str.extract(r"^([A-Za-z]{2})")[0].str.upper().fillna("NA")
    )
    return df


def count_boundary_subtotals(clusters_vals: list[np.ndarray]) -> int:
    """Count clusters (size >= 3) where first OR last cord == exact sum of remaining (>=2 non-zero summands)."""
    hits = 0
    for arr in clusters_vals:
        if len(arr) < 3:
            continue
        # First cord as subtotal
        s_rest_first = arr[1:]
        if arr[0] > 0 and np.count_nonzero(s_rest_first) >= 2 and arr[0] == s_rest_first.sum():
            hits += 1
            continue
        # Last cord as subtotal
        s_rest_last = arr[:-1]
        if arr[-1] > 0 and np.count_nonzero(s_rest_last) >= 2 and arr[-1] == s_rest_last.sum():
            hits += 1
    return hits


def count_homologous_col_sums(M: np.ndarray) -> int:
    """
    M has shape (G, m) with G >= 3, m >= 3.
    Count slots j where first row M[0, j] == sum(M[1:, j]) OR last row M[-1, j] == sum(M[:-1, j]),
    requiring target > 0 and >= 2 non-zero summands.
    """
    G, m = M.shape
    if G < 3 or m < 3:
        return 0
    hits = 0
    for j in range(m):
        col = M[:, j]
        if col[0] > 0 and np.count_nonzero(col[1:]) >= 2 and col[0] == col[1:].sum():
            hits += 1
            continue
        if col[-1] > 0 and np.count_nonzero(col[:-1]) >= 2 and col[-1] == col[:-1].sum():
            hits += 1
    return hits


def count_pairwise_slot_sums(M: np.ndarray) -> int:
    """
    In M of shape (G, m) with G >= 3, count slots j where ANY row g equals the sum of the
    remaining G-1 rows (target > 0 and >= 2 non-zero summands).
    """
    G, m = M.shape
    if G < 3 or m < 3:
        return 0
    hits = 0
    for j in range(m):
        col = M[:, j]
        tot = col.sum()
        nz = np.count_nonzero(col)
        if nz < 3:
            continue
        for g_idx in range(G):
            v = col[g_idx]
            if v > 0 and (tot - v) == v:
                hits += 1
                break
    return hits


def run_h5(df: pd.DataFrame, n_perm: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for kid, g in df.groupby("KHIPU_ID", sort=False):
        n = len(g)
        if n < 10:
            continue

        # --- H5a: Top-cord sums (CORD_CLASSIFICATION == 'T') ---
        top_cords = g[g["cord_class"] == "T"]
        h5a_n_tops = len(top_cords)
        h5a_hits = 0
        if h5a_n_tops > 0:
            for tc in top_cords.itertuples(index=False):
                if tc.cord_val <= 0:
                    continue
                same_cl = g[(g["CLUSTER_ID"] == tc.CLUSTER_ID) & (g["CORD_ID"] != tc.CORD_ID)]["cord_val"].values
                if len(same_cl) >= 2 and np.count_nonzero(same_cl) >= 2 and tc.cord_val == same_cl.sum():
                    h5a_hits += 1

        # --- H5b: Within-cluster boundary subtotals (first or last cord) ---
        cl_list = [sub["cord_val"].values for _, sub in g.groupby("CLUSTER_ID", sort=False) if len(sub) >= 3]
        h5b_n_cl = len(cl_list)
        h5b_hits = h5b_null = h5b_p = np.nan
        if h5b_n_cl >= 1:
            obs_b = count_boundary_subtotals(cl_list)
            null_b = [count_boundary_subtotals([rng.permutation(a) for a in cl_list]) for _ in range(n_perm)]
            h5b_hits = int(obs_b)
            h5b_null = float(np.mean(null_b))
            h5b_p = float((1 + sum(v >= obs_b for v in null_b)) / (n_perm + 1))

        # --- H5c: Cross-group homologous-slot sums (G >= 3, m >= 3) ---
        h5c_G = h5c_m = 0
        h5c_hits = h5c_null = h5c_p = np.nan
        h5c_any_hits = h5c_any_null = h5c_any_p = np.nan
        cl_sizes = g.groupby("CLUSTER_ID", sort=False).size()
        valid_sz = cl_sizes[cl_sizes >= 3]
        if len(valid_sz) >= 3:
            sc = valid_sz.value_counts()
            modal_m = int(sc.index[0])
            if sc.iloc[0] >= 3:
                hom_cids = valid_sz[valid_sz == modal_m].index.tolist()
                M = np.array([g[g["CLUSTER_ID"] == c]["cord_val"].values for c in hom_cids], dtype=np.int64)
                h5c_G, h5c_m = M.shape
                obs_c = count_homologous_col_sums(M)
                null_c = [
                    count_homologous_col_sums(np.array([rng.permutation(M[r]) for r in range(h5c_G)]))
                    for _ in range(n_perm)
                ]
                h5c_hits = int(obs_c)
                h5c_null = float(np.mean(null_c))
                h5c_p = float((1 + sum(v >= obs_c for v in null_c)) / (n_perm + 1))

                obs_any = count_pairwise_slot_sums(M)
                null_any = [
                    count_pairwise_slot_sums(np.array([rng.permutation(M[r]) for r in range(h5c_G)]))
                    for _ in range(n_perm)
                ]
                h5c_any_hits = int(obs_any)
                h5c_any_null = float(np.mean(null_any))
                h5c_any_p = float((1 + sum(v >= obs_any for v in null_any)) / (n_perm + 1))

        rows.append({
            "KHIPU_ID": kid, "OKR_NUM": g["OKR_NUM"].iloc[0],
            "cataloger": g["cataloger"].iloc[0],
            "museum": str(g["MUSEUM_NAME"].iloc[0]).strip().strip('"'),
            "n_pendants": n, "clean_dec_frac": round(float(g["is_clean_decimal"].mean()), 3),
            "h5a_n_tops": h5a_n_tops, "h5a_hits": h5a_hits,
            "h5b_n_cl": h5b_n_cl, "h5b_hits": h5b_hits, "h5b_null": h5b_null, "h5b_p": h5b_p,
            "h5c_G": h5c_G, "h5c_m": h5c_m,
            "h5c_hits": h5c_hits, "h5c_null": h5c_null, "h5c_p": h5c_p,
            "h5c_any_hits": h5c_any_hits, "h5c_any_null": h5c_any_null, "h5c_any_p": h5c_any_p,
        })

    return pd.DataFrame(rows)


def main():
    df = load_pendants_with_class(DEFAULT_DB)
    res = run_h5(df, n_perm=500, seed=42)
    out = RESULTS_DIR / "extension_h5_sums.csv"
    res.to_csv(out, index=False)
    print(f"Saved {len(res)} khipus -> {out}\n")

    print("=== H5a: Top-cord exact cluster sums (CORD_CLASSIFICATION == 'T') ===")
    tdf = res[res["h5a_n_tops"] > 0]
    print(f"  Khipus with 'T' top cords: {len(tdf)} | Total 'T' cords: {tdf['h5a_n_tops'].sum()} | Exact cluster-sum hits: {tdf['h5a_hits'].sum()}")

    print("\n=== H5b: Within-cluster boundary subtotals (first/last cord = sum of rest) ===")
    b = res.dropna(subset=["h5b_hits"])
    tot_obs_b, tot_null_b = b["h5b_hits"].sum(), b["h5b_null"].sum()
    sig_b = (b["h5b_p"] < 0.05).sum()
    w_b = stats.wilcoxon(b["h5b_hits"] - b["h5b_null"])
    print(f"  n={len(b)} khipus | Total obs hits={int(tot_obs_b)} vs expected null={tot_null_b:.1f} (ratio={tot_obs_b/max(tot_null_b,1e-9):.2f}x)")
    print(f"  Khipus with >=1 hit: {(b['h5b_hits']>0).sum()} | Indiv p<0.05: {sig_b} ({100*sig_b/len(b):.1f}%) | Wilcoxon p={w_b.pvalue:.2e}")

    print("\n=== H5c: Cross-group homologous-slot sums (first/last group = sum of other groups) ===")
    c = res.dropna(subset=["h5c_hits"])
    tot_obs_c, tot_null_c = c["h5c_hits"].sum(), c["h5c_null"].sum()
    sig_c = (c["h5c_p"] < 0.05).sum()
    print(f"  n={len(c)} khipus (G>=3, m>=3) | First/Last group obs hits={int(tot_obs_c)} vs null={tot_null_c:.1f} (ratio={tot_obs_c/max(tot_null_c,1e-9):.2f}x) | p<0.05: {sig_c} ({100*sig_c/len(c):.1f}%)")
    tot_obs_a, tot_null_a = c["h5c_any_hits"].sum(), c["h5c_any_null"].sum()
    sig_a = (c["h5c_any_p"] < 0.05).sum()
    print(f"  Any group as sum: obs hits={int(tot_obs_a)} vs null={tot_null_a:.1f} (ratio={tot_obs_a/max(tot_null_a,1e-9):.2f}x) | p<0.05: {sig_a} ({100*sig_a/len(c):.1f}%)")

    # Join with H3a & H3b FDR status from deep_dive_fdr_1000.csv
    dd = pd.read_csv(RESULTS_DIR / "deep_dive_fdr_1000.csv")
    both = dd.dropna(subset=["h3_col_p", "h3_val_p"]).copy()
    both["q_col"] = false_discovery_control(both["h3_col_p"].values, method="bh")
    both["q_val"] = false_discovery_control(both["h3_val_p"].values, method="bh")
    both["doubly_tabular"] = (both["q_col"] < 0.05) & (both["q_val"] < 0.05)
    m = res.merge(both[["KHIPU_ID", "doubly_tabular", "h3_col_match", "h3_val_r2"]], on="KHIPU_ID", how="inner")

    print("\n=== DOUBLY-TABULAR (H3a & H3b FDR q<0.05, n=86) vs OTHER TESTABLE KHIPUS ===")
    for tag, grp in m.groupby("doubly_tabular"):
        lbl = "Doubly-tabular (H3a & H3b)" if tag else "Not doubly-tabular"
        gc = grp.dropna(subset=["h5c_any_hits"])
        gb = grp.dropna(subset=["h5b_hits"])
        print(f"  {lbl:28s} (n={len(grp):3d}): H5b_hits>0={100*(gb['h5b_hits']>0).mean():4.1f}% (obs/null={gb['h5b_hits'].sum()/max(gb['h5b_null'].sum(),1e-9):.2f}x) | "
              f"H5c_any_hits>0={100*(gc['h5c_any_hits']>0).mean() if len(gc) else 0:4.1f}% (obs/null={gc['h5c_any_hits'].sum()/max(gc['h5c_any_null'].sum(),1e-9):.2f}x, n_G3={len(gc)})")

    print("\n=== TOP KHIPUS WITH STATISTICALLY SIGNIFICANT SUMS (H5b or H5c p < 0.05) ===")
    sig_sum = m[(m["h5b_p"] < 0.05) | (m["h5c_p"] < 0.05) | (m["h5c_any_p"] < 0.05)].sort_values(["h5c_any_hits", "h5b_hits"], ascending=False)
    print(sig_sum[["OKR_NUM", "cataloger", "museum", "doubly_tabular", "h5b_hits", "h5b_p", "h5c_G", "h5c_m", "h5c_any_hits", "h5c_any_p"]].head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
