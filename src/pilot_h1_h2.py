"""
pilot_h1_h2.py — Fast pilot testing H1 (color vs cord_cluster & banding/seriation),
H2 (color/cluster boundaries vs knot count, blank cords & subsidiaries), and
H4 (cataloger invariance) using within-khipu permutation nulls on OKR v2.1.0.
"""

import os
import pathlib
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score

DEFAULT_DB = pathlib.Path(
    os.environ.get(
        "KHIPU_DB_PATH",
        r"C:\Users\maria\.gemini\antigravity\scratch\Khipu-ML\data\open-khipu-repository-2.1.0\data\khipu.db",
    )
)
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_pendant_table(db_path: pathlib.Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    q = """
    WITH subs AS (
        SELECT ATTACHED_TO AS parent_cord_id, COUNT(*) AS n_subs
        FROM cord
        WHERE CORD_LEVEL > 1 AND ATTACHED_TO IS NOT NULL
        GROUP BY ATTACHED_TO
    ),
    kn AS (
        SELECT CORD_ID,
               COUNT(*) AS n_knots,
               SUM(COALESCE(NUM_TURNS, 1)) AS sum_turns
        FROM knot
        GROUP BY CORD_ID
    ),
    col AS (
        SELECT CORD_ID,
               MIN(COALESCE(NULLIF(TRIM(FULL_COLOR), ''), COLOR_CD_1)) AS color_key,
               MIN(COLOR_CD_1) AS color_primary
        FROM ascher_cord_color
        GROUP BY CORD_ID
    )
    SELECT
        c.KHIPU_ID,
        k.OKR_NUM,
        k.INVESTIGATOR_NUM,
        k.CREATED_BY,
        k.MUSEUM_NAME,
        c.CORD_ID,
        c.CLUSTER_ID,
        c.CORD_ORDINAL,
        c.ATTACH_POS,
        col.color_key,
        col.color_primary,
        COALESCE(kn.n_knots, 0) AS n_knots,
        COALESCE(kn.sum_turns, 0) AS sum_turns,
        COALESCE(subs.n_subs, 0) AS n_subs
    FROM cord c
    JOIN khipu_main k ON c.KHIPU_ID = k.KHIPU_ID
    LEFT JOIN col ON c.CORD_ID = col.CORD_ID
    LEFT JOIN kn ON c.CORD_ID = kn.CORD_ID
    LEFT JOIN subs ON c.CORD_ID = subs.parent_cord_id
    WHERE c.CORD_LEVEL = 1
    ORDER BY c.KHIPU_ID, c.CORD_ORDINAL, c.ATTACH_POS, c.CORD_ID
    """
    df = pd.read_sql(q, con)
    con.close()
    df["cataloger"] = (
        df["INVESTIGATOR_NUM"].fillna("").astype(str).str.extract(r"^([A-Za-z]{2})")[0].str.upper().fillna("NA")
    )
    return df


def run_pilot(df: pd.DataFrame, n_perm: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for kid, g in df.groupby("KHIPU_ID", sort=False):
        n = len(g)
        if n < 10:
            continue

        colors = g["color_key"].fillna("__MISSING__").values
        clusters = g["CLUSTER_ID"].fillna(-1).values
        n_knots = g["n_knots"].values.astype(float)
        log_kn = np.log1p(n_knots)
        has_sub = (g["n_subs"].values > 0).astype(float)
        is_blank = (n_knots == 0).astype(float)

        n_colors = len(set(colors) - {"__MISSING__"})
        n_clusters = len(set(clusters) - {-1})

        # --- H1a: Color vs cord_cluster NMI (within-khipu permutation null) ---
        h1_nmi, h1_nmi_null, h1_nmi_p = np.nan, np.nan, np.nan
        if n_colors >= 2 and n_clusters >= 2:
            h1_nmi = float(normalized_mutual_info_score(clusters, colors))
            null_nmi = [
                normalized_mutual_info_score(clusters, rng.permutation(colors))
                for _ in range(n_perm)
            ]
            h1_nmi_null = float(np.mean(null_nmi))
            h1_nmi_p = float((1 + sum(v >= h1_nmi for v in null_nmi)) / (n_perm + 1))

        # --- H1b: Adjacent color runs (banding > null vs seriation/alternation < null) ---
        h1_same_adj, h1_same_null, h1_same_z = np.nan, np.nan, np.nan
        if n_colors >= 2:
            same_obs = float(np.mean(colors[:-1] == colors[1:]))
            null_same = np.array([
                np.mean((p := rng.permutation(colors))[:-1] == p[1:])
                for _ in range(n_perm)
            ])
            h1_same_adj = same_obs
            h1_same_null = float(null_same.mean())
            sd = float(null_same.std())
            h1_same_z = float((same_obs - h1_same_null) / sd) if sd > 1e-9 else 0.0

        # --- H2a: Color boundary vs numeric jump |log1p(n_knots)_{i+1} - log1p(n_knots)_i| ---
        h2_col_diff, h2_col_p = np.nan, np.nan
        col_change = (colors[:-1] != colors[1:])
        if col_change.sum() >= 3 and (~col_change).sum() >= 3 and np.std(log_kn) > 0:
            jumps = np.abs(np.diff(log_kn))
            obs_diff = float(jumps[col_change].mean() - jumps[~col_change].mean())
            null_diffs = []
            for _ in range(n_perm):
                lp = rng.permutation(log_kn)
                jp = np.abs(np.diff(lp))
                null_diffs.append(jp[col_change].mean() - jp[~col_change].mean())
            h2_col_diff = obs_diff
            h2_col_p = float((1 + sum(v >= obs_diff for v in null_diffs)) / (n_perm + 1))

        # --- H2b: Cluster boundary vs numeric jump ---
        h2_cl_diff, h2_cl_p = np.nan, np.nan
        cl_change = (clusters[:-1] != clusters[1:])
        if cl_change.sum() >= 3 and (~cl_change).sum() >= 3 and np.std(log_kn) > 0:
            jumps = np.abs(np.diff(log_kn))
            obs_diff = float(jumps[cl_change].mean() - jumps[~cl_change].mean())
            null_diffs = []
            for _ in range(n_perm):
                lp = rng.permutation(log_kn)
                jp = np.abs(np.diff(lp))
                null_diffs.append(jp[cl_change].mean() - jp[~cl_change].mean())
            h2_cl_diff = obs_diff
            h2_cl_p = float((1 + sum(v >= obs_diff for v in null_diffs)) / (n_perm + 1))

        # --- H2c: Subsidiaries concentrated at color boundaries? ---
        h2_sub_diff = np.nan
        if has_sub.sum() >= 2 and (1 - has_sub).sum() >= 2 and col_change.sum() >= 3 and (~col_change).sum() >= 3:
            at_b = has_sub[1:][col_change].mean()
            in_b = has_sub[1:][~col_change].mean()
            h2_sub_diff = float(at_b - in_b)

        rows.append({
            "KHIPU_ID": kid,
            "OKR_NUM": g["OKR_NUM"].iloc[0],
            "cataloger": g["cataloger"].iloc[0],
            "created_by": g["CREATED_BY"].iloc[0],
            "n_pendants": n,
            "n_colors": n_colors,
            "n_clusters": n_clusters,
            "h1_nmi": h1_nmi,
            "h1_nmi_excess": h1_nmi - h1_nmi_null if not np.isnan(h1_nmi) else np.nan,
            "h1_nmi_p": h1_nmi_p,
            "h1_same_adj": h1_same_adj,
            "h1_same_null": h1_same_null,
            "h1_same_z": h1_same_z,
            "h2_col_jump_diff": h2_col_diff,
            "h2_col_jump_p": h2_col_p,
            "h2_cl_jump_diff": h2_cl_diff,
            "h2_cl_jump_p": h2_cl_p,
            "h2_sub_at_col_boundary": h2_sub_diff,
            "blank_rate": float(is_blank.mean()),
        })

    return pd.DataFrame(rows)


def main():
    df = load_pendant_table(DEFAULT_DB)
    print(f"Loaded {len(df):,} pendant cords across {df['KHIPU_ID'].nunique()} khipus.")
    res = run_pilot(df, n_perm=200, seed=42)
    out_csv = RESULTS_DIR / "pilot_h1_h2_results.csv"
    res.to_csv(out_csv, index=False)
    print(f"Analyzed {len(res)} khipus (>=10 pendants). Saved: {out_csv}\n")

    # Summary stats
    print("=== H1a: Color vs cord_cluster (NMI excess over within-khipu null) ===")
    s1 = res["h1_nmi_excess"].dropna()
    w1 = stats.wilcoxon(s1)
    sig1 = (res["h1_nmi_p"] < 0.05).mean()
    print(f"  n={len(s1)} khipus | median NMI={res['h1_nmi'].median():.3f} | median excess={s1.median():+.3f} | Wilcoxon p={w1.pvalue:.2e} | share p<0.05 = {100*sig1:.1f}%")

    print("\n=== H1b: Adjacent color runs (banding z>2 vs seriation/alternation z<-2) ===")
    z = res["h1_same_z"].dropna()
    print(f"  n={len(z)} khipus | median z={z.median():+.2f} | banding (z>+2): {100*(z>2).mean():.1f}% | alternation/seriation (z<-2): {100*(z<-2).mean():.1f}%")

    print("\n=== H2a: Color boundary -> numeric jump |Delta log1p(n_knots)| ===")
    s2a = res["h2_col_jump_diff"].dropna()
    w2a = stats.wilcoxon(s2a)
    print(f"  n={len(s2a)} khipus | median diff={s2a.median():+.4f} | mean diff={s2a.mean():+.4f} | Wilcoxon p={w2a.pvalue:.2e} | share p<0.05 = {100*(res['h2_col_jump_p']<0.05).mean():.1f}%")

    print("\n=== H2b: Cluster boundary -> numeric jump |Delta log1p(n_knots)| ===")
    s2b = res["h2_cl_jump_diff"].dropna()
    w2b = stats.wilcoxon(s2b)
    print(f"  n={len(s2b)} khipus | median diff={s2b.median():+.4f} | mean diff={s2b.mean():+.4f} | Wilcoxon p={w2b.pvalue:.2e} | share p<0.05 = {100*(res['h2_cl_jump_p']<0.05).mean():.1f}%")

    print("\n=== H4: Cataloger invariance (median H1 NMI excess, H1 banding z, H2 color-jump diff) ===")
    top_cats = res["cataloger"].value_counts()
    top_cats = top_cats[top_cats >= 10].index.tolist()
    inv = res[res["cataloger"].isin(top_cats)].groupby("cataloger").agg(
        n_khipus=("KHIPU_ID", "count"),
        med_nmi_excess=("h1_nmi_excess", "median"),
        pct_nmi_sig=("h1_nmi_p", lambda x: round(100 * (x < 0.05).mean(), 1)),
        med_band_z=("h1_same_z", "median"),
        med_col_jump=("h2_col_jump_diff", "median"),
        med_cl_jump=("h2_cl_jump_diff", "median"),
    ).round(3)
    print(inv.to_string())


if __name__ == "__main__":
    main()
