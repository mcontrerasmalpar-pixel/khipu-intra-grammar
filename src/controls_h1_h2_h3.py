"""
controls_h1_h2_h3.py — Rigorous controls for khipu-intra-grammar:
  1) Circular-shift null (preserving local spatial autocorrelation) vs free-permutation null
  2) 2x2 decoupling of color boundaries vs cord_cluster boundaries (pure color within cluster,
     pure cluster boundary with identical color)
  3) Base-10 numeric decoding from knot + knot_cluster, plus H3 positional homology across
     equal-sized cord_clusters (color homology & numeric homology)
  4) H4 cataloger invariance across AS, HP, JC, QU, UR.
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


def decode_base10_cords(con: sqlite3.Connection) -> pd.DataFrame:
    """
    Decode base-10 numeric values per CORD_ID from knot_cluster + knot.
    Standard Ascher/Urton decimal convention:
      - Clusters ordered top-to-bottom by (ORDINAL, START_POS).
      - Bottom cluster: E -> 1, L(t) -> t (2..9), S(k) -> units=0 and k*10^1 upward.
      - Higher clusters: S(k) -> k (1..9).
    Flags `is_clean_decimal` when all cluster digits are in 1..9 and E/L appear only at bottom.
    """
    q = """
    SELECT kc.CORD_ID, kc.CLUSTER_ID, kc.ORDINAL AS cl_ord, kc.START_POS,
           UPPER(TRIM(COALESCE(k.TYPE_CODE, ''))) AS tcode,
           COALESCE(k.NUM_TURNS, 0) AS turns
    FROM knot_cluster kc
    JOIN knot k ON kc.CLUSTER_ID = k.CLUSTER_ID
    ORDER BY kc.CORD_ID, kc.ORDINAL, kc.START_POS, k.KNOT_ORDINAL
    """
    df = pd.read_sql(q, con)

    # Summarize each knot_cluster
    cl = df.groupby(["CORD_ID", "CLUSTER_ID", "cl_ord", "START_POS"], sort=False).agg(
        n_rows=("tcode", "size"),
        n_S=("tcode", lambda s: int((s == "S").sum())),
        n_L=("tcode", lambda s: int((s == "L").sum())),
        n_E=("tcode", lambda s: int((s == "E").sum())),
        max_turns=("turns", "max"),
    ).reset_index()

    out = []
    for cid, g in cl.groupby("CORD_ID", sort=False):
        g = g.sort_values(["cl_ord", "START_POS"])
        n_cl = len(g)
        clean = True
        digits = []
        bottom_is_units = False
        for i, row in enumerate(g.itertuples(index=False)):
            is_bottom = (i == n_cl - 1)
            if row.n_S == row.n_rows and 1 <= row.n_S <= 9:
                digits.append(int(row.n_S))
            elif row.n_E == row.n_rows and row.n_E == 1 and is_bottom:
                digits.append(1)
                bottom_is_units = True
            elif row.n_L == row.n_rows and row.n_L == 1 and 2 <= row.max_turns <= 9 and is_bottom:
                digits.append(int(round(row.max_turns)))
                bottom_is_units = True
            else:
                clean = False
                # Best-effort digit for non-clean cords
                if row.n_L > 0 and row.max_turns > 0:
                    digits.append(int(round(row.max_turns)))
                    if is_bottom:
                        bottom_is_units = True
                elif row.n_E > 0:
                    digits.append(int(row.n_E))
                    if is_bottom:
                        bottom_is_units = True
                else:
                    digits.append(int(row.n_rows))

        # Power assignment: if bottom is E/L -> 10^0..10^(k-1); if bottom is S -> 10^1..10^k
        val = 0
        offset = 0 if bottom_is_units else 1
        for pos_from_bottom, d in enumerate(reversed(digits)):
            power = pos_from_bottom + offset
            if power <= 7:
                val += d * (10 ** power)
            else:
                clean = False
        out.append((cid, float(val), int(clean), n_cl))

    return pd.DataFrame(out, columns=["CORD_ID", "cord_val", "is_clean_decimal", "n_knot_clusters"])


def load_pendants(db_path: pathlib.Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    vals = decode_base10_cords(con)
    q = """
    WITH kn AS (
        SELECT CORD_ID, COUNT(*) AS n_knots FROM knot GROUP BY CORD_ID
    ),
    col AS (
        SELECT CORD_ID,
               MIN(COALESCE(NULLIF(TRIM(FULL_COLOR), ''), COLOR_CD_1)) AS color_key
        FROM ascher_cord_color
        GROUP BY CORD_ID
    )
    SELECT
        c.KHIPU_ID, k.OKR_NUM, k.INVESTIGATOR_NUM, k.CREATED_BY, k.MUSEUM_NAME,
        c.CORD_ID, c.CLUSTER_ID, c.CORD_ORDINAL, c.ATTACH_POS,
        col.color_key,
        COALESCE(kn.n_knots, 0) AS n_knots
    FROM cord c
    JOIN khipu_main k ON c.KHIPU_ID = k.KHIPU_ID
    LEFT JOIN col ON c.CORD_ID = col.CORD_ID
    LEFT JOIN kn ON c.CORD_ID = kn.CORD_ID
    WHERE c.CORD_LEVEL = 1
    ORDER BY c.KHIPU_ID, c.CORD_ORDINAL, c.ATTACH_POS, c.CORD_ID
    """
    df = pd.read_sql(q, con)
    con.close()
    df = df.merge(vals, on="CORD_ID", how="left")
    df["cord_val"] = df["cord_val"].fillna(0.0)
    df["is_clean_decimal"] = df["is_clean_decimal"].fillna(1).astype(int)  # 0 knots = valid 0
    df["cataloger"] = (
        df["INVESTIGATOR_NUM"].fillna("").astype(str).str.extract(r"^([A-Za-z]{2})")[0].str.upper().fillna("NA")
    )
    return df


def circular_shifts(n: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    if n - 1 <= n_perm:
        return np.arange(1, n)
    return rng.integers(1, n, size=n_perm)


def run_controls(df: pd.DataFrame, n_perm: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for kid, g in df.groupby("KHIPU_ID", sort=False):
        n = len(g)
        if n < 10:
            continue

        colors = g["color_key"].fillna("__MISSING__").values
        clusters = g["CLUSTER_ID"].fillna(-1).values
        log_kn = np.log1p(g["n_knots"].values.astype(float))
        log_val = np.log1p(g["cord_val"].values.astype(float))
        clean_frac = float(g["is_clean_decimal"].mean())

        n_colors = len(set(colors) - {"__MISSING__"})
        n_clusters = len(set(clusters) - {-1})
        shifts = circular_shifts(n, n_perm, rng)

        # ---- CONTROL 1: H1a NMI with FREE perm vs CIRCULAR shift ----
        h1_nmi = h1_exc_free = h1_p_free = h1_exc_circ = h1_p_circ = np.nan
        if n_colors >= 2 and n_clusters >= 2:
            h1_nmi = float(normalized_mutual_info_score(clusters, colors))
            null_f = [normalized_mutual_info_score(clusters, rng.permutation(colors)) for _ in range(n_perm)]
            null_c = [normalized_mutual_info_score(clusters, np.roll(colors, int(s))) for s in shifts]
            h1_exc_free = h1_nmi - float(np.mean(null_f))
            h1_p_free = float((1 + sum(v >= h1_nmi for v in null_f)) / (len(null_f) + 1))
            h1_exc_circ = h1_nmi - float(np.mean(null_c))
            h1_p_circ = float((1 + sum(v >= h1_nmi for v in null_c)) / (len(null_c) + 1))

        col_ch = (colors[:-1] != colors[1:])
        cl_ch = (clusters[:-1] != clusters[1:])

        def jump_test_circular(y, mask):
            if mask.sum() < 3 or (~mask).sum() < 3 or np.std(y) == 0:
                return np.nan, np.nan, np.nan
            j = np.abs(np.diff(y))
            obs = float(j[mask].mean() - j[~mask].mean())
            nf = []
            for _ in range(n_perm):
                jp = np.abs(np.diff(rng.permutation(y)))
                nf.append(jp[mask].mean() - jp[~mask].mean())
            nc = []
            for s in shifts:
                jp = np.abs(np.diff(np.roll(y, int(s))))
                nc.append(jp[mask].mean() - jp[~mask].mean())
            pf = float((1 + sum(v >= obs for v in nf)) / (len(nf) + 1))
            pc = float((1 + sum(v >= obs for v in nc)) / (len(nc) + 1))
            return obs, pf, pc

        # H2a & H2b on log1p(cord_val) with BOTH free and circular nulls
        h2a_val, h2a_pf, h2a_pc = jump_test_circular(log_val, col_ch)
        h2b_val, h2b_pf, h2b_pc = jump_test_circular(log_val, cl_ch)

        # ---- CONTROL 2: 2x2 Decoupling (within same cluster / within same color) ----
        # 2a) Pure color effect: pairs strictly INSIDE the same cord_cluster
        same_cl = ~cl_ch
        pure_col_diff = pure_col_p = np.nan
        if (same_cl & col_ch).sum() >= 3 and (same_cl & (~col_ch)).sum() >= 3 and np.std(log_val) > 0:
            j = np.abs(np.diff(log_val))
            pure_col_diff = float(j[same_cl & col_ch].mean() - j[same_cl & (~col_ch)].mean())
            # Null: permute log_val strictly WITHIN each cord_cluster
            cl_indices = [np.where(clusters == c)[0] for c in np.unique(clusters)]
            null_pc = []
            for _ in range(n_perm):
                yp = log_val.copy()
                for idxs in cl_indices:
                    if len(idxs) > 1:
                        yp[idxs] = rng.permutation(yp[idxs])
                jp = np.abs(np.diff(yp))
                null_pc.append(jp[same_cl & col_ch].mean() - jp[same_cl & (~col_ch)].mean())
            pure_col_p = float((1 + sum(v >= pure_col_diff for v in null_pc)) / (n_perm + 1))

        # 2b) Pure spatial cluster effect: pairs with IDENTICAL color
        same_col = ~col_ch
        pure_cl_diff = pure_cl_p = np.nan
        if (same_col & cl_ch).sum() >= 3 and (same_col & (~cl_ch)).sum() >= 3 and np.std(log_val) > 0:
            j = np.abs(np.diff(log_val))
            pure_cl_diff = float(j[same_col & cl_ch].mean() - j[same_col & (~cl_ch)].mean())
            # Null: permute log_val strictly WITHIN each color
            col_indices = [np.where(colors == c)[0] for c in np.unique(colors)]
            null_pl = []
            for _ in range(n_perm):
                yp = log_val.copy()
                for idxs in col_indices:
                    if len(idxs) > 1:
                        yp[idxs] = rng.permutation(yp[idxs])
                jp = np.abs(np.diff(yp))
                null_pl.append(jp[same_col & cl_ch].mean() - jp[same_col & (~cl_ch)].mean())
            pure_cl_p = float((1 + sum(v >= pure_cl_diff for v in null_pl)) / (n_perm + 1))

        # ---- CONTROL 3: H3 Positional Homology across equal-sized cord_clusters ----
        h3_col_match = h3_col_exc = h3_col_p = np.nan
        h3_val_r2 = h3_val_exc = h3_val_p = np.nan
        cl_sizes = g.groupby("CLUSTER_ID", sort=False).size()
        valid_sizes = cl_sizes[cl_sizes >= 3]
        if len(valid_sizes) >= 2:
            size_counts = valid_sizes.value_counts()
            modal_m = int(size_counts.index[0])
            if size_counts.iloc[0] >= 2:
                hom_cids = valid_sizes[valid_sizes == modal_m].index.tolist()
                mat_col = []
                mat_val = []
                for cid_cl in hom_cids:
                    sub = g[g["CLUSTER_ID"] == cid_cl]
                    mat_col.append(sub["color_key"].fillna("__MISSING__").values)
                    mat_val.append(np.log1p(sub["cord_val"].values.astype(float)))
                mat_col = np.array(mat_col)  # shape (G, m)
                mat_val = np.array(mat_val)  # shape (G, m)
                G_cnt, m_len = mat_col.shape

                # H3a: Mean pairwise color match at same position j across groups
                def col_hom_score(M):
                    matches = 0
                    total = 0
                    for a in range(M.shape[0]):
                        for b in range(a + 1, M.shape[0]):
                            matches += int(np.sum(M[a] == M[b]))
                            total += M.shape[1]
                    return matches / total if total > 0 else np.nan

                if len(set(mat_col.ravel()) - {"__MISSING__"}) >= 2:
                    obs_cm = col_hom_score(mat_col)
                    null_cm = [
                        col_hom_score(np.array([rng.permutation(mat_col[r]) for r in range(G_cnt)]))
                        for _ in range(n_perm)
                    ]
                    h3_col_match = float(obs_cm)
                    h3_col_exc = float(obs_cm - np.mean(null_cm))
                    h3_col_p = float((1 + sum(v >= obs_cm for v in null_cm)) / (n_perm + 1))

                # H3b: Numeric homology (eta-squared / R^2 of log_val explained by within-group position j)
                def pos_r2(M):
                    tot_var = np.var(M)
                    if tot_var <= 1e-12:
                        return np.nan
                    pos_means = M.mean(axis=0, keepdims=True)
                    between = np.mean((pos_means - M.mean()) ** 2)
                    return float(between / tot_var)

                obs_r2 = pos_r2(mat_val)
                if not np.isnan(obs_r2):
                    null_r2 = [
                        pos_r2(np.array([rng.permutation(mat_val[r]) for r in range(G_cnt)]))
                        for _ in range(n_perm)
                    ]
                    h3_val_r2 = obs_r2
                    h3_val_exc = float(obs_r2 - np.mean(null_r2))
                    h3_val_p = float((1 + sum(v >= obs_r2 for v in null_r2)) / (n_perm + 1))

        rows.append({
            "KHIPU_ID": kid, "OKR_NUM": g["OKR_NUM"].iloc[0], "cataloger": g["cataloger"].iloc[0],
            "n_pendants": n, "clean_dec_frac": round(clean_frac, 3),
            "h1_nmi": h1_nmi, "h1_exc_free": h1_exc_free, "h1_p_free": h1_p_free,
            "h1_exc_circ": h1_exc_circ, "h1_p_circ": h1_p_circ,
            "h2a_val_diff": h2a_val, "h2a_p_free": h2a_pf, "h2a_p_circ": h2a_pc,
            "h2b_val_diff": h2b_val, "h2b_p_free": h2b_pf, "h2b_p_circ": h2b_pc,
            "pure_col_diff": pure_col_diff, "pure_col_p": pure_col_p,
            "pure_cl_diff": pure_cl_diff, "pure_cl_p": pure_cl_p,
            "h3_col_match": h3_col_match, "h3_col_exc": h3_col_exc, "h3_col_p": h3_col_p,
            "h3_val_r2": h3_val_r2, "h3_val_exc": h3_val_exc, "h3_val_p": h3_val_p,
        })

    return pd.DataFrame(rows)


def _summ(s_diff, s_p, label):
    d = s_diff.dropna()
    p_s = s_p.dropna()
    w = stats.wilcoxon(d) if len(d) >= 10 else None
    wp = f"{w.pvalue:.2e}" if w is not None else "n/a"
    print(f"  {label:42s}: n={len(d):3d} | med={d.median():+.4f} | Wilcoxon p={wp:>9s} | indiv p<0.05={100*(p_s<0.05).mean():4.1f}%")


def main():
    df = load_pendants(DEFAULT_DB)
    print(f"Loaded {len(df):,} pendants across {df['KHIPU_ID'].nunique()} khipus.")
    knotted = df[df["n_knots"] > 0]
    print(f"Base-10 decoder clean rate on knotted pendants: {100*knotted['is_clean_decimal'].mean():.1f}% ({knotted['is_clean_decimal'].sum():,}/{len(knotted):,})")
    res = run_controls(df, n_perm=200, seed=42)
    out = RESULTS_DIR / "controls_h1_h2_h3_results.csv"
    res.to_csv(out, index=False)
    print(f"Saved: {out}\n")

    print("=== CONTROL 1: CIRCULAR-SHIFT NULL vs FREE PERMUTATION ===")
    _summ(res["h1_exc_free"], res["h1_p_free"], "H1a NMI(cluster, color) [free perm]")
    _summ(res["h1_exc_circ"], res["h1_p_circ"], "H1a NMI(cluster, color) [CIRCULAR shift]")
    _summ(res["h2a_val_diff"], res["h2a_p_free"], "H2a Color->|Delta log(val)| [free perm]")
    _summ(res["h2a_val_diff"], res["h2a_p_circ"], "H2a Color->|Delta log(val)| [CIRCULAR]")
    _summ(res["h2b_val_diff"], res["h2b_p_free"], "H2b Cluster->|Delta log(val)| [free perm]")
    _summ(res["h2b_val_diff"], res["h2b_p_circ"], "H2b Cluster->|Delta log(val)| [CIRCULAR]")

    print("\n=== CONTROL 2: 2x2 DECOUPLING (on base-10 log1p(cord_val)) ===")
    _summ(res["pure_col_diff"], res["pure_col_p"], "Pure COLOR effect (inside same cluster)")
    _summ(res["pure_cl_diff"], res["pure_cl_p"], "Pure CLUSTER effect (same color)")

    print("\n=== CONTROL 3: H3 POSITIONAL HOMOLOGY (equal-sized cord_clusters, m>=3) ===")
    _summ(res["h3_col_exc"], res["h3_col_p"], "H3a Color homology across groups")
    _summ(res["h3_val_exc"], res["h3_val_p"], "H3b Numeric R^2 homology across groups")

    print("\n=== CONTROL 4: H4 CATALOGER INVARIANCE (new controls) ===")
    top = res["cataloger"].value_counts()
    top = top[top >= 10].index.tolist()
    inv = res[res["cataloger"].isin(top)].groupby("cataloger").agg(
        n=("KHIPU_ID", "count"),
        h1_circ_sig=("h1_p_circ", lambda x: round(100 * (x.dropna() < 0.05).mean(), 1)),
        pure_col_med=("pure_col_diff", "median"),
        pure_cl_med=("pure_cl_diff", "median"),
        h3_col_sig=("h3_col_p", lambda x: round(100 * (x.dropna() < 0.05).mean(), 1)),
        h3_val_sig=("h3_val_p", lambda x: round(100 * (x.dropna() < 0.05).mean(), 1)),
    ).round(4)
    print(inv.to_string())


if __name__ == "__main__":
    main()
