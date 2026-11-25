"""
round2_checks.py — Round-2 peer-review empirical checks for khipu-intra-grammar:
  1) H2b and pure_cluster restricted strictly to cord_cluster boundaries with physical SPACING > 0
  2) Full H2 (H2a, H2b, pure_col, pure_cl) + H3 on 100% clean decimal khipus (n=214)
  3) Exact aggregate Monte Carlo p-value for H5c (sum of hits across all 244 khipus vs null sum distribution, N=1000)
  4) Holm-Bonferroni correction across the 5 museum-collapsed Wilcoxon tests.
"""

import pathlib, sqlite3, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from controls_h1_h2_h3 import DEFAULT_DB, RESULTS_DIR, load_pendants, circular_shifts
from extension_h5_sums import load_pendants_with_class, count_homologous_col_sums, count_pairwise_slot_sums

con = sqlite3.connect(str(DEFAULT_DB))
cl_sp = pd.read_sql("SELECT CLUSTER_ID, COALESCE(SPACING, 0) AS spacing FROM cord_cluster", con)
con.close()
sp_map = dict(zip(cl_sp["CLUSTER_ID"], cl_sp["spacing"]))

df = load_pendants(DEFAULT_DB)
df["cl_spacing"] = df["CLUSTER_ID"].map(sp_map).fillna(0.0)

# --- CHECK 1: H2b and pure_cluster strictly on boundaries where adjacent cluster has physical SPACING > 0 ---
print("=== 1. H2b & PURE CLUSTER RESTRICTED TO PHYSICAL SPACING > 0 ===")
rng = np.random.default_rng(42)
N_PERM = 500
h2b_sp_diffs, h2b_sp_ps = [], []
pure_sp_diffs, pure_sp_ps = [], []

for kid, g in df.groupby("KHIPU_ID", sort=False):
    n = len(g)
    if n < 10:
        continue
    colors = g["color_key"].fillna("__MISSING__").values
    clusters = g["CLUSTER_ID"].fillna(-1).values
    spacings = g["cl_spacing"].values
    log_val = np.log1p(g["cord_val"].values.astype(float))
    if np.std(log_val) == 0:
        continue

    cl_ch = (clusters[:-1] != clusters[1:])
    # Physical boundary: cluster changes AND at least one of the two adjacent clusters has SPACING > 0
    phys_cl_ch = cl_ch & ((spacings[:-1] > 0) | (spacings[1:] > 0))
    same_cl = ~cl_ch

    # H2b physical vs within-cluster
    if phys_cl_ch.sum() >= 3 and same_cl.sum() >= 3:
        j = np.abs(np.diff(log_val))
        obs = float(j[phys_cl_ch].mean() - j[same_cl].mean())
        shifts = circular_shifts(n, N_PERM, rng)
        cnt = 0
        for s in shifts:
            jp = np.abs(np.diff(np.roll(log_val, int(s))))
            if (jp[phys_cl_ch].mean() - jp[same_cl].mean()) >= obs:
                cnt += 1
        h2b_sp_diffs.append(obs)
        h2b_sp_ps.append((1 + cnt) / (len(shifts) + 1))

    # Pure physical cluster (same color)
    same_col = (colors[:-1] == colors[1:])
    m1 = same_col & phys_cl_ch
    m0 = same_col & same_cl
    if m1.sum() >= 3 and m0.sum() >= 3:
        j = np.abs(np.diff(log_val))
        obs = float(j[m1].mean() - j[m0].mean())
        col_idxs = [np.where(colors == c)[0] for c in np.unique(colors)]
        cnt = 0
        for _ in range(N_PERM):
            yp = log_val.copy()
            for idxs in col_idxs:
                if len(idxs) > 1:
                    yp[idxs] = rng.permutation(yp[idxs])
            jp = np.abs(np.diff(yp))
            if (jp[m1].mean() - jp[m0].mean()) >= obs:
                cnt += 1
        pure_sp_diffs.append(obs)
        pure_sp_ps.append((1 + cnt) / (N_PERM + 1))

h2b_sp_diffs, h2b_sp_ps = np.array(h2b_sp_diffs), np.array(h2b_sp_ps)
pure_sp_diffs, pure_sp_ps = np.array(pure_sp_diffs), np.array(pure_sp_ps)
print(f"  H2b (SPACING > 0 only)        : n={len(h2b_sp_diffs):3d} | med={np.median(h2b_sp_diffs):+.4f} | Wilcoxon p={stats.wilcoxon(h2b_sp_diffs).pvalue:.2e} | raw p<0.05={100*(h2b_sp_ps<0.05).mean():.1f}%")
print(f"  Pure CLUSTER (SPACING > 0)    : n={len(pure_sp_diffs):3d} | med={np.median(pure_sp_diffs):+.4f} | Wilcoxon p={stats.wilcoxon(pure_sp_diffs).pvalue:.2e} | raw p<0.05={100*(pure_sp_ps<0.05).mean():.1f}%")

# --- CHECK 2: Full H2 & H3 on 100% clean decimal khipus (n=214) ---
print("\n=== 2. FULL H2 & H3 ON 100% CLEAN DECIMAL KHIPUS (clean_dec_frac == 1.0) ===")
dd = pd.read_csv(RESULTS_DIR / "deep_dive_fdr_1000.csv")
cln = dd[dd["clean_dec_frac"] == 1.0]
for col, pcol, lbl in [
    ("h2a_diff", "h2a_p_circ", "H2a Color->jump [circular]"),
    ("h2b_diff", "h2b_p_circ", "H2b Cluster->jump [circular]"),
    ("pure_col_diff", "pure_col_p", "Pure COLOR (inside cluster)"),
    ("pure_cl_diff", "pure_cl_p", "Pure CLUSTER (same color)"),
    ("h3_col_exc", "h3_col_p", "H3a Color homology"),
    ("h3_val_exc", "h3_val_p", "H3b Numeric R^2 homology"),
]:
    s = cln[col].dropna()
    ps = cln[pcol].dropna()
    wp = stats.wilcoxon(s).pvalue
    print(f"  {lbl:32s}: n={len(s):3d} | med={s.median():+.4f} | Wilcoxon p={wp:.2e} | raw p<0.05={100*(ps<0.05).mean():.1f}%")

# --- CHECK 3: Exact aggregate Monte Carlo p-value for H5c (N=1000 corpus-wide sum) ---
print("\n=== 3. H5c AGGREGATE CORPUS MONTE CARLO P-VALUE (N=1000) ===")
df5 = load_pendants_with_class(DEFAULT_DB)
matrices = []
for kid, g in df5.groupby("KHIPU_ID", sort=False):
    if len(g) < 10:
        continue
    cs = g.groupby("CLUSTER_ID", sort=False).size()
    vs = cs[cs >= 3]
    if len(vs) >= 3:
        sc = vs.value_counts()
        if sc.iloc[0] >= 3:
            m_mod = int(sc.index[0])
            cids = vs[vs == m_mod].index.tolist()
            M = np.array([g[g["CLUSTER_ID"] == c]["cord_val"].values for c in cids], dtype=np.int64)
            matrices.append(M)

obs_fl = sum(count_homologous_col_sums(M) for M in matrices)
obs_any = sum(count_pairwise_slot_sums(M) for M in matrices)
rng5 = np.random.default_rng(42)
N_AGG = 1000
null_fl = np.zeros(N_AGG, dtype=int)
null_any = np.zeros(N_AGG, dtype=int)
for it in range(N_AGG):
    s_fl = 0
    s_any = 0
    for M in matrices:
        Mp = np.array([rng5.permutation(M[r]) for r in range(M.shape[0])])
        s_fl += count_homologous_col_sums(Mp)
        s_any += count_pairwise_slot_sums(Mp)
    null_fl[it] = s_fl
    null_any[it] = s_any

cnt_fl = int((null_fl >= obs_fl).sum())
cnt_any = int((null_any >= obs_any).sum())
print(f"  n_khipus={len(matrices)}")
print(f"  H5c first/last: obs={obs_fl}, null mean={null_fl.mean():.1f} (sd={null_fl.std():.1f}, p95={np.percentile(null_fl,95):.1f}), count_ge={cnt_fl}/{N_AGG}, agg p={(cnt_fl+1)/(N_AGG+1):.4f}")
print(f"  H5c any-group : obs={obs_any}, null mean={null_any.mean():.1f} (sd={null_any.std():.1f}, p95={np.percentile(null_any,95):.1f}), count_ge={cnt_any}/{N_AGG}, agg p={(cnt_any+1)/(N_AGG+1):.4f}")

# --- CHECK 4: Holm-Bonferroni on the 5 museum-collapsed Wilcoxon tests ---
print("\n=== 4. HOLM-BONFERRONI ON THE 5 MUSEUM-COLLAPSED WILCOXON TESTS ===")
mus_df = dd[dd["museum"].notna() & (dd["museum"] != "") & (dd["museum"] != "nan")]
m_tests = [
    ("H1a circ", "h1_exc_circ"),
    ("Pure COLOR", "pure_col_diff"),
    ("Pure CLUSTER", "pure_cl_diff"),
    ("H3a color", "h3_col_exc"),
    ("H3b numeric", "h3_val_exc"),
]
raw_ps = []
m_info = []
for lbl, col in m_tests:
    meds = mus_df.groupby("museum")[col].median().dropna()
    p = float(stats.wilcoxon(meds).pvalue)
    raw_ps.append(p)
    m_info.append((lbl, len(meds), float(meds.median()), p))

# Holm step-down adjustment
order = np.argsort(raw_ps)
m_k = len(raw_ps)
holm = np.zeros(m_k)
running_max = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, raw_ps[idx] * (m_k - rank))
    running_max = max(running_max, adj)
    holm[idx] = running_max

for i, (lbl, nm, med, p) in enumerate(m_info):
    print(f"  {lbl:14s}: n_mus={nm:2d} | med={med:+.4f} | p_raw={p:.2e} | p_Holm={holm[i]:.2e}")
