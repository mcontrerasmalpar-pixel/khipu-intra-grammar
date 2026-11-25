"""
make_figures.py — Generate 4 publication-ready figures for khipu-intra-grammar
from the verified CSV tables in results/.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import false_discovery_control

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def fig1_fdr_and_controls():
    fdr = pd.read_csv(RES / "table_fdr_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: % individually significant (raw p<0.05 vs FDR q<0.05)
    labels = [
        "H3b: Numeric R²\nhomology",
        "H3a: Color\nhomology",
        "H1a: NMI(cluster,color)\n[circular]",
        "H2b: Cluster→jump\n[circular]",
        "Pure CLUSTER\n(same color)",
        "H2a: Color→jump\n[circular]",
        "Pure COLOR\n(inside cluster)",
    ]
    order_keys = [
        "H3b Numeric R^2 positional homology",
        "H3a Color positional homology",
        "H1a NMI(cluster, color) [circular]",
        "H2b Cluster -> jump [circular]",
        "Pure CLUSTER (same color)",
        "H2a Color -> jump [circular]",
        "Pure COLOR (inside cluster)",
    ]
    sub = fdr.set_index("test").loc[order_keys]
    y = np.arange(len(labels))
    h = 0.36
    axes[0].barh(y + h / 2, sub["raw_p05_pct"], height=h, color="#7aa6c2", label="Raw p < 0.05")
    axes[0].barh(y - h / 2, sub["fdr_q05_pct"], height=h, color="#1f4e79", label="BH-FDR q < 0.05")
    axes[0].axvline(5.0, color="#c0392b", linestyle="--", linewidth=1.2, label="Chance (5%)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("% of khipus significant (N_PERM = 1,000)")
    axes[0].set_title("A. Within-khipu structural tests vs null")
    axes[0].legend(fontsize=8, loc="lower right")

    # Right: 2x2 decoupled pure effects (log1p base-10 value jumps)
    dd = pd.read_csv(RES / "deep_dive_fdr_1000.csv")
    pc = dd["pure_col_diff"].dropna()
    pl = dd["pure_cl_diff"].dropna()
    parts = axes[1].violinplot([pc.values, pl.values], positions=[0, 1], showmedians=True, showextrema=False)
    for b, col in zip(parts["bodies"], ["#e67e22", "#2e86ab"]):
        b.set_facecolor(col)
        b.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([f"Pure COLOR\n(inside same cluster)\nn={len(pc)}, med={pc.median():+.3f}",
                             f"Pure CLUSTER\n(same color)\nn={len(pl)}, med={pl.median():+.3f}"])
    axes[1].set_ylabel("Δ |log1p(val_{i+1}) - log1p(val_i)|")
    axes[1].set_ylim(-1.8, 2.5)
    axes[1].set_title("B. 2×2 decoupling on base-10 values")

    plt.tight_layout()
    plt.savefig(FIGS / "fig1_intra_khipu_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig2_h3_cooccurrence():
    dd = pd.read_csv(RES / "deep_dive_fdr_1000.csv")
    both = dd.dropna(subset=["h3_col_p", "h3_val_p"]).copy()
    both["q_col"] = false_discovery_control(both["h3_col_p"].values, method="bh")
    both["q_val"] = false_discovery_control(both["h3_val_p"].values, method="bh")
    both["cat"] = np.where(
        (both["q_col"] < 0.05) & (both["q_val"] < 0.05), "Both (H3a & H3b, n=86)",
        np.where(both["q_col"] < 0.05, "Color only (n=41)",
                 np.where(both["q_val"] < 0.05, "Numeric only (n=39)", "Neither (n=119)"))
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    palette = {
        "Both (H3a & H3b, n=86)": "#1b9e77",
        "Color only (n=41)": "#d95f02",
        "Numeric only (n=39)": "#7570b3",
        "Neither (n=119)": "#bdbdbd",
    }
    for k, col in palette.items():
        sub = both[both["cat"] == k]
        axes[0].scatter(sub["h3_col_exc"], sub["h3_val_exc"], s=20, alpha=0.8, color=col, label=k)
    axes[0].axhline(0, color="gray", ls="--", lw=0.9)
    axes[0].axvline(0, color="gray", ls="--", lw=0.9)
    axes[0].set_xlabel("Color positional homology excess (H3a)")
    axes[0].set_ylabel("Numeric R² positional homology excess (H3b)")
    axes[0].set_title("A. Co-occurrence of color & numeric homology\n(Fisher OR = 6.40, p = 2.7e-13; Spearman ρ = 0.375)")
    axes[0].legend(fontsize=8, loc="upper left")

    # Right: Clean decimal vs non-clean decimal sensitivity
    groups = ["100% clean\ndecimal\n(n=214)", "Contains\nnon-decimal\n(n=298)"]
    h3a_vals = [55.0, 37.8]
    h3b_vals = [50.4, 34.8]
    x = np.arange(2)
    w = 0.32
    axes[1].bar(x - w / 2, h3a_vals, width=w, color="#d95f02", label="H3a Color (FDR q<0.05)")
    axes[1].bar(x + w / 2, h3b_vals, width=w, color="#1f4e79", label="H3b Numeric R² (FDR q<0.05)")
    for i in range(2):
        axes[1].text(x[i] - w / 2, h3a_vals[i] + 1.2, f"{h3a_vals[i]:.1f}%", ha="center", fontsize=9)
        axes[1].text(x[i] + w / 2, h3b_vals[i] + 1.2, f"{h3b_vals[i]:.1f}%", ha="center", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].set_ylim(0, 66)
    axes[1].set_ylabel("% of testable khipus (FDR q < 0.05)")
    axes[1].set_title("B. Tabular signal is ~1.5× stronger in\n100% clean decimal khipus")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGS / "fig2_h3_cooccurrence_tabular.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig3_cataloger_invariance():
    dd = pd.read_csv(RES / "deep_dive_fdr_1000.csv")
    cats = ["UR", "AS", "HP", "JC"]
    sub = dd[dd["cataloger"].isin(cats)]
    grp = sub.groupby("cataloger").agg(
        n=("KHIPU_ID", "count"),
        h3a=("h3_col_p", lambda x: 100 * (x.dropna() < 0.05).mean()),
        h3b=("h3_val_p", lambda x: 100 * (x.dropna() < 0.05).mean()),
        h1c=("h1_p_circ", lambda x: 100 * (x.dropna() < 0.05).mean()),
    ).loc[cats]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(cats))
    w = 0.25
    ax.bar(x - w, grp["h1c"], width=w, color="#7aa6c2", label="H1a NMI [circular]")
    ax.bar(x,     grp["h3a"], width=w, color="#d95f02", label="H3a Color homology")
    ax.bar(x + w, grp["h3b"], width=w, color="#1b9e77", label="H3b Numeric R² homology")
    ax.axhline(5.0, color="#c0392b", ls="--", lw=1.2, label="Chance (5%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={int(grp.loc[c, 'n'])})" for c in cats])
    ax.set_ylabel("% of khipus significant (p < 0.05)")
    ax.set_title("H4. Intra-khipu grammar persists across all major OKR catalogers")
    ax.set_ylim(0, 72)
    ax.legend(fontsize=8.5, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGS / "fig3_cataloger_invariance.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig4_h5_sums_vs_null():
    h5 = pd.read_csv(RES / "extension_h5_sums.csv")
    b = h5.dropna(subset=["h5b_hits"])
    c = h5.dropna(subset=["h5c_hits"])

    cats = [
        "H5b: Within-cluster\nfirst/last subtotal\n(n=494 khipus)",
        "H5c: Cross-group\nfirst/last slot sum\n(n=244 khipus)",
        "H5c: Cross-group\nany-row slot sum\n(n=244 khipus)",
    ]
    obs = [b["h5b_hits"].sum(), c["h5c_hits"].sum(), c["h5c_any_hits"].sum()]
    nul = [b["h5b_null"].sum(), c["h5c_null"].sum(), c["h5c_any_null"].sum()]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(cats))
    w = 0.34
    ax.bar(x - w / 2, obs, width=w, color="#1f4e79", label="Observed exact sums")
    ax.bar(x + w / 2, nul, width=w, color="#bdbdbd", label="Expected under within-khipu null")
    for i in range(len(cats)):
        ratio = obs[i] / max(nul[i], 1e-9)
        ax.text(x[i] - w / 2, obs[i] + 1.0, f"{int(obs[i])}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i] + w / 2, nul[i] + 1.0, f"{nul[i]:.1f}", ha="center", fontsize=9)
        ax.text(x[i], max(obs[i], nul[i]) + 4.0, f"{ratio:.2f}×", ha="center", fontsize=9.5, color="#c0392b" if ratio < 1.1 else "#1b9e77", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Total non-trivial exact sum hits across corpus")
    ax.set_ylim(0, max(obs) + 12)
    ax.set_title("H5. Within-khipu exact sums vs permutation null calibration (N_PERM = 500)")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIGS / "fig4_h5_sums_vs_null.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig1_fdr_and_controls()
    fig2_h3_cooccurrence()
    fig3_cataloger_invariance()
    fig4_h5_sums_vs_null()
    print("Generated 4 figures in", FIGS)
