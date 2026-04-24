"""Global YouTube Statistics 2023: creator segmentation.

995 top creators with subscriber count, video views, country, category, and
channel-type metadata. The level-up is KMeans clustering on standardised metrics
to find natural creator archetypes, plus country-density and category-earnings
views. Data is a mid-2023 snapshot; treat it that way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _palette import YOUTUBE_GLOBAL as P, apply_to_mpl  # noqa: E402

sns.set_style("whitegrid")
apply_to_mpl(P)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 11})


def _cmap_native():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("project", [P.bg, P.muted, P.accent, P.header_bg])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--outputs", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    fig_dir, out_dir = Path(args.figures), Path(args.outputs)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, encoding="latin-1")
    print(f"rows: {len(df)}  cols: {len(df.columns)}")

    # Parse numeric columns
    for c in ["subscribers", "video views", "uploads"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lowest_yearly_earnings", "highest_yearly_earnings"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mid_yearly_earnings"] = (df["lowest_yearly_earnings"] + df["highest_yearly_earnings"]) / 2

    # Filter
    df_ = df.loc[(df["subscribers"] > 0) & (df["video views"] > 0) & (df["uploads"] > 0)].copy()
    print(f"after filter: {len(df_)}")

    # Top countries bar chart
    country_ct = df_["Country"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(country_ct.index, country_ct.values, color=P.accent)
    ax.set_xticklabels(country_ct.index, rotation=45, ha="right")
    ax.set_ylabel("Channels in top-995")
    ax.set_title("Top 15 countries by count of channels in the global top-995")
    fig.tight_layout()
    fig.savefig(fig_dir / "top-countries.png")
    plt.close(fig)

    # Top categories bar
    cat_ct = df_["category"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(cat_ct.index[::-1], cat_ct.values[::-1], color=P.highlight)
    ax.set_xlabel("Channels")
    ax.set_title("Top 15 categories among the global top-995")
    fig.tight_layout()
    fig.savefig(fig_dir / "top-categories.png")
    plt.close(fig)

    # Log-scale subs vs views scatter
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(df_["subscribers"] / 1e6, df_["video views"] / 1e9, s=14, alpha=0.6, color=P.accent)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Subscribers (millions, log)")
    ax.set_ylabel("Total video views (billions, log)")
    ax.set_title("Subscribers vs. lifetime video views (log-log)")
    fig.tight_layout()
    fig.savefig(fig_dir / "subs-vs-views.png")
    plt.close(fig)

    # Feature engineering for clustering
    df_["views_per_sub"] = df_["video views"] / df_["subscribers"]
    df_["views_per_upload"] = df_["video views"] / df_["uploads"]
    df_["uploads_per_sub"] = df_["uploads"] / (df_["subscribers"] / 1e6)  # uploads per million subs

    cluster_feats = ["subscribers", "video views", "uploads", "views_per_sub", "views_per_upload", "uploads_per_sub"]
    Xc = df_[cluster_feats].copy().dropna()
    # Log-transform for clustering stability
    Xc_log = np.log1p(Xc)
    scaler = StandardScaler().fit(Xc_log)
    Xs = scaler.transform(Xc_log)

    # Elbow + pick k=5
    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(2, 11), inertias, "o-", color=P.accent, lw=2)
    ax.set_xlabel("k"); ax.set_ylabel("Within-cluster SSE")
    ax.set_title("KMeans elbow: k=5 sits at the bend")
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans-elbow.png")
    plt.close(fig)

    km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xs)
    df_.loc[Xc.index, "cluster"] = km.labels_
    profile = df_.groupby("cluster")[cluster_feats].median().round(1)
    profile["count"] = df_.groupby("cluster").size()
    print(profile)
    profile.to_csv(out_dir / "cluster-profile.csv")

    # Cluster scatter on two log axes: subs vs views_per_sub (engagement efficiency)
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_colors = [P.accent, P.header_bg, P.highlight, P.muted, P.cover_subtitle]
    for c in sorted(df_["cluster"].dropna().unique()):
        sub = df_.loc[df_["cluster"] == c]
        ax.scatter(sub["subscribers"] / 1e6, sub["views_per_sub"], s=22, alpha=0.7, color=cluster_colors[int(c)], label=f"Cluster {int(c)} (n={len(sub)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Subscribers (millions, log)")
    ax.set_ylabel("Views per subscriber (log)")
    ax.set_title("KMeans creator segments on subs × views-per-subscriber")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "cluster-scatter.png")
    plt.close(fig)

    # Country-density: subscriber concentration per million population
    country_stats = df_.groupby("Country").agg(
        channels=("subscribers", "size"),
        pop=("Population", "mean"),
        total_subs=("subscribers", "sum"),
    ).reset_index()
    country_stats = country_stats.dropna(subset=["pop"])
    country_stats["channels_per_10m_pop"] = country_stats["channels"] / (country_stats["pop"] / 1e7)
    country_stats = country_stats.loc[country_stats["channels"] >= 5].sort_values("channels_per_10m_pop", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(country_stats["Country"], country_stats["channels_per_10m_pop"], color=P.highlight)
    ax.set_xticklabels(country_stats["Country"], rotation=45, ha="right")
    ax.set_ylabel("Top-995 channels per 10M population")
    ax.set_title("Creator density per capita (countries with 5+ top-995 channels)")
    fig.tight_layout()
    fig.savefig(fig_dir / "creator-density.png")
    plt.close(fig)

    # Earnings distribution by category
    cat_earn = df_.groupby("category")["mid_yearly_earnings"].median().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.barh(cat_earn.index[::-1], cat_earn.values[::-1] / 1e6, color=P.accent)
    ax.set_xlabel("Median yearly earnings (USD, midpoint of reported band)")
    ax.set_title("Estimated median yearly earnings by category (midpoint)")
    fig.tight_layout()
    fig.savefig(fig_dir / "category-earnings.png")
    plt.close(fig)

    # Animation: a small tour showing cluster centroids drawn one at a time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(df_["subscribers"].min() / 1e6 * 0.9, df_["subscribers"].max() / 1e6 * 1.1)
    ax.set_ylim(0.1, df_["views_per_sub"].max() * 1.1)
    ax.set_xlabel("Subscribers (millions, log)")
    ax.set_ylabel("Views per subscriber (log)")
    ax.set_title("")
    scatter_all = ax.scatter(df_["subscribers"] / 1e6, df_["views_per_sub"], s=14, color=P.muted, alpha=0.35, zorder=1)
    highlighted_artists = []
    cluster_order = sorted(df_["cluster"].dropna().unique(), key=lambda c: df_.loc[df_["cluster"] == c, "subscribers"].median())

    def animate(i):
        ax.set_title(f"Cluster {i+1} of {len(cluster_order)}")
        for a in highlighted_artists:
            a.set_alpha(0.4)
        cur = cluster_order[i]
        sub = df_.loc[df_["cluster"] == cur]
        art = ax.scatter(sub["subscribers"] / 1e6, sub["views_per_sub"], s=40, color=cluster_colors[int(cur)], edgecolor=P.text, linewidth=0.8, zorder=3, label=f"cluster {int(cur)} n={len(sub)}")
        highlighted_artists.append(art)
        return [scatter_all, *highlighted_artists]

    anim = animation.FuncAnimation(fig, animate, frames=len(cluster_order), interval=1200, blit=False)
    anim.save(str(fig_dir / "cluster-reveal-animation.gif"), writer="pillow", fps=1)
    plt.close(fig)

    summary = {
        "rows_used": int(len(df_)),
        "countries_represented": int(df_["Country"].nunique()),
        "cluster_profile": profile.to_dict(),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    md = ["# YouTube global top-995 summary", ""]
    md.append(f"Rows used: {len(df_)}. Countries represented: {df_['Country'].nunique()}.")
    md.append("")
    md.append("## KMeans cluster profile (median values per cluster)")
    md.append("")
    md.append("| Cluster | Subs | Views | Uploads | Views/sub | Views/upload | Count |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for c, row in profile.iterrows():
        md.append(f"| {int(c)} | {row['subscribers']:,.0f} | {row['video views']:,.0f} | {row['uploads']:,.0f} | {row['views_per_sub']:.1f} | {row['views_per_upload']:,.0f} | {int(row['count'])} |")
    (out_dir / "analysis_summary.md").write_text("\n".join(md))
    print("Done")


if __name__ == "__main__":
    main()
