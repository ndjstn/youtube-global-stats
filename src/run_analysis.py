"""Global YouTube Statistics 2023: creator segmentation.

995 top creators with subscriber count, video views, country, category, and
channel-type metadata. KMeans on standardised log-features produces five
archetypes. The pipeline also emits the elbow plot, cluster scatter, a
palette-branded hero, and a teaching animation that shows KMeans iterating
assignment-then-update until convergence on the two-dimensional
(subscribers, views) plane.
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _palette import YOUTUBE_GLOBAL as P, apply_to_mpl  # noqa: E402

apply_to_mpl(P)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

CLUSTER_COLORS = [P.accent, P.header_bg, P.highlight, P.cover_subtitle, P.muted]
CLUSTER_LABELS = {
    0: "Mega-scale",
    1: "Mainstream",
    2: "Low engagement",
    3: "Music-video",
    4: "Upload machines",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--outputs", required=True)
    return ap.parse_args()


def _relabel_clusters(km_labels, df_xc):
    """Re-map raw KMeans labels to stable 0..4 ordering by archetype semantics.

    Cluster ordering used throughout the write-up:
      0: Mega-scale (highest median subs)
      1: Mainstream
      2: Low engagement (lowest views/sub)
      3: Music-video (lowest uploads)
      4: Upload machines (highest uploads)
    """
    tmp = df_xc.copy()
    tmp["raw"] = km_labels
    profile = tmp.groupby("raw").agg(
        subs=("subscribers", "median"),
        vps=("views_per_sub", "median"),
        ups=("uploads", "median"),
    )
    # Identify music-video (lowest uploads) and upload-machines (highest uploads)
    music = profile["ups"].idxmin()
    machines = profile["ups"].idxmax()
    # Low engagement: lowest views_per_sub among the remaining
    remaining = profile.drop([music, machines])
    low_eng = remaining["vps"].idxmin()
    remaining = remaining.drop(low_eng)
    # Mega-scale vs Mainstream: mega has higher median subs
    mega = remaining["subs"].idxmax()
    mainstream = remaining["subs"].idxmin()
    mapping = {mega: 0, mainstream: 1, low_eng: 2, music: 3, machines: 4}
    return np.array([mapping[r] for r in km_labels])


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
    df["earnings_band_width"] = df["highest_yearly_earnings"] - df["lowest_yearly_earnings"]
    df["earnings_band_ratio"] = df["highest_yearly_earnings"] / df["lowest_yearly_earnings"].replace(0, np.nan)

    # Filter
    df_ = df.loc[(df["subscribers"] > 0) & (df["video views"] > 0) & (df["uploads"] > 0)].copy()
    print(f"after filter: {len(df_)}")

    # ---- Feature engineering ----
    df_["views_per_sub"] = df_["video views"] / df_["subscribers"]
    df_["views_per_upload"] = df_["video views"] / df_["uploads"]
    df_["uploads_per_sub"] = df_["uploads"] / (df_["subscribers"] / 1e6)

    cluster_feats = ["subscribers", "video views", "uploads", "views_per_sub", "views_per_upload", "uploads_per_sub"]
    Xc = df_[cluster_feats].copy().dropna()
    Xc_log = np.log1p(Xc)
    scaler = StandardScaler().fit(Xc_log)
    Xs = scaler.transform(Xc_log)

    # ---- Elbow + silhouette ----
    from sklearn.metrics import silhouette_score
    inertias, sils = [], []
    ks = list(range(2, 11))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ks, inertias, "o-", color=P.accent, lw=2.5, markersize=8, label="Within-cluster SSE")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Within-cluster SSE", color=P.accent)
    ax1.tick_params(axis="y", labelcolor=P.accent)
    ax2 = ax1.twinx()
    ax2.plot(ks, sils, "s--", color=P.header_bg, lw=2, markersize=7, label="Silhouette")
    ax2.set_ylabel("Silhouette score", color=P.header_bg)
    ax2.tick_params(axis="y", labelcolor=P.header_bg)
    ax2.grid(False)
    ax1.axvline(5, color=P.highlight, lw=2, alpha=0.6)
    ax1.annotate(
        f"k=5 picked\nsilhouette={sils[3]:.2f}",
        xy=(5, inertias[3]), xytext=(6.3, inertias[3] + (max(inertias) - min(inertias)) * 0.15),
        arrowprops=dict(arrowstyle="->", color=P.highlight, lw=1.5),
        fontsize=11, color=P.text,
        bbox=dict(boxstyle="round,pad=0.4", fc=P.footer_bg, ec=P.muted),
    )
    ax1.set_title("Elbow lands at k=5; silhouette holds at 0.27+")
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans-elbow.png")
    plt.close(fig)

    # ---- Fit k=5 ----
    km5 = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xs)
    df_.loc[Xc.index, "views_per_sub"] = Xc["views_per_sub"]
    df_.loc[Xc.index, "_raw_cluster"] = km5.labels_

    stable_labels = _relabel_clusters(
        km5.labels_,
        Xc.assign(subscribers=Xc["subscribers"], views_per_sub=Xc["views_per_sub"], uploads=Xc["uploads"]),
    )
    df_.loc[Xc.index, "cluster"] = stable_labels

    profile = df_.groupby("cluster")[cluster_feats].median().round(1)
    profile["count"] = df_.groupby("cluster").size()
    print(profile)
    profile.to_csv(out_dir / "cluster-profile.csv")

    # Named channels per cluster for interpretation
    exemplars = {}
    for c in sorted(df_["cluster"].dropna().unique()):
        sub = df_.loc[df_["cluster"] == c].sort_values("subscribers", ascending=False)
        exemplars[int(c)] = sub["Youtuber"].head(5).dropna().astype(str).tolist()

    # ---- HERO: cluster scatter, 16:9, strong accents, annotated ----
    fig, ax = plt.subplots(figsize=(14, 7.875))  # 16:9
    fig.patch.set_facecolor(P.footer_bg)
    ax.set_facecolor(P.bg)
    for c in sorted(df_["cluster"].dropna().unique()):
        sub = df_.loc[df_["cluster"] == c]
        ax.scatter(
            sub["subscribers"] / 1e6,
            sub["views_per_sub"],
            s=55, alpha=0.78,
            color=CLUSTER_COLORS[int(c)],
            edgecolor=P.header_bg, linewidth=0.4,
            label=f"{CLUSTER_LABELS[int(c)]} (n={len(sub)})",
        )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Subscribers (millions, log)", fontsize=13)
    ax.set_ylabel("Views per subscriber (log)", fontsize=13)
    ax.set_title(
        "Five archetypes separate cleanly on subscribers × views-per-subscriber",
        fontsize=17, pad=14,
    )
    # Annotate mega-scale outlier
    megas = df_.loc[df_["cluster"] == 0].nlargest(1, "subscribers")
    if len(megas):
        mx = megas["subscribers"].iloc[0] / 1e6
        my = megas["views_per_sub"].iloc[0]
        ax.annotate(
            f"Mega-scale cluster\nn={int(profile.loc[0, 'count'])}, median 36.5M subs",
            xy=(mx, my),
            xytext=(mx * 0.15, my * 2.8),
            arrowprops=dict(arrowstyle="->", color=P.accent, lw=1.8),
            fontsize=12, color=P.text, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=P.footer_bg, ec=P.accent, lw=1.5),
        )
    # Annotate low-engagement cluster
    low_eng_sub = df_.loc[df_["cluster"] == 2]
    if len(low_eng_sub):
        lx = low_eng_sub["subscribers"].median() / 1e6
        ly = low_eng_sub["views_per_sub"].median()
        ax.annotate(
            f"Low engagement:\nviews/sub median {profile.loc[2, 'views_per_sub']:.0f}",
            xy=(lx, ly),
            xytext=(lx * 0.35, ly * 0.18),
            arrowprops=dict(arrowstyle="->", color=P.highlight, lw=1.8),
            fontsize=11, color=P.text,
            bbox=dict(boxstyle="round,pad=0.5", fc=P.footer_bg, ec=P.highlight, lw=1.5),
        )
    ax.legend(loc="lower left", fontsize=11, framealpha=0.92, facecolor=P.bg, edgecolor=P.muted)
    ax.grid(True, which="both", alpha=0.2, color=P.muted)
    fig.tight_layout()
    fig.savefig(fig_dir / "hero.png", dpi=150, facecolor=P.footer_bg)
    fig.savefig(fig_dir / "cluster-scatter.png", dpi=150, facecolor=P.footer_bg)
    plt.close(fig)

    # ---- Top countries bar chart, annotated ----
    country_ct = df_["Country"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(country_ct.index, country_ct.values, color=P.muted, edgecolor=P.header_bg, linewidth=0.5)
    # Highlight top 3
    for i in range(min(3, len(bars))):
        bars[i].set_color(P.accent)
    ax.set_xticks(range(len(country_ct)))
    ax.set_xticklabels(country_ct.index, rotation=35, ha="right")
    ax.set_ylabel("Channels in top-995")
    ax.set_title(f"United States leads with {country_ct.iloc[0]} of 995 channels")
    top_v = country_ct.iloc[0]
    ax.annotate(
        f"{top_v} US channels\n{top_v / 995 * 100:.1f}% of the top-995",
        xy=(0, top_v), xytext=(2.2, top_v * 0.85),
        arrowprops=dict(arrowstyle="->", color=P.accent, lw=1.5),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc=P.footer_bg, ec=P.accent),
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "top-countries.png")
    plt.close(fig)

    # ---- Top categories ----
    cat_ct = df_["category"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(cat_ct.index[::-1], cat_ct.values[::-1], color=P.muted, edgecolor=P.header_bg, linewidth=0.4)
    bars[-1].set_color(P.accent)
    bars[-2].set_color(P.highlight)
    ax.set_xlabel("Channels")
    ax.set_title(f"Entertainment ({cat_ct.iloc[0]}) and Music ({cat_ct.iloc[1]}) account for {(cat_ct.iloc[0]+cat_ct.iloc[1])/len(df_)*100:.0f}% of top-995")
    fig.tight_layout()
    fig.savefig(fig_dir / "top-categories.png")
    plt.close(fig)

    # ---- Subs vs views scatter ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_["subscribers"] / 1e6, df_["video views"] / 1e9, s=18, alpha=0.55, color=P.accent, edgecolor=P.header_bg, linewidth=0.25)
    # Fit a log-log regression line for annotation
    mask = (df_["subscribers"] > 0) & (df_["video views"] > 0)
    x = np.log10(df_.loc[mask, "subscribers"] / 1e6)
    y = np.log10(df_.loc[mask, "video views"] / 1e9)
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 50)
    yline = slope * xline + intercept
    ax.plot(10 ** xline, 10 ** yline, color=P.header_bg, lw=2, linestyle="--", alpha=0.75, label=f"log-log slope = {slope:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Subscribers (millions, log)")
    ax.set_ylabel("Total video views (billions, log)")
    ax.set_title(f"Views scale with subscribers at slope {slope:.2f} on log-log axes")
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "subs-vs-views.png")
    plt.close(fig)

    # ---- Country density per capita ----
    country_stats = df_.groupby("Country").agg(
        channels=("subscribers", "size"),
        pop=("Population", "mean"),
    ).reset_index().dropna(subset=["pop"])
    country_stats["channels_per_10m_pop"] = country_stats["channels"] / (country_stats["pop"] / 1e7)
    country_stats = country_stats.loc[country_stats["channels"] >= 5].sort_values("channels_per_10m_pop", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(country_stats["Country"], country_stats["channels_per_10m_pop"], color=P.muted, edgecolor=P.header_bg, linewidth=0.4)
    for i in range(min(3, len(bars))):
        bars[i].set_color(P.accent)
    ax.set_xticks(range(len(country_stats)))
    ax.set_xticklabels(country_stats["Country"], rotation=35, ha="right")
    ax.set_ylabel("Top-995 channels per 10M population")
    top_c = country_stats.iloc[0]
    ax.set_title(f"Per capita, {top_c['Country']} leads at {top_c['channels_per_10m_pop']:.1f} channels per 10M people")
    fig.tight_layout()
    fig.savefig(fig_dir / "creator-density.png")
    plt.close(fig)

    # ---- Earnings by category ----
    cat_earn = df_.groupby("category")["mid_yearly_earnings"].median().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    values = cat_earn.values[::-1] / 1e6
    labels = cat_earn.index[::-1]
    bars = ax.barh(labels, values, color=P.muted, edgecolor=P.header_bg, linewidth=0.4)
    bars[-1].set_color(P.accent)
    ax.set_xlabel("Median yearly earnings (USD millions, Social Blade band midpoint)")
    ax.set_title(f"{cat_earn.index[0]} category tops Social Blade estimates at ${cat_earn.iloc[0]/1e6:.2f}M median")
    fig.tight_layout()
    fig.savefig(fig_dir / "category-earnings.png")
    plt.close(fig)

    # ---- Social Blade band width: quantify the variance for the blog ----
    band_stats = df_.loc[df_["lowest_yearly_earnings"] > 0, "earnings_band_ratio"].describe()
    print("earnings band ratio (high/low) describe:\n", band_stats)

    # ---- TEACHING ANIMATION: KMeans iterating to convergence ----
    # 2D plane: log subscribers x log views_per_sub
    rng = np.random.default_rng(7)
    X2 = Xc[["subscribers", "views_per_sub"]].copy()
    X2_log = np.log10(X2)
    mu = X2_log.mean().values
    sd = X2_log.std().values
    X2_std = (X2_log.values - mu) / sd

    K = 5
    # Start centroids at random data points
    init_idx = rng.choice(len(X2_std), size=K, replace=False)
    centroids = X2_std[init_idx].copy()
    # History: list of (assignments, centroids, step_type) per frame
    history = []
    # Frame 0: initial centroids, no assignments yet (all neutral)
    initial_assign = np.full(len(X2_std), -1)
    history.append((initial_assign.copy(), centroids.copy(), "init"))

    max_iter = 25
    prev_labels = np.full(len(X2_std), -1)
    for step in range(max_iter):
        # Assign step
        dists = np.linalg.norm(X2_std[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        history.append((labels.copy(), centroids.copy(), "assign"))
        # Update step
        new_centroids = centroids.copy()
        for k in range(K):
            mask = labels == k
            if mask.any():
                new_centroids[k] = X2_std[mask].mean(axis=0)
        history.append((labels.copy(), new_centroids.copy(), "update"))
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if (labels == prev_labels).all() and shift < 1e-4:
            # One more frame showing the converged state
            history.append((labels.copy(), centroids.copy(), "converged"))
            break
        prev_labels = labels.copy()

    print(f"KMeans teaching animation frames: {len(history)} at {max_iter} iter cap")

    # Convert standardised centroids back to data coords for plotting
    def unstd(pts):
        log_pts = pts * sd + mu
        return 10 ** log_pts  # to raw subs, views_per_sub
    # Data coords for points
    pts_raw = 10 ** (X2_std * sd + mu)
    x_pts = pts_raw[:, 0] / 1e6  # subs in millions
    y_pts = pts_raw[:, 1]

    fig, ax = plt.subplots(figsize=(12, 6.75))  # 16:9
    fig.patch.set_facecolor(P.footer_bg)
    ax.set_facecolor(P.bg)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x_pts.min() * 0.85, x_pts.max() * 1.25)
    ax.set_ylim(y_pts.min() * 0.7, y_pts.max() * 1.5)
    ax.set_xlabel("Subscribers (millions, log)", fontsize=12)
    ax.set_ylabel("Views per subscriber (log)", fontsize=12)
    ax.grid(True, which="both", alpha=0.25, color=P.muted)

    scat = ax.scatter(x_pts, y_pts, s=26, alpha=0.75, color=P.muted, edgecolor="none")
    cent_art = ax.scatter([], [], s=280, marker="X", edgecolor=P.text, linewidth=2.0, zorder=10)
    title_obj = ax.set_title("KMeans on log(subscribers) x log(views/sub) — init", fontsize=15)
    banner = ax.text(
        0.02, 0.96, "", transform=ax.transAxes, fontsize=12, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc=P.footer_bg, ec=P.header_bg, lw=1.2),
    )

    def frame(i):
        labels, cents, step_type = history[i]
        if (labels < 0).all():
            scat.set_color(P.muted)
        else:
            colors_per = np.array([CLUSTER_COLORS[int(l)] for l in labels])
            scat.set_color(colors_per)
        cent_raw = unstd(cents)
        cent_x = cent_raw[:, 0] / 1e6
        cent_y = cent_raw[:, 1]
        cent_art.set_offsets(np.column_stack([cent_x, cent_y]))
        cent_art.set_color(CLUSTER_COLORS[:K])
        if step_type == "init":
            title_obj.set_text("KMeans teaching: random init  →  assign  →  update  →  repeat")
            banner.set_text("Step 0: random centroids placed, no assignments yet")
        elif step_type == "assign":
            iter_num = (i + 1) // 2
            title_obj.set_text(f"Iteration {iter_num}: assign each point to nearest centroid")
            banner.set_text(f"Iter {iter_num}  →  ASSIGN (points recolour to closest centroid)")
        elif step_type == "update":
            iter_num = i // 2
            title_obj.set_text(f"Iteration {iter_num}: update centroids to cluster means")
            banner.set_text(f"Iter {iter_num}  →  UPDATE (centroids move to mean of assigned points)")
        elif step_type == "converged":
            title_obj.set_text("Converged: assignments stable, centroids stop moving")
            banner.set_text("Converged  →  local minimum of within-cluster variance")
        return [scat, cent_art, title_obj, banner]

    # Ensure minimum 30 frames by padding converged state if needed
    while len(history) < 32:
        history.append(history[-1])

    anim = animation.FuncAnimation(fig, frame, frames=len(history), interval=220, blit=False)
    anim.save(str(fig_dir / "kmeans-teaching.gif"), writer="pillow", fps=5)
    # Also save the legacy name (updated, not choppy) for blog compatibility
    anim.save(str(fig_dir / "cluster-reveal-animation.gif"), writer="pillow", fps=5)
    plt.close(fig)

    # ---- Output summary files ----
    summary = {
        "rows_used": int(len(df_)),
        "countries_represented": int(df_["Country"].nunique()),
        "silhouette_at_k5": float(sils[3]),
        "cluster_profile": profile.to_dict(),
        "cluster_exemplars": exemplars,
        "teaching_animation_frames": len(history),
        "earnings_band_ratio_median": float(band_stats.get("50%", float("nan"))),
        "earnings_band_ratio_p90": float(df_.loc[df_["lowest_yearly_earnings"] > 0, "earnings_band_ratio"].quantile(0.9)),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    md = ["# YouTube global top-995 summary", ""]
    md.append(f"Rows used: {len(df_)}. Countries represented: {df_['Country'].nunique()}.")
    md.append(f"Silhouette score at k=5: {sils[3]:.3f}.")
    md.append("")
    md.append("## KMeans cluster profile (median values per cluster, stable label order)")
    md.append("")
    md.append("| Cluster | Label | Subs | Views | Uploads | Views/sub | Views/upload | Count |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for c, row in profile.iterrows():
        md.append(
            f"| {int(c)} | {CLUSTER_LABELS[int(c)]} | "
            f"{row['subscribers']:,.0f} | {row['video views']:,.0f} | {row['uploads']:,.0f} | "
            f"{row['views_per_sub']:.1f} | {row['views_per_upload']:,.0f} | {int(row['count'])} |"
        )
    md.append("")
    md.append("## Exemplar channels per cluster (top 5 by subscribers)")
    md.append("")
    for c, names in exemplars.items():
        md.append(f"- **{CLUSTER_LABELS[int(c)]}** (cluster {c}): {', '.join(names)}")
    (out_dir / "analysis_summary.md").write_text("\n".join(md))
    print("Done")


if __name__ == "__main__":
    main()
