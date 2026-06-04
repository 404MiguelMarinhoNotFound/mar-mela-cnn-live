"""Step 0 data audit / EDA.

Produces the human-readable tables and figures that justify (or challenge) the
pre-made splits before we train anything. The doctrine demands source-stratified
splits and per-source metrics; this module checks whether the splits we were handed
actually support that — class balance per split, source composition per split, a
source-leakage flag, resolution spread, and a corrupt-file list.

All figures are saved under ``cfg.figures_dir``; tables are printed and the key
ones returned for programmatic use.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works on Databricks / CI / CPU box
import matplotlib.pyplot as plt
import pandas as pd


def _crosstab(df: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    return pd.crosstab(df[index], df[columns], margins=True, margins_name="Total")


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of benign vs melanoma per split."""
    t = pd.crosstab(df["split"], df["class_dir"], margins=True, margins_name="Total")
    return t


def source_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of each source per split."""
    return _crosstab(df, "source", "split")


def leakage_report(df: pd.DataFrame) -> pd.DataFrame:
    """For each source, how many of the 3 splits it appears in.

    A source confined to a single split signals distribution shift across splits
    (the model may never see that source in training, or never be tested on it).
    This is a coarse proxy — true patient/lesion leakage needs IDs we don't have.
    """
    g = (
        df.groupby("source")["split"]
        .agg(n_splits=lambda s: s.nunique(), splits=lambda s: ",".join(sorted(s.unique())))
        .reset_index()
    )
    g["n_images"] = df.groupby("source").size().values
    g["single_split_warning"] = g["n_splits"] < 3
    return g.sort_values("n_images", ascending=False)


def corrupt_files(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["corrupt"]][["path", "split", "class_dir", "source"]]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, figures_dir: Path, name: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / name
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out


def fig_class_balance(df: pd.DataFrame, figures_dir: Path) -> Path:
    ct = pd.crosstab(df["split"], df["class_dir"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ct.plot(kind="bar", ax=ax)
    ax.set_title("Class balance per split")
    ax.set_xlabel("split")
    ax.set_ylabel("images")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, figures_dir, "class_balance.png")


def fig_source_composition(df: pd.DataFrame, figures_dir: Path) -> Path:
    ct = pd.crosstab(df["split"], df["source"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Source composition per split")
    ax.set_xlabel("split")
    ax.set_ylabel("images")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="source", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save(fig, figures_dir, "source_composition.png")


def fig_domain_by_class(df: pd.DataFrame, figures_dir: Path) -> Path:
    ct = pd.crosstab(df["domain"], df["class_dir"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ct.plot(kind="bar", ax=ax)
    ax.set_title("Domain (dermoscopic vs clinical) by class")
    ax.set_xlabel("domain")
    ax.set_ylabel("images")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, figures_dir, "domain_by_class.png")


def fig_resolution(df: pd.DataFrame, figures_dir: Path) -> Path | None:
    sub = df.dropna(subset=["width", "height"])
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(sub["width"], sub["height"], s=3, alpha=0.2)
    axes[0].set_title("Image dimensions")
    axes[0].set_xlabel("width")
    axes[0].set_ylabel("height")
    longest = sub[["width", "height"]].max(axis=1)
    axes[1].hist(longest, bins=50)
    axes[1].set_title("Longest side (px)")
    axes[1].set_xlabel("pixels")
    axes[1].set_ylabel("count")
    return _save(fig, figures_dir, "resolution.png")


def run_audit(cfg, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Print tables, write figures, return the key tables."""
    figures_dir = cfg.figures_dir

    tables = {
        "class_balance": class_balance(df),
        "source_composition": source_composition(df),
        "leakage": leakage_report(df),
        "corrupt": corrupt_files(df),
    }

    def banner(title: str) -> None:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

    banner("Manifest summary")
    print(f"total images : {len(df)}")
    print(f"by split     : {df['split'].value_counts().to_dict()}")
    print(f"by class     : {df['class_dir'].value_counts().to_dict()}")
    print(f"by domain    : {df['domain'].value_counts().to_dict()}")
    print(f"formats      : {df['format'].value_counts().to_dict()}")

    banner("Class balance per split")
    print(tables["class_balance"])

    banner("Source composition per split")
    print(tables["source_composition"])

    banner("Source leakage / single-split check")
    print(tables["leakage"].to_string(index=False))
    warned = tables["leakage"]["single_split_warning"].sum()
    if warned:
        print(f"\n[!] {warned} source(s) not present in all 3 splits "
              f"(distribution-shift risk).")

    banner("Corrupt files")
    n_corrupt = len(tables["corrupt"])
    print(f"{n_corrupt} corrupt/unreadable file(s).")
    if n_corrupt:
        print(tables["corrupt"].to_string(index=False))

    figs = [
        fig_class_balance(df, figures_dir),
        fig_source_composition(df, figures_dir),
        fig_domain_by_class(df, figures_dir),
        fig_resolution(df, figures_dir),
    ]
    banner("Figures written")
    for f in figs:
        if f is not None:
            print(f" - {f}")

    return tables
