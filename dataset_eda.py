"""
Dataset metadata analysis and class distribution visualization.
Generates a single PNG with horizontal bar charts for each feature.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV_PATH = "data/metadata.csv"
OUTPUT_PATH = "dataset_eda.png"

PALETTE = {"Benign": "#4C9BE8", "Malignant": "#E8654C", "NaN / Indeterminate": "#BBBBBB"}
TITLE_FONTSIZE = 12
LABEL_FONTSIZE = 9


def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df_labeled = df[df["diagnosis_1"].isin(["Benign", "Malignant"])].copy()
    return df, df_labeled


def hbar_single(ax, series: pd.Series, title: str, color: str = "#4C9BE8"):
    """Single-group horizontal bar chart (overall counts)."""
    series = series.dropna().value_counts().sort_values()
    ax.barh(series.index, series.values, color=color, edgecolor="white", linewidth=0.5)
    for i, (label, val) in enumerate(zip(series.index, series.values)):
        ax.text(val + series.values.max() * 0.01, i, f"{val:,}", va="center", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, series.values.max() * 1.15)


def hbar_stacked(ax, df: pd.DataFrame, col: str, title: str, top_n: int = 10):
    """Stacked horizontal bar chart: Benign vs Malignant per category."""
    grouped = (
        df.groupby([col, "diagnosis_1"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Benign", "Malignant"], fill_value=0)
    )
    if len(grouped) > top_n:
        grouped = grouped.loc[grouped.sum(axis=1).nlargest(top_n).index]
    grouped = grouped.loc[grouped.sum(axis=1).sort_values().index]

    benign_vals = grouped["Benign"].values
    malignant_vals = grouped["Malignant"].values
    y = np.arange(len(grouped))

    ax.barh(y, benign_vals, color=PALETTE["Benign"], label="Benign", edgecolor="white", linewidth=0.4)
    ax.barh(y, malignant_vals, left=benign_vals, color=PALETTE["Malignant"], label="Malignant",
            edgecolor="white", linewidth=0.4)

    totals = benign_vals + malignant_vals
    max_total = totals.max()
    for i, (b, m, t) in enumerate(zip(benign_vals, malignant_vals, totals)):
        ax.text(t + max_total * 0.01, i, f"{t:,}", va="center", fontsize=LABEL_FONTSIZE)

    ax.set_yticks(y)
    ax.set_yticklabels(grouped.index, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max_total * 1.15)


def age_distribution(ax, df: pd.DataFrame):
    """Age group breakdown as stacked horizontal bar."""
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 120]
    labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    df = df.copy()
    df["age_group"] = pd.cut(df["age_approx"], bins=bins, labels=labels, right=False)
    hbar_stacked(ax, df.dropna(subset=["age_group"]), "age_group",
                 "Age Group Distribution (Benign vs Malignant)")


def summary_stats(df: pd.DataFrame, df_labeled: pd.DataFrame):
    total = len(df)
    labeled = len(df_labeled)
    dropped = total - labeled
    benign = (df_labeled["diagnosis_1"] == "Benign").sum()
    malignant = (df_labeled["diagnosis_1"] == "Malignant").sum()
    print(f"Total samples      : {total:,}")
    print(f"Labeled (used)     : {labeled:,}  (dropped {dropped:,} NaN/Indeterminate)")
    print(f"  Benign           : {benign:,}  ({benign/labeled*100:.1f}%)")
    print(f"  Malignant        : {malignant:,}  ({malignant/labeled*100:.1f}%)")


def main():
    df, df_labeled = load_data(CSV_PATH)
    summary_stats(df, df_labeled)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        f"ISIC Skin Lesion Dataset — Class Distribution\n"
        f"Total: {len(df):,} samples  |  Labeled: {len(df_labeled):,}  "
        f"(Benign {(df_labeled['diagnosis_1']=='Benign').sum():,} / "
        f"Malignant {(df_labeled['diagnosis_1']=='Malignant').sum():,})",
        fontsize=14, fontweight="bold", y=1.01
    )

    # 1. Overall diagnosis (includes NaN/Indeterminate)
    diag_counts = df["diagnosis_1"].fillna("NaN / Indeterminate").value_counts()
    colors_diag = [PALETTE.get(k, "#BBBBBB") for k in diag_counts.index]
    ax = axes[0, 0]
    ax.barh(diag_counts.index, diag_counts.values, color=colors_diag, edgecolor="white")
    for i, (label, val) in enumerate(zip(diag_counts.index, diag_counts.values)):
        ax.text(val + diag_counts.values.max() * 0.01, i, f"{val:,}", va="center", fontsize=LABEL_FONTSIZE)
    ax.set_title("Overall Diagnosis Distribution (all samples)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, diag_counts.values.max() * 1.15)

    # 2. Sex breakdown
    hbar_stacked(axes[0, 1], df_labeled.dropna(subset=["sex"]), "sex",
                 "Sex Distribution (Benign vs Malignant)")

    # 3. Anatomical site
    hbar_stacked(axes[1, 0], df_labeled.dropna(subset=["anatom_site_1"]), "anatom_site_1",
                 "Anatomical Site Distribution (Benign vs Malignant)")

    # 4. Age group
    age_distribution(axes[1, 1], df_labeled)

    # 5. Diagnosis confirmation type
    hbar_stacked(axes[2, 0], df_labeled.dropna(subset=["diagnosis_confirm_type"]),
                 "diagnosis_confirm_type",
                 "Diagnosis Confirmation Type (Benign vs Malignant)")

    # 6. Melanocytic status
    hbar_stacked(axes[2, 1], df_labeled.dropna(subset=["melanocytic"]), "melanocytic",
                 "Melanocytic Status (Benign vs Malignant)")

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=PALETTE["Benign"], label="Benign"),
        mpatches.Patch(color=PALETTE["Malignant"], label="Malignant"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
