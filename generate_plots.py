from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 300, "font.size": 10})


def plot_forecast_comparison():
    df = pd.read_csv(REPORTS / "forecast_comparison.csv")
    cols = {c.lower(): c for c in df.columns}
    model_col = cols.get("model", df.columns[0])
    mae_col = next(c for c in df.columns if "mae" in c.lower())
    r2_col = next(c for c in df.columns if "r2" in c.lower() or "r²" in c.lower())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models = df[model_col].astype(str)

    axes[0].bar(models, df[mae_col], color=["#1f77b4", "#4c94c8", "#7fb3d5", "#b2d1e5"])
    axes[0].set_title("Forecasting Error (MAE - Lower is Better)")
    axes[0].set_ylabel("MAE (kWh)")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(df[mae_col]):
        axes[0].text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=8)

    axes[1].bar(models, df[r2_col], color=["#2ca02c", "#5ab85a", "#88cf88", "#b6e3b6"])
    axes[1].set_title("Forecasting Fit (R² - Higher is Better)")
    axes[1].set_ylabel("R² Score")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(df[r2_col]):
        axes[1].text(i, v + 0.015, f"{v:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig1_forecast_comparison.png", bbox_inches="tight")
    plt.close()


def plot_f1_precision_recall():
    df = pd.read_csv(REPORTS / "threshold_ablation_results.csv")
    sub = df[(df["method"] == "percentile") & (df["param"] == "p=97.5")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Precision, Recall, F1 across routine groups (p=97.5)
    x = np.arange(len(sub))
    w = 0.25
    axes[0].bar(x - w, sub["syn_precision"], width=w, label="Precision", color="#1f77b4")
    axes[0].bar(x, sub["syn_recall"], width=w, label="Recall", color="#ff7f0e")
    axes[0].bar(x + w, sub["syn_f1"], width=w, label="F1-Score", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sub["group"])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision, Recall & F1 by Routine Group (p=97.5)")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: F1-Score across all 4 threshold methods
    df["label"] = df["method"] + " (" + df["param"] + ")"
    pivot = df.pivot(index="group", columns="label", values="syn_f1")
    pivot = pivot.reindex(["global", "hour", "hour_weekend", "hour_week"])
    pivot.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("F1-Score")
    axes[1].set_title("F1-Score Comparison Across Threshold Methods")
    axes[1].legend(title="Threshold", fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_f1_precision_recall.png", bbox_inches="tight")
    plt.close()


def plot_quantification_mae():
    df = pd.read_csv(REPORTS / "threshold_ablation_results.csv")
    sub = df[(df["method"] == "percentile") & (df["param"] == "p=97.5")].copy()

    x = np.arange(len(sub))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w / 2, sub["mae_above_threshold_kwh"], width=w, label="Above-Threshold (Conservative)", color="#d62728")
    ax.bar(x + w / 2, sub["mae_above_baseline_kwh"], width=w, label="Above-Baseline (Flagged Hours)", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["group"])
    ax.set_ylabel("Quantification MAE (kWh)")
    ax.set_title("Synthetic Excess Quantification Error (p=97.5)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig3_quantification_mae.png", bbox_inches="tight")
    plt.close()


def plot_category_attribution():
    df = pd.read_csv(REPORTS / "unmodified_attribution_audit.csv")
    counts = df["largest_positive_contributor"].value_counts()

    fig, ax = plt.subplots(figsize=(7.5, 4))
    counts.plot(kind="barh", ax=ax, color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlabel("Number of Flagged Test Hours")
    ax.set_title("Dominant Category Contributors on Unmodified Flagged Hours (n=138)")
    for i, v in enumerate(counts.values):
        ax.text(v + 1, i, str(v), va="center", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig4_category_attribution.png", bbox_inches="tight")
    plt.close()


def main():
    plot_forecast_comparison()
    plot_f1_precision_recall()
    plot_quantification_mae()
    plot_category_attribution()
    print(f"All 4 graphs saved to: {FIGURES}")


if __name__ == "__main__":
    main()