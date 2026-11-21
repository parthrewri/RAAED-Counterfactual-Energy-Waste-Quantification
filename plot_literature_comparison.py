from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 300, "font.size": 9.5})

# Panel 1 & 2: Same-scale UCI Hourly Household models (kWh)
models = [
    "Banga & Sharma (2024)\n[LR+XGB+GA Hybrid]",
    "Khan et al. (2020)\n[CNN-LSTM-AE]",
    "RAAED — XGBoost\n(Proposed, 2026)",
    "RAAED — Random Forest\n(Baseline)",
    "Sajjad et al. (2020)\n[CNN-GRU]",
    "RAAED — Linear Reg.\n(Baseline)",
    "RAAED — Persistence\n(Baseline)",
]

mae_vals = [0.2830, 0.3100, 0.3232, 0.3269, 0.3300, 0.3549, 0.3861]
rmse_vals = [0.3870, 0.4700, 0.4657, 0.4717, 0.4700, 0.5011, 0.5843]

colors = [
    "#7f8c8d",  # Banga & Sharma (2024)
    "#7f8c8d",  # Khan et al. (2020)
    "#1e8449",  # RAAED XGBoost (Proposed)
    "#3498db",  # RAAED RF
    "#7f8c8d",  # Sajjad et al. (2020)
    "#5dade2",  # RAAED LR
    "#85c1e9",  # RAAED Persistence
]

fig, axes = plt.subplots(
    1, 3, figsize=(15.5, 5.2),
    gridspec_kw={"width_ratios": [1.35, 1.25, 0.95]}
)
y = np.arange(len(models))

# Panel 1: UCI Household MAE (kWh)
axes[0].barh(y, mae_vals, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_yticks(y)
axes[0].set_yticklabels(models, fontsize=8.8)
axes[0].invert_yaxis()
axes[0].set_xlabel("MAE (kWh) — Lower is Better")
axes[0].set_title("UCI Household MAE (kWh)")
axes[0].set_xlim(0, 0.47)
axes[0].grid(axis="x", linestyle="--", alpha=0.4)
for i, v in enumerate(mae_vals):
    axes[0].text(v + 0.006, i, f"{v:.4f}", va="center", fontsize=8.5,
                 fontweight="bold" if i == 2 else "normal")

# Panel 2: UCI Household RMSE (kWh)
axes[1].barh(y, rmse_vals, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_yticks(y)
axes[1].set_yticklabels([""] * len(models))
axes[1].invert_yaxis()
axes[1].set_xlabel("RMSE (kWh) — Lower is Better")
axes[1].set_title("UCI Household RMSE (kWh)")
axes[1].set_xlim(0, 0.69)
axes[1].grid(axis="x", linestyle="--", alpha=0.4)
for i, v in enumerate(rmse_vals):
    axes[1].text(v + 0.008, i, f"{v:.4f}", va="center", fontsize=8.5,
                 fontweight="bold" if i == 2 else "normal")

# Panel 3: Norouzi et al. (2026) — Different scale (Aggregate Grid Demand)
# Base XGBoost inferred from paper's reported -11.19% MAE & -11.30% RMSE reduction
grid_metrics = ["MAE", "RMSE"]
norouzi_vals = [15.8683, 22.1588]
x_pos = np.arange(len(grid_metrics))

bars = axes[2].bar(x_pos, norouzi_vals, width=0.5, color="#e67e22", edgecolor="black", linewidth=0.6)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(grid_metrics, fontsize=9.5)
axes[2].set_ylabel("Error (Grid Demand Scale)")
axes[2].set_title("Norouzi et al. (2026)*\n[XGBoost-RF, Grid Scale]")
axes[2].set_ylim(0, 27.5)
axes[2].grid(axis="y", linestyle="--", alpha=0.4)
for bar, v in zip(bars, norouzi_vals):
    axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

axes[2].text(
    0.5, 0.87,
    "R² = 0.9858 | MAPE = 1.31%\n(Aggregate Grid Demand,\n2015–2020; separate scale)",
    transform=axes[2].transAxes, ha="center", va="top", fontsize=7.8,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef5e7", edgecolor="#e67e22", alpha=0.9)
)

legend_handles = [
    Patch(facecolor="#1e8449", edgecolor="black", label="RAAED Proposed (XGBoost, UCI)"),
    Patch(facecolor="#3498db", edgecolor="black", label="RAAED Internal Baselines (UCI)"),
    Patch(facecolor="#7f8c8d", edgecolor="black", label="Published UCI Household Models (2020–2024)"),
    Patch(facecolor="#e67e22", edgecolor="black", label="Norouzi et al. (2026) — Grid-Scale Hybrid XGB-RF"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = FIGURES / "fig5_literature_comparison.png"
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"Saved updated comparison graph to: {out_path}")