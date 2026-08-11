"""
build_b7_matchup.py — builds a real batter-vs-bowler matchup summary directly from the
raw ball-by-ball dataset. B7 was described in PROJECT_REPORT.md and training/train_batting.py
as "a batter x bowler lookup table... left untouched/out of scope" -- no current model,
training code, or results file exists for it anywhere in the codebase. Rather than leave a
gap or invent numbers, this script computes a genuine descriptive summary (not a predictive
model) straight from data/traning_dataset/dataset.csv: balls faced, runs scored, strike
rate, and dismissals per batter-bowler pair, restricted to pairs with a minimum sample size
so single-ball flukes don't dominate the chart.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "traning_dataset", "dataset.csv")
OUT_DIR = os.path.join(ROOT, "data", "output_data", "research_figures", "batting", "B7")
MIN_BALLS = 30

BLUE = "#3b6ea5"
ORANGE = "#d0793a"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["legal_ball"] = ((df["wides"] == 0) & (df["noballs"] == 0)).astype(int)

    grp = df.groupby(["batter", "bowler"]).agg(
        balls_faced=("legal_ball", "sum"),
        runs_scored=("batter_runs", "sum"),
        dismissals=("is_wicket", "sum"),
    ).reset_index()
    grp = grp[grp["balls_faced"] >= MIN_BALLS].copy()
    grp["strike_rate"] = (grp["runs_scored"] / grp["balls_faced"]) * 100
    grp["dismissal_rate"] = grp["dismissals"] / grp["balls_faced"]
    grp["pair"] = grp["batter"] + " vs " + grp["bowler"]

    print(f"Total batter-bowler pairs with >= {MIN_BALLS} balls faced: {len(grp)}")

    top_sr = grp.sort_values("strike_rate", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(top_sr["pair"], top_sr["strike_rate"], color=BLUE)
    for y, (sr, balls) in enumerate(zip(top_sr["strike_rate"], top_sr["balls_faced"])):
        ax.text(sr + 1, y, f"{int(balls)} balls", va="center", fontsize=8, color="#444")
    ax.set_xlabel("Strike rate (runs per 100 balls)")
    ax.set_title(f"B7: Top 15 Batter–Bowler Matchups by Strike Rate\n"
                 f"(minimum {MIN_BALLS} balls faced, computed directly from raw ball-by-ball data)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "B7_top_strike_rate_matchups.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.scatter(grp["strike_rate"], grp["dismissal_rate"] * 100, s=14, alpha=0.35, color=ORANGE)
    ax.set_xlabel("Strike rate (runs per 100 balls)")
    ax.set_ylabel("Dismissal rate (%) per ball faced")
    ax.set_title(f"B7: Matchup Strike Rate vs. Dismissal Rate\n"
                 f"({len(grp)} batter–bowler pairs, minimum {MIN_BALLS} balls faced)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "B7_strike_rate_vs_dismissal_rate.png"), dpi=180)
    plt.close(fig)

    print("Wrote B7 figures to", OUT_DIR)


if __name__ == "__main__":
    main()
