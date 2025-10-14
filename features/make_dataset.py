import pandas as pd
import numpy as np
from pathlib import Path

PBP   = Path("data/pbp.parquet")
SCHED = Path("data/schedules.parquet")
OUT   = Path("data/processed.csv")

pbp   = pd.read_parquet(PBP)
sched = pd.read_parquet(SCHED)

# ---- Per-team, per-game aggregates from play-by-play ----
# posteam = team on offense for the play (works fine for most aggregates)
grp = pbp.groupby(["game_id", "posteam"], dropna=True).agg(
    yards_per_play=("yards_gained", "mean"),
    turnovers=("interception", "sum"),            # ints
    fumbles_lost=("fumble_lost", "sum"),          # lost fumbles
    passing_yards=("passing_yards", "sum"),
    rushing_yards=("rushing_yards", "sum"),
    penalties=("penalty", "sum"),
    sacks=("sack", "sum"),
    third_down_rate=("third_down_converted", "mean")
).reset_index()

grp["turnovers_total"] = grp["turnovers"].fillna(0) + grp["fumbles_lost"].fillna(0)
grp = grp.drop(columns=["turnovers", "fumbles_lost"])

# ---- Map to home/away by joining schedules (unique per game) ----
games = sched[["game_id", "home_team", "away_team", "home_score", "away_score"]].drop_duplicates("game_id")

home_stats = games.merge(grp, left_on=["game_id","home_team"], right_on=["game_id","posteam"], how="left", suffixes=("", "_home"))
home_stats = home_stats.rename(columns={
    "yards_per_play":"home_ypp",
    "turnovers_total":"home_turnovers",
    "passing_yards":"home_passing_yards",
    "rushing_yards":"home_rushing_yards",
    "penalties":"home_penalties",
    "sacks":"home_sacks",
    "third_down_rate":"home_third_down_rate"
}).drop(columns=["posteam"])

away_stats = games.merge(grp, left_on=["game_id","away_team"], right_on=["game_id","posteam"], how="left", suffixes=("", "_away"))
away_stats = away_stats.rename(columns={
    "yards_per_play":"away_ypp",
    "turnovers_total":"away_turnovers",
    "passing_yards":"away_passing_yards",
    "rushing_yards":"away_rushing_yards",
    "penalties":"away_penalties",
    "sacks":"away_sacks",
    "third_down_rate":"away_third_down_rate"
}).drop(columns=["posteam"])

# Combine home + away
full = home_stats.merge(
    away_stats[["game_id","away_ypp","away_turnovers","away_passing_yards","away_rushing_yards","away_penalties","away_sacks","away_third_down_rate"]],
    on="game_id", how="left"
)

# ---- Features (keep it small & interpretable) ----
full["yards_per_play_diff"] = full["home_ypp"] - full["away_ypp"]
full["turnovers_diff"]      = full["home_turnovers"] - full["away_turnovers"]
full["passing_yards"]       = full["home_passing_yards"]
full["rushing_yards"]       = full["home_rushing_yards"]
full["sacks"]               = full["home_sacks"]          # sacks taken by home offense
full["penalties"]           = full["home_penalties"]
full["third_down_rate"]     = full["home_third_down_rate"]  # keep as 0..1

# Target: did home win?
full["home_win"] = (full["home_score"] > full["away_score"]).astype(int)

keep = [
    "game_id","home_team","away_team","home_score","away_score",
    "yards_per_play_diff","turnovers_diff","passing_yards","rushing_yards",
    "sacks","penalties","third_down_rate","home_win"
]

df = full[keep].dropna().reset_index(drop=True)
Path("data").mkdir(exist_ok=True)
df.to_csv(OUT, index=False)
print("Wrote", OUT, "rows:", len(df))
