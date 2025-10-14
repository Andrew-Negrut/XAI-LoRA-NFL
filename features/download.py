import nfl_data_py as nfl
from pathlib import Path
import pandas as pd

Path("data").mkdir(exist_ok=True)

YEARS = list(range(2019, 2024))  

print("Downloading play-by-play…")
pbp = nfl.import_pbp_data(YEARS)        # big table: 1 row per play
pbp.to_parquet("data/pbp.parquet")

print("Downloading schedules…")
sched = nfl.import_schedules(YEARS)     # game-level: one row per game
sched.to_parquet("data/schedules.parquet")

print("Done:", len(pbp), "plays;", len(sched), "games")
