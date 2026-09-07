import sqlite3
import nflreadpy as nfl
import pandas as pd

# 1. Connect to SQLite
conn = sqlite3.connect("nfl_data.db")
print("Connecting to database...")

# 2. Completed PBP data only exists up to 2025
pbp_seasons = [2021, 2022, 2023, 2024, 2025]

# 3. Pull schedules (load_schedules without arguments gets the full historical + upcoming ledger)
print("1. Downloading schedules...")
try:
    schedules_pl = nfl.load_schedules()
except TypeError:
    schedules_pl = nfl.load_schedules(pbp_seasons)

schedules_df = schedules_pl.to_pandas()
schedules_df.to_sql("games", conn, if_exists="replace", index=False)
print(f"Saved {len(schedules_df):,} games to the 'games' table.")

# 4. Download play-by-play only for valid historical seasons
print("2. Downloading play-by-play data (takes ~20-30s)...")
pbp_pl = nfl.load_pbp(pbp_seasons)
pbp_df = pbp_pl.to_pandas()

pbp_df = pbp_df[["game_id", "posteam", "defteam", "epa", "play_id"]].dropna(subset=["epa"])
pbp_df.to_sql("play_by_play", conn, if_exists="replace", index=False)
print(f"Saved {len(pbp_df):,} plays to the 'play_by_play' table.")

conn.close()
print("\nSuccess! Database 'nfl_data.db' is ready.")