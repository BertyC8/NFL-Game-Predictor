import sqlite3
import nflreadpy as nfl
import pandas as pd

print("Connecting to database...")
conn = sqlite3.connect("nfl_data.db")

# Target regular 5 seasons
seasons = list(range(2021, 2026))

print(f"1. Downloading schedule and game results for {seasons}...")
schedules = nfl.load_schedules(seasons).to_pandas()

# Filter for completed regular-season games
reg_games = schedules[(schedules["game_type"] == "REG") & (schedules["result"].notna())]
reg_games.to_sql("games", conn, if_exists="replace", index=False)
print(f"Saved {len(reg_games)} games to the 'games' table.")

print("2. Downloading play-by-play data (takes ~20-30s)...")
pbp = nfl.load_pbp(seasons).to_pandas()

# Select key columns for EPA and pace metrics
core_cols = [
    "game_id", "season", "week", "posteam", "defteam",
    "epa", "play_type", "pass_attempt", "rush_attempt"
]
pbp_filtered = pbp[core_cols].dropna(subset=["posteam", "defteam", "epa"])
pbp_filtered.to_sql("play_by_play", conn, if_exists="replace", index=False)
print(f"Saved {len(pbp_filtered):,} plays to the 'play_by_play' table.")

conn.close()
print("\nSuccess! Database 'nfl_data.db' is ready.")