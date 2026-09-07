import sqlite3
import nflreadpy as nfl
import pandas as pd

# 1. Connect to SQLite
conn = sqlite3.connect("nfl_data.db")
print("Connecting to database...")

# 2. Download Vegas Season Win Totals
url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/win_totals.csv"
team_map = {"OAK": "LV", "WAS": "WSH", "STL": "LA", "LAR": "LA", "SD": "LAC"}

try:
    print("Downloading Vegas season win totals...")
    win_totals_df = pd.read_csv(url)
    win_totals_df["team"] = win_totals_df["team"].replace(team_map)
    clean_totals = win_totals_df[["season", "team", "line"]].rename(columns={"line": "vegas_win_total"})
    clean_totals.to_sql("win_totals", conn, if_exists="replace", index=False)
    print(f"Saved {len(clean_totals)} team win total lines to 'win_totals' table.")
except Exception as e:
    print(f"Warning: Could not fetch win totals ({e}). Skipping table creation.")

# 3. Pull Schedules (Full historical + upcoming schedule)
pbp_seasons = [2021, 2022, 2023, 2024, 2025]
print("Downloading schedules...")
try:
    schedules_pl = nfl.load_schedules()
except TypeError:
    schedules_pl = nfl.load_schedules(pbp_seasons)

schedules_df = schedules_pl.to_pandas()
schedules_df.to_sql("games", conn, if_exists="replace", index=False)
print(f"Saved {len(schedules_df):,} games to the 'games' table.")

# 4. Download Play-by-Play Data for Model Training
print("Downloading play-by-play data (takes ~20-30s)...")
pbp_pl = nfl.load_pbp(pbp_seasons)
pbp_df = pbp_pl.to_pandas()

pbp_df = pbp_df[["game_id", "posteam", "defteam", "epa", "play_id"]].dropna(subset=["epa"])
pbp_df.to_sql("play_by_play", conn, if_exists="replace", index=False)
print(f"Saved {len(pbp_df):,} plays to the 'play_by_play' table.")

conn.close()
print("\nSuccess! Database 'nfl_data.db' is ready.")