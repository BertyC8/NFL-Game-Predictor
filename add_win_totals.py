import sqlite3
import pandas as pd

conn = sqlite3.connect("nfl_data.db")
print("Downloading Vegas season win totals...")

# Public nflverse win totals ledger
url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/win_totals.csv"
win_totals_df = pd.read_csv(url)

# Clean team abbreviations to match standard database codes
team_map = {"OAK": "LV", "WAS": "WSH", "STL": "LA", "LAR": "LA", "SD": "LAC"}
win_totals_df["team"] = win_totals_df["team"].replace(team_map)

# Keep season, team, and the line
clean_totals = win_totals_df[["season", "team", "line"]].rename(columns={"line": "vegas_win_total"})

clean_totals.to_sql("win_totals", conn, if_exists="replace", index=False)
print(f"Saved {len(clean_totals)} team win total lines to 'win_totals' table.")
conn.close()