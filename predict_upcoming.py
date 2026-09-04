import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Connect to Database & Load Historical Records
conn = sqlite3.connect("nfl_data.db")

games = pd.read_sql_query("""
    SELECT game_id, season, week, gameday, home_team, away_team, 
           spread_line, result
    FROM games
    WHERE game_type = 'REG'
    ORDER BY season, week
""", conn)

pbp = pd.read_sql_query("""
    SELECT game_id, posteam, defteam, epa
    FROM play_by_play
""", conn)

conn.close()

# 2. Build Historical Team Stats
off_stats = pbp.groupby(["game_id", "posteam"]).agg(
    off_epa=("epa", "mean"),
    plays=("epa", "count")
).reset_index().rename(columns={"posteam": "team"})

def_stats = pbp.groupby(["game_id", "defteam"]).agg(
    def_epa=("epa", "mean")
).reset_index().rename(columns={"defteam": "team"})

team_game_stats = pd.merge(off_stats, def_stats, on=["game_id", "team"])
team_stats_detailed = team_game_stats.merge(
    games[["game_id", "season", "week", "gameday"]], on="game_id"
).sort_values(["team", "season", "week"])

# Rolling 3-game metrics (shifted to avoid lookahead leakage)
team_stats_detailed["gameday"] = pd.to_datetime(team_stats_detailed["gameday"])
team_stats_detailed["rest_days"] = team_stats_detailed.groupby(["team", "season"])["gameday"].diff().dt.days.fillna(7)
team_stats_detailed["roll_off_epa"] = team_stats_detailed.groupby("team")["off_epa"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
team_stats_detailed["roll_def_epa"] = team_stats_detailed.groupby("team")["def_epa"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

features_df = team_stats_detailed[["game_id", "team", "roll_off_epa", "roll_def_epa", "rest_days"]].dropna()

# 3. Train Classifier on Historical Data
home_features = features_df.rename(columns={
    "team": "home_team",
    "roll_off_epa": "home_off_epa",
    "roll_def_epa": "home_def_epa",
    "rest_days": "home_rest"
})
away_features = features_df.rename(columns={
    "team": "away_team",
    "roll_off_epa": "away_off_epa",
    "roll_def_epa": "away_def_epa",
    "rest_days": "away_rest"
})

model_df = games.merge(home_features, on=["game_id", "home_team"])
model_df = model_df.merge(away_features, on=["game_id", "away_team"])

model_df["diff_off_epa"] = model_df["home_off_epa"] - model_df["away_off_epa"]
model_df["diff_def_epa"] = model_df["away_def_epa"] - model_df["home_def_epa"]
model_df["diff_rest"] = model_df["home_rest"] - model_df["away_rest"]
model_df["home_win"] = (model_df["result"] > 0).astype(int)

feature_cols = ["diff_off_epa", "diff_def_epa", "diff_rest", "spread_line"]
model_df = model_df.dropna(subset=feature_cols + ["home_win"])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(model_df[feature_cols])
y_train = model_df["home_win"]

clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# 4. Get Latest Known Form for Each Team
latest_team_form = team_stats_detailed.sort_values("gameday").groupby("team").last().reset_index()

def predict_matchup(home_team, away_team, spread_line=0.0, home_rest=7, away_rest=7):
    """Predicts outcome probabilities between any two teams."""
    h_data = latest_team_form[latest_team_form["team"] == home_team]
    a_data = latest_team_form[latest_team_form["team"] == away_team]
    
    if h_data.empty or a_data.empty:
        print(f"Error: Team abbreviation not found in database.")
        return

    diff_off = h_data["roll_off_epa"].values[0] - a_data["roll_off_epa"].values[0]
    diff_def = a_data["roll_def_epa"].values[0] - h_data["roll_def_epa"].values[0]
    diff_rest = home_rest - away_rest

    sample = pd.DataFrame([{
        "diff_off_epa": diff_off,
        "diff_def_epa": diff_def,
        "diff_rest": diff_rest,
        "spread_line": spread_line
    }])

    sample_scaled = scaler.transform(sample[feature_cols])
    home_win_prob = clf.predict_proba(sample_scaled)[0][1]
    away_win_prob = 1.0 - home_win_prob

    print(f"\n==========================================")
    print(f" Matchup: {away_team} @ {home_team}")
    print(f" Vegas Spread: {home_team} {spread_line:+0.1f}")
    print(f"------------------------------------------")
    print(f" Win Probability:")
    print(f"   {home_team} (Home): {home_win_prob * 100:.1f}%")
    print(f"   {away_team} (Away): {away_win_prob * 100:.1f}%")
    
    pred_winner = home_team if home_win_prob > 0.5 else away_team
    print(f" Projected Winner: {pred_winner}")
    print(f"==========================================\n")

# Example Test Matchups
predict_matchup(home_team="KC", away_team="BUF", spread_line=-2.5)
predict_matchup(home_team="SF", away_team="DAL", spread_line=-3.5)