import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Load data from SQLite database
conn = sqlite3.connect("nfl_data.db")

games = pd.read_sql_query("""
    SELECT game_id, season, week, gameday, home_team, away_team, 
           home_score, away_score, spread_line, result
    FROM games
    WHERE game_type = 'REG'
    ORDER BY season, week
""", conn)

pbp = pd.read_sql_query("""
    SELECT game_id, season, week, posteam, defteam, epa
    FROM play_by_play
""", conn)

conn.close()

print(f"Loaded {len(games)} games and {len(pbp):,} play records.")

# 2. Calculate Offensive & Defensive EPA and Pace per team/game
off_stats = pbp.groupby(["game_id", "posteam"]).agg(
    off_epa=("epa", "mean"),
    plays=("epa", "count")
).reset_index().rename(columns={"posteam": "team"})

def_stats = pbp.groupby(["game_id", "defteam"]).agg(
    def_epa=("epa", "mean")
).reset_index().rename(columns={"defteam": "team"})

team_game_stats = pd.merge(off_stats, def_stats, on=["game_id", "team"])

# Merge team stats back with game schedule info
team_stats_detailed = team_game_stats.merge(
    games[["game_id", "season", "week", "gameday"]], on="game_id"
).sort_values(["team", "season", "week"])

# 3. Calculate Rest Days and Rolling Metrics
team_stats_detailed["gameday"] = pd.to_datetime(team_stats_detailed["gameday"])
team_stats_detailed["rest_days"] = team_stats_detailed.groupby(["team", "season"])["gameday"].diff().dt.days.fillna(7)

# 3-game rolling averages (shift 1 to prevent data leakage)
team_stats_detailed["roll_off_epa"] = team_stats_detailed.groupby("team")["off_epa"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
team_stats_detailed["roll_def_epa"] = team_stats_detailed.groupby("team")["def_epa"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
team_stats_detailed["roll_pace"] = team_stats_detailed.groupby("team")["plays"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

features_df = team_stats_detailed[["game_id", "team", "roll_off_epa", "roll_def_epa", "roll_pace", "rest_days"]].dropna()

# 4. Combine Home and Away stats
home_features = features_df.rename(columns={
    "team": "home_team",
    "roll_off_epa": "home_off_epa",
    "roll_def_epa": "home_def_epa",
    "roll_pace": "home_pace",
    "rest_days": "home_rest"
})

away_features = features_df.rename(columns={
    "team": "away_team",
    "roll_off_epa": "away_off_epa",
    "roll_def_epa": "away_def_epa",
    "away_pace": "away_pace",
    "rest_days": "away_rest"
})

model_df = games.merge(home_features, on=["game_id", "home_team"])
model_df = model_df.merge(away_features, on=["game_id", "away_team"])

# 5. Create Differential Features & Target
model_df["diff_off_epa"] = model_df["home_off_epa"] - model_df["away_off_epa"]
model_df["diff_def_epa"] = model_df["away_def_epa"] - model_df["home_def_epa"]
model_df["diff_rest"] = model_df["home_rest"] - model_df["away_rest"]

model_df["home_win"] = (model_df["result"] > 0).astype(int)

feature_cols = ["diff_off_epa", "diff_def_epa", "diff_rest", "spread_line"]
model_df = model_df.dropna(subset=feature_cols + ["home_win"])

# 6. Train/Test Split (Train on older seasons, test on 2025)
train = model_df[model_df["season"] < 2025]
test = model_df[model_df["season"] == 2025]

X_train, y_train = train[feature_cols], train["home_win"]
X_test, y_test = test[feature_cols], test["home_win"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Logistic Regression
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# 8. Evaluate Predictions
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation (2025 Season Test Set) ---")
print(f"Overall Prediction Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))