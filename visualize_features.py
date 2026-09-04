import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Pull data from the SQLite database
conn = sqlite3.connect("nfl_data.db")
games = pd.read_sql_query("""
    SELECT game_id, season, week, gameday, home_team, away_team, 
           spread_line, result
    FROM games
    WHERE game_type = 'REG'
    ORDER BY season, week
""", conn)
pbp = pd.read_sql_query("SELECT game_id, posteam, defteam, epa FROM play_by_play", conn)
conn.close()

# 2. Build rolling team metrics
off_stats = pbp.groupby(["game_id", "posteam"]).agg(off_epa=("epa", "mean"), plays=("epa", "count")).reset_index().rename(columns={"posteam": "team"})
def_stats = pbp.groupby(["game_id", "defteam"]).agg(def_epa=("epa", "mean")).reset_index().rename(columns={"defteam": "team"})
team_stats = pd.merge(off_stats, def_stats, on=["game_id", "team"]).merge(games[["game_id", "season", "week", "gameday"]], on="game_id").sort_values(["team", "season", "week"])

team_stats["gameday"] = pd.to_datetime(team_stats["gameday"])
team_stats["rest_days"] = team_stats.groupby(["team", "season"])["gameday"].diff().dt.days.fillna(7)
team_stats["roll_off_epa"] = team_stats.groupby("team")["off_epa"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
team_stats["roll_def_epa"] = team_stats.groupby("team")["def_epa"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

features_df = team_stats[["game_id", "team", "roll_off_epa", "roll_def_epa", "rest_days"]].dropna()

# 3. Combine home & away data
home = features_df.rename(columns={"team": "home_team", "roll_off_epa": "home_off_epa", "roll_def_epa": "home_def_epa", "rest_days": "home_rest"})
away = features_df.rename(columns={"team": "away_team", "roll_off_epa": "away_off_epa", "roll_def_epa": "away_def_epa", "rest_days": "away_rest"})
df = games.merge(home, on=["game_id", "home_team"]).merge(away, on=["game_id", "away_team"])

df["diff_off_epa"] = df["home_off_epa"] - df["away_off_epa"]
df["diff_def_epa"] = df["away_def_epa"] - df["home_def_epa"]
df["diff_rest"] = df["home_rest"] - df["away_rest"]
df["home_win"] = (df["result"] > 0).astype(int)

cols_to_plot = ["home_win", "spread_line", "diff_off_epa", "diff_def_epa", "diff_rest"]
plot_df = df[cols_to_plot].dropna()

# 4. Generate Visualizations with Seaborn
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Correlation Heatmap
corr = plot_df.corr()
sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", linewidths=0.8, ax=axes[0], square=True)
axes[0].set_title("Feature Correlation Matrix (Predicting Home Win)", fontsize=13, weight="bold")

# Plot 2: Offensive EPA Difference vs. Outcome
sns.kdeplot(data=plot_df, x="diff_off_epa", hue="home_win", common_norm=False, fill=True, palette=["#e74c3c", "#2ecc71"], ax=axes[1])
axes[1].set_title("Offensive EPA Advantage: Wins vs. Losses", fontsize=13, weight="bold")
axes[1].set_xlabel("Home Off EPA - Away Off EPA")
axes[1].legend(title="Home Result", labels=["Home Win", "Home Loss"])

plt.tight_layout()
plt.savefig("nfl_model_features.png", dpi=300)
print("Plot saved as 'nfl_model_features.png'. Displaying chart window...")
plt.show()