import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="NFL Prediction Dashboard", layout="wide")

st.title("🏈 NFL Dynamic Predictor & Performance Dashboard")
st.markdown("Live analytics powered by SQLite database metrics and logistic win probability modeling.")

# 1. Load Data from SQLite
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect("nfl_data.db")
    games = pd.read_sql_query("SELECT * FROM games ORDER BY season, week", conn)
    pbp = pd.read_sql_query("SELECT game_id, posteam, defteam, epa FROM play_by_play", conn)
    conn.close()
    return games, pbp

games, pbp = load_data()

# 2. Standardize Franchise Abbreviations (fixes historic naming mismatches)
team_map = {
    "OAK": "LV",
    "WAS": "WSH",
    "STL": "LA",
    "LAR": "LA",
    "SD": "LAC"
}
games["home_team"] = games["home_team"].replace(team_map)
games["away_team"] = games["away_team"].replace(team_map)
pbp["posteam"] = pbp["posteam"].replace(team_map)
pbp["defteam"] = pbp["defteam"].replace(team_map)

# 3. Dynamic Feature Engineering
off_stats = pbp.groupby(["game_id", "posteam"]).agg(
    off_epa=("epa", "mean"), plays=("epa", "count")
).reset_index().rename(columns={"posteam": "team"})

def_stats = pbp.groupby(["game_id", "defteam"]).agg(
    def_epa=("epa", "mean")
).reset_index().rename(columns={"defteam": "team"})

team_stats = pd.merge(off_stats, def_stats, on=["game_id", "team"]).merge(
    games[["game_id", "season", "week", "gameday"]].drop_duplicates(subset=["game_id"]), on="game_id"
).sort_values(["team", "season", "week"])

# Rest Days
team_stats["gameday"] = pd.to_datetime(team_stats["gameday"])
team_stats["rest_days"] = team_stats.groupby(["team", "season"])["gameday"].diff().dt.days.fillna(7)

# 3-game rolling averages (shift 1 to prevent data leakage)
team_stats["roll_off_epa"] = (
    team_stats.groupby("team")["off_epa"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .bfill()
    .fillna(0)
)
team_stats["roll_def_epa"] = (
    team_stats.groupby("team")["def_epa"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .bfill()
    .fillna(0)
)

# 4. Map Stats Directly to Home and Away Matchups
metrics_dict = team_stats.set_index(["game_id", "team"])[["roll_off_epa", "roll_def_epa", "rest_days"]].to_dict("index")

model_df = games.copy()

# Look up metrics safely without merge drops
model_df["home_off_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("roll_off_epa", 0.0), axis=1)
model_df["home_def_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("roll_def_epa", 0.0), axis=1)
model_df["home_rest"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("rest_days", 7.0), axis=1)

model_df["away_off_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("roll_off_epa", 0.0), axis=1)
model_df["away_def_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("roll_def_epa", 0.0), axis=1)
model_df["away_rest"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("rest_days", 7.0), axis=1)

# Differentials
model_df["diff_off_epa"] = model_df["home_off_epa"] - model_df["away_off_epa"]
model_df["diff_def_epa"] = model_df["away_def_epa"] - model_df["home_def_epa"]
model_df["diff_rest"] = model_df["home_rest"] - model_df["away_rest"]
model_df["spread_line"] = model_df["spread_line"].fillna(0.0)
model_df["home_win"] = (model_df["result"] > 0).astype(int)

feature_cols = ["diff_off_epa", "diff_def_epa", "diff_rest", "spread_line"]
training_data = model_df.dropna(subset=feature_cols + ["home_win"]).copy()

# 5. Train Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(training_data[feature_cols])
y = training_data["home_win"]

clf = LogisticRegression()
clf.fit(X_scaled, y)

# 6. Seaborn Visualizations
st.subheader("📊 Live Data Distribution & EPA Advantage")
col_vis1, col_vis2 = st.columns(2)

sns.set_theme(style="whitegrid")

with col_vis1:
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    corr = training_data[["home_win", "spread_line", "diff_off_epa", "diff_def_epa", "diff_rest"]].corr()
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", linewidths=0.5, ax=ax1)
    ax1.set_title("Feature Correlation with Win Outcome", weight="bold")
    st.pyplot(fig1)

with col_vis2:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.kdeplot(
        data=training_data, 
        x="diff_off_epa", 
        hue="home_win", 
        common_norm=False, 
        fill=True, 
        palette=["#e74c3c", "#2ecc71"], 
        ax=ax2
    )
    ax2.set_title("Offensive EPA Advantage (Wins vs Losses)", weight="bold")
    ax2.set_xlabel("Home Off EPA - Away Off EPA")
    st.pyplot(fig2)

st.markdown("---")

# 7. Dynamic Matchup Simulator
st.subheader("🎯 Upcoming Matchup Predictor")

latest_team_form = team_stats.sort_values("gameday").groupby("team").last().reset_index()
team_list = sorted(latest_team_form["team"].unique())

c1, c2, c3 = st.columns(3)
with c1:
    home_select = st.selectbox("Home Team", team_list, index=team_list.index("KC") if "KC" in team_list else 0)
with c2:
    away_select = st.selectbox("Away Team", team_list, index=team_list.index("BUF") if "BUF" in team_list else 1)
with c3:
    spread_input = st.number_input("Vegas Spread Line (Home Team)", value=-2.5, step=0.5)

if home_select == away_select:
    st.warning("Please choose two different teams.")
else:
    h_row = latest_team_form[latest_team_form["team"] == home_select].iloc[0]
    a_row = latest_team_form[latest_team_form["team"] == away_select].iloc[0]

    diff_off = h_row["roll_off_epa"] - a_row["roll_off_epa"]
    diff_def = a_row["roll_def_epa"] - h_row["roll_def_epa"]
    diff_rest = h_row["rest_days"] - a_row["rest_days"]

    matchup_sample = pd.DataFrame([{
        "diff_off_epa": diff_off,
        "diff_def_epa": diff_def,
        "diff_rest": diff_rest,
        "spread_line": spread_input
    }])

    scaled_sample = scaler.transform(matchup_sample[feature_cols])
    home_prob = clf.predict_proba(scaled_sample)[0][1]
    away_prob = 1.0 - home_prob

    st.markdown("### Prediction Results")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric(f"{home_select} (Home Win Probability)", f"{home_prob * 100:.1f}%")
    res_col2.metric(f"{away_select} (Away Win Probability)", f"{away_prob * 100:.1f}%")
    st.progress(float(home_prob))