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
    try:
        win_totals = pd.read_sql_query("SELECT * FROM win_totals", conn)
    except Exception:
        win_totals = pd.DataFrame(columns=["season", "team", "vegas_win_total"])
    conn.close()
    return games, pbp, win_totals

games, pbp, win_totals = load_data()

# 2. Standardize Franchise Abbreviations
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
win_totals["team"] = win_totals["team"].replace(team_map)

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

model_df["home_off_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("roll_off_epa", 0.0), axis=1)
model_df["home_def_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("roll_def_epa", 0.0), axis=1)
model_df["home_rest"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("rest_days", 7.0), axis=1)

model_df["away_off_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("roll_off_epa", 0.0), axis=1)
model_df["away_def_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("roll_def_epa", 0.0), axis=1)
model_df["away_rest"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("rest_days", 7.0), axis=1)

# Merge Vegas Season Win Totals
model_df = model_df.merge(
    win_totals.rename(columns={"team": "home_team", "vegas_win_total": "home_win_total"}),
    on=["season", "home_team"], how="left"
)
model_df = model_df.merge(
    win_totals.rename(columns={"team": "away_team", "vegas_win_total": "away_win_total"}),
    on=["season", "away_team"], how="left"
)

# Default to 8.5 wins if missing line
model_df["home_win_total"] = model_df["home_win_total"].fillna(8.5)
model_df["away_win_total"] = model_df["away_win_total"].fillna(8.5)

# Differentials
model_df["diff_off_epa"] = model_df["home_off_epa"] - model_df["away_off_epa"]
model_df["diff_def_epa"] = model_df["away_def_epa"] - model_df["home_def_epa"]
model_df["diff_rest"] = model_df["home_rest"] - model_df["away_rest"]
model_df["diff_win_total"] = model_df["home_win_total"] - model_df["away_win_total"]
model_df["spread_line"] = model_df["spread_line"].fillna(0.0)
model_df["home_win"] = (model_df["result"] > 0).astype(int)

# 5. Train Model
feature_cols = ["diff_off_epa", "diff_def_epa", "diff_rest", "diff_win_total", "spread_line"]
training_data = model_df.dropna(subset=feature_cols + ["home_win"]).copy()

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
    corr = training_data[["home_win", "spread_line", "diff_off_epa", "diff_def_epa", "diff_rest", "diff_win_total"]].corr()
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

# 7. Official Scheduled Matchups
st.subheader("📅 Official Scheduled Matchups")

upcoming_games = model_df[model_df["result"].isnull()].copy()
latest_team_form = team_stats.sort_values("gameday").groupby("team").last().reset_index().set_index("team")
latest_win_totals = win_totals.sort_values("season").groupby("team").last().reset_index().set_index("team")

if not upcoming_games.empty:
    target_season = upcoming_games["season"].max()
    season_sched = upcoming_games[upcoming_games["season"] == target_season]
    available_weeks = sorted(season_sched["week"].unique())
    
    selected_week = st.selectbox("Select Upcoming Week", available_weeks)
    week_games = season_sched[season_sched["week"] == selected_week].copy()

    cards = []
    for _, row in week_games.iterrows():
        ht, at = row["home_team"], row["away_team"]
        h_epa = latest_team_form.loc[ht, "roll_off_epa"] if ht in latest_team_form.index else 0.0
        a_epa = latest_team_form.loc[at, "roll_off_epa"] if at in latest_team_form.index else 0.0
        h_def = latest_team_form.loc[ht, "roll_def_epa"] if ht in latest_team_form.index else 0.0
        a_def = latest_team_form.loc[at, "roll_def_epa"] if at in latest_team_form.index else 0.0
        h_rest = latest_team_form.loc[ht, "rest_days"] if ht in latest_team_form.index else 7.0
        a_rest = latest_team_form.loc[at, "rest_days"] if at in latest_team_form.index else 7.0
        
        h_wt = latest_win_totals.loc[ht, "vegas_win_total"] if ht in latest_win_totals.index else 8.5
        a_wt = latest_win_totals.loc[at, "vegas_win_total"] if at in latest_win_totals.index else 8.5
        spread = row["spread_line"] if pd.notnull(row["spread_line"]) else 0.0

        sample = pd.DataFrame([{
            "diff_off_epa": h_epa - a_epa,
            "diff_def_epa": a_def - h_def,
            "diff_rest": h_rest - a_rest,
            "diff_win_total": h_wt - a_wt,
            "spread_line": spread
        }])
        prob = clf.predict_proba(scaler.transform(sample[feature_cols]))[0][1]
        cards.append({
            "Matchup": f"{at} @ {ht}",
            "Date": row.get("gameday", "TBD"),
            "Vegas Spread": spread,
            f"{ht} Win Total": h_wt,
            f"{at} Win Total": a_wt,
            f"{ht} Win Prob": f"{prob * 100:.1f}%",
            f"{at} Win Prob": f"{(1 - prob) * 100:.1f}%"
        })
    st.dataframe(pd.DataFrame(cards), use_container_width=True)
else:
    st.info("No upcoming games without final scores found in nfl_data.db. You can run the custom simulator below.")

st.markdown("---")

# 8. Dynamic Matchup Simulator
st.subheader("🎯 Custom Matchup Simulator")

team_list = sorted(team_stats["team"].unique())

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
    h_row = latest_team_form.loc[home_select]
    a_row = latest_team_form.loc[away_select]

    diff_off = h_row["roll_off_epa"] - a_row["roll_off_epa"]
    diff_def = a_row["roll_def_epa"] - h_row["roll_def_epa"]
    diff_rest = h_row["rest_days"] - a_row["rest_days"]

    h_wt = latest_win_totals.loc[home_select, "vegas_win_total"] if home_select in latest_win_totals.index else 8.5
    a_wt = latest_win_totals.loc[away_select, "vegas_win_total"] if away_select in latest_win_totals.index else 8.5
    diff_wt = h_wt - a_wt

    matchup_sample = pd.DataFrame([{
        "diff_off_epa": diff_off,
        "diff_def_epa": diff_def,
        "diff_rest": diff_rest,
        "diff_win_total": diff_wt,
        "spread_line": spread_input
    }])

    scaled_sample = scaler.transform(matchup_sample[feature_cols])
    home_prob = clf.predict_proba(scaled_sample)[0][1]
    away_prob = 1.0 - home_prob

    st.markdown(f"**Preseason Win Totals:** {home_select}: `{h_wt}` | {away_select}: `{a_wt}` (Differential: `{diff_wt:+.1f}`)")
    
    res_col1, res_col2 = st.columns(2)
    res_col1.metric(f"{home_select} (Home Win Probability)", f"{home_prob * 100:.1f}%")
    res_col2.metric(f"{away_select} (Away Win Probability)", f"{away_prob * 100:.1f}%")
    st.progress(float(home_prob))