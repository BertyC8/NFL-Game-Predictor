import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="NFL Prediction Dashboard", layout="wide")

st.title("🏈 NFL Matchup Predictor & Performance Dashboard")
st.markdown("Automated game winner predictions using 3-game rolling EPA, rest margins, spread consensus, and Vegas season win totals.")

# 1. Load Data
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect("nfl_data.db")
    games = pd.read_sql_query("SELECT * FROM games ORDER BY season, week", conn)
    pbp = pd.read_sql_query("SELECT game_id, posteam, defteam, epa FROM play_by_play", conn)
    conn.close()
    return games, pbp

games, pbp = load_data()

# 2. Standardize Team Abbreviations
team_map = {"OAK": "LV", "WAS": "WSH", "STL": "LA", "LAR": "LA", "SD": "LAC"}
games["home_team"] = games["home_team"].replace(team_map)
games["away_team"] = games["away_team"].replace(team_map)
pbp["posteam"] = pbp["posteam"].replace(team_map)
pbp["defteam"] = pbp["defteam"].replace(team_map)

# 3. Dynamic Rolling Feature Engineering
off_stats = pbp.groupby(["game_id", "posteam"]).agg(off_epa=("epa", "mean"), plays=("epa", "count")).reset_index().rename(columns={"posteam": "team"})
def_stats = pbp.groupby(["game_id", "defteam"]).agg(def_epa=("epa", "mean")).reset_index().rename(columns={"defteam": "team"})

team_stats = pd.merge(off_stats, def_stats, on=["game_id", "team"]).merge(
    games[["game_id", "season", "week", "gameday"]].drop_duplicates(subset=["game_id"]), on="game_id"
).sort_values(["team", "season", "week"])

team_stats["gameday"] = pd.to_datetime(team_stats["gameday"])
team_stats["rest_days"] = team_stats.groupby(["team", "season"])["gameday"].diff().dt.days.fillna(7)
team_stats["roll_off_epa"] = team_stats.groupby("team")["off_epa"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).bfill().fillna(0)
team_stats["roll_def_epa"] = team_stats.groupby("team")["def_epa"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).bfill().fillna(0)

metrics_dict = team_stats.set_index(["game_id", "team"])[["roll_off_epa", "roll_def_epa", "rest_days"]].to_dict("index")

model_df = games.copy()
model_df["home_off_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("roll_off_epa", 0.0), axis=1)
model_df["home_def_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("roll_def_epa", 0.0), axis=1)
model_df["home_rest"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["home_team"]), {}).get("rest_days", 7.0), axis=1)

model_df["away_off_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("roll_off_epa", 0.0), axis=1)
model_df["away_def_epa"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("roll_def_epa", 0.0), axis=1)
model_df["away_rest"] = model_df.apply(lambda r: metrics_dict.get((r["game_id"], r["away_team"]), {}).get("rest_days", 7.0), axis=1)

# Current season baseline win totals mapping
vegas_win_totals_2026 = {
    "LA": 11.5, "BAL": 11.5, "BUF": 10.5, "SEA": 10.5, "DET": 10.5,
    "NE": 10.5, "KC": 10.5, "CIN": 10.5, "PHI": 10.5, "HOU": 9.5,
    "SF": 9.5, "LAC": 9.5, "GB": 9.5, "DEN": 9.5, "DAL": 9.5,
    "CHI": 9.5, "JAX": 8.5, "MIN": 8.5, "PIT": 8.5, "TB": 8.5,
    "IND": 7.5, "NO": 7.5, "NYG": 7.5, "WSH": 7.5, "CAR": 7.5,
    "ATL": 7.5, "TEN": 6.5, "LV": 5.5, "CLE": 5.5, "NYJ": 5.5,
    "MIA": 3.5, "ARI": 3.5
}

model_df["home_win_total"] = model_df["home_team"].map(vegas_win_totals_2026).fillna(8.5)
model_df["away_win_total"] = model_df["away_team"].map(vegas_win_totals_2026).fillna(8.5)

model_df["diff_off_epa"] = model_df["home_off_epa"] - model_df["away_off_epa"]
model_df["diff_def_epa"] = model_df["away_def_epa"] - model_df["home_def_epa"]
model_df["diff_rest"] = model_df["home_rest"] - model_df["away_rest"]
model_df["diff_win_total"] = model_df["home_win_total"] - model_df["away_win_total"]
model_df["spread_line"] = model_df["spread_line"].fillna(0.0)

# 4. Train Model
completed_games = model_df[model_df["result"].notnull()].copy()
completed_games["home_win"] = (completed_games["result"] > 0).astype(int)

feature_cols = ["diff_off_epa", "diff_def_epa", "diff_rest", "diff_win_total", "spread_line"]
training_data = completed_games.dropna(subset=feature_cols + ["home_win"]).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(training_data[feature_cols])
y = training_data["home_win"]

clf = LogisticRegression()
clf.fit(X_scaled, y)

# 5. Visualizations
st.subheader("📊 Model Visualizations")
c_vis1, c_vis2 = st.columns(2)
sns.set_theme(style="whitegrid")

with c_vis1:
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    corr = training_data[["home_win", "spread_line", "diff_off_epa", "diff_def_epa", "diff_win_total"]].corr()
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", linewidths=0.5, ax=ax1)
    ax1.set_title("Feature Correlations with Game Winner", weight="bold")
    st.pyplot(fig1)

with c_vis2:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.kdeplot(data=training_data, x="diff_win_total", hue="home_win", common_norm=False, fill=True, palette=["#e74c3c", "#2ecc71"], ax=ax2)
    ax2.set_title("Vegas Win Total Differential (Wins vs Losses)", weight="bold")
    ax2.set_xlabel("Home Vegas Line - Away Vegas Line")
    st.pyplot(fig2)

st.markdown("---")

# 6. Official Scheduled Matchups with Machine Picks
st.subheader("📅 Scheduled Matchups & Machine Winner Picks")

upcoming_games = model_df[model_df["result"].isnull()].copy()
latest_team_form = team_stats.sort_values("gameday").groupby("team").last().reset_index().set_index("team")

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
        h_wt = vegas_win_totals_2026.get(ht, 8.5)
        a_wt = vegas_win_totals_2026.get(at, 8.5)
        spread = row["spread_line"] if pd.notnull(row["spread_line"]) else 0.0

        sample = pd.DataFrame([{
            "diff_off_epa": h_epa - a_epa,
            "diff_def_epa": a_def - h_def,
            "diff_rest": h_rest - a_rest,
            "diff_win_total": h_wt - a_wt,
            "spread_line": spread
        }])
        
        scaled_sample = scaler.transform(sample[feature_cols])
        prob = clf.predict_proba(scaled_sample)[0][1]
        winner = ht if prob >= 0.50 else at
        confidence = prob if prob >= 0.50 else (1 - prob)

        cards.append({
            "Matchup": f"{at} @ {ht}",
            "Date": row.get("gameday", "TBD"),
            "Spread": spread,
            f"{ht} Win Total": h_wt,
            f"{at} Win Total": a_wt,
            "Machine Pick": f"🏆 {winner}",
            "Confidence": f"{confidence * 100:.1f}%"
        })
    st.dataframe(pd.DataFrame(cards), use_container_width=True)
else:
    st.info("No upcoming games without final scores found in nfl_data.db. Use the simulator below.")

st.markdown("---")

# 7. Custom Matchup Simulator (Predicts Clear Winner)
st.subheader("🎯 Custom Matchup Simulator")

team_list = sorted(vegas_win_totals_2026.keys())
c1, c2, c3 = st.columns(3)
with c1:
    home_select = st.selectbox("Home Team", team_list, index=team_list.index("LA") if "LA" in team_list else 0)
with c2:
    away_select = st.selectbox("Away Team", team_list, index=team_list.index("BAL") if "BAL" in team_list else 1)
with c3:
    spread_input = st.number_input("Vegas Spread Line (Home Team)", value=-2.5, step=0.5)

if home_select == away_select:
    st.warning("Please choose two different teams.")
else:
    h_epa = latest_team_form.loc[home_select, "roll_off_epa"] if home_select in latest_team_form.index else 0.0
    a_epa = latest_team_form.loc[away_select, "roll_off_epa"] if away_select in latest_team_form.index else 0.0
    h_def = latest_team_form.loc[home_select, "roll_def_epa"] if home_select in latest_team_form.index else 0.0
    a_def = latest_team_form.loc[away_select, "roll_def_epa"] if away_select in latest_team_form.index else 0.0
    h_rest = latest_team_form.loc[home_select, "rest_days"] if home_select in latest_team_form.index else 7.0
    a_rest = latest_team_form.loc[away_select, "rest_days"] if away_select in latest_team_form.index else 7.0

    h_wt = vegas_win_totals_2026.get(home_select, 8.5)
    a_wt = vegas_win_totals_2026.get(away_select, 8.5)

    matchup_sample = pd.DataFrame([{
        "diff_off_epa": h_epa - a_epa,
        "diff_def_epa": a_def - h_def,
        "diff_rest": h_rest - a_rest,
        "diff_win_total": h_wt - a_wt,
        "spread_line": spread_input
    }])

    scaled_sample = scaler.transform(matchup_sample[feature_cols])
    prediction = clf.predict(scaled_sample)[0]
    prob = clf.predict_proba(scaled_sample)[0][1]

    predicted_winner = home_select if prediction == 1 else away_select
    winner_prob = prob if prediction == 1 else (1.0 - prob)

    st.markdown(f"**Season Win Totals:** {home_select}: `{h_wt}` | {away_select}: `{a_wt}` (Differential: `{h_wt - a_wt:+.1f}`)")
    st.success(f"### 🏆 Machine Pick: **{predicted_winner}** to win outright")
    st.caption(f"Calculated win confidence: **{winner_prob * 100:.1f}%**")
    st.progress(float(prob))

st.markdown("---")

# 8. Machine Accuracy & Performance Tracking
st.subheader("📈 Machine Accuracy & Performance Tracker")

completed = training_data.copy()

# Generate model predictions
completed["pred_prob"] = clf.predict_proba(X_scaled)[:, 1]
completed["pred_home_win"] = (completed["pred_prob"] >= 0.50).astype(int)
completed["correct_pick"] = (completed["pred_home_win"] == completed["home_win"]).astype(int)

eval_seasons = sorted(completed["season"].unique(), reverse=True)
selected_eval_season = st.selectbox("Select Season to Evaluate", eval_seasons)

season_df = completed[completed["season"] == selected_eval_season].copy()

total_games = len(season_df)
correct_picks = season_df["correct_pick"].sum()
overall_acc = (correct_picks / total_games) * 100 if total_games > 0 else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Season Accuracy", f"{overall_acc:.1f}%")
m2.metric("Total Correct Picks", f"{correct_picks} / {total_games}")
m3.metric("Underdog Hits", f"{len(season_df[(season_df['correct_pick'] == 1) & (season_df['spread_line'] > 0)])}")

st.markdown("#### Weekly Accuracy Trend")
weekly_acc = season_df.groupby("week")["correct_pick"].agg(["count", "sum"]).reset_index()
weekly_acc["accuracy"] = (weekly_acc["sum"] / weekly_acc["count"]) * 100

fig_week, ax_week = plt.subplots(figsize=(10, 3))
sns.lineplot(data=weekly_acc, x="week", y="accuracy", marker="o", color="#2ecc71", linewidth=2.5, ax=ax_week)
ax_week.axhline(50, color="gray", linestyle="--", alpha=0.6, label="50% Coin Flip")
ax_week.set_ylim(30, 100)
ax_week.set_ylabel("Accuracy %")
ax_week.set_xlabel("Week")
ax_week.set_title(f"{selected_eval_season} Machine Prediction Accuracy by Week", weight="bold")
st.pyplot(fig_week)

st.markdown("#### Team-by-Team Performance")

home_records = season_df[["home_team", "correct_pick"]].rename(columns={"home_team": "team"})
away_records = season_df[["away_team", "correct_pick"]].rename(columns={"away_team": "team"})
team_eval = pd.concat([home_records, away_records])

team_perf = team_eval.groupby("team")["correct_pick"].agg(
    total_games="count",
    correct_picks="sum"
).reset_index()

team_perf["accuracy_pct"] = (team_perf["correct_picks"] / team_perf["total_games"]) * 100
team_perf = team_perf.sort_values(by="accuracy_pct", ascending=False).reset_index(drop=True)

team_perf["Accuracy"] = team_perf["accuracy_pct"].map("{:.1f}%".format)
team_perf = team_perf.rename(columns={
    "team": "Team",
    "total_games": "Games Predicted",
    "correct_picks": "Correct Calls"
})

st.dataframe(team_perf[["Team", "Accuracy", "Correct Calls", "Games Predicted"]], use_container_width=True)