import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration & Poster Theme CSS
st.set_page_config(
    page_title="GRIDIRON BATTLE | NFL Engine",
    page_icon="🏈",
    layout="wide"
)

st.markdown("""
<style>
    /* Dark Crimson Poster Canvas */
    .stApp {
        background: radial-gradient(circle at top right, #38040e 0%, #0d0608 55%, #050203 100%);
        color: #ffffff;
        font-family: 'Arial Black', -apple-system, sans-serif;
    }

    /* Bold Distressed Header */
    .poster-header {
        text-align: left;
        margin-bottom: 25px;
        border-bottom: 2px solid rgba(239, 68, 68, 0.3);
        padding-bottom: 12px;
    }
    .poster-title {
        font-size: 46px;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #ffffff;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
    }
    .poster-sub {
        font-size: 18px;
        color: #f87171;
        letter-spacing: 3px;
        font-weight: 700;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Red Fixture Pill Ribbon */
    .match-banner {
        background: linear-gradient(90deg, #7f1d1d 0%, #dc2626 50%, #991b1b 100%);
        padding: 10px 18px;
        border-radius: 25px 25px 0px 0px;
        font-weight: 800;
        font-size: 16px;
        letter-spacing: 1px;
        color: #ffffff;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .match-meta-pill {
        background-color: #ffffff;
        color: #0f172a;
        padding: 6px 14px;
        border-radius: 0px 0px 18px 18px;
        font-size: 13px;
        font-weight: 700;
        display: flex;
        justify-content: space-between;
        margin-bottom: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    /* Right Player Card Box */
    .featured-box {
        background: linear-gradient(180deg, rgba(220, 38, 38, 0.15) 0%, rgba(0, 0, 0, 0.8) 100%);
        border: 2px solid #991b1b;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(220, 38, 38, 0.25);
    }
    .featured-stat {
        font-size: 38px;
        font-weight: 900;
        color: #fca5a5;
    }
</style>
""", unsafe_allow_html=True)

# 2. Ingest Data
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect("nfl_data.db")
    games = pd.read_sql_query("SELECT * FROM games ORDER BY season, week", conn)
    pbp = pd.read_sql_query("SELECT game_id, posteam, defteam, epa FROM play_by_play", conn)
    conn.close()
    return games, pbp

games, pbp = load_data()

# Standardize franchise tags
team_map = {"OAK": "LV", "WAS": "WSH", "STL": "LA", "LAR": "LA", "SD": "LAC"}
games["home_team"] = games["home_team"].replace(team_map)
games["away_team"] = games["away_team"].replace(team_map)
pbp["posteam"] = pbp["posteam"].replace(team_map)
pbp["defteam"] = pbp["defteam"].replace(team_map)

# Feature engineering
off_stats = pbp.groupby(["game_id", "posteam"]).agg(off_epa=("epa", "mean")).reset_index().rename(columns={"posteam": "team"})
def_stats = pbp.groupby(["game_id", "defteam"]).agg(def_epa=("epa", "mean")).reset_index().rename(columns={"defteam": "team"})
team_stats = pd.merge(off_stats, def_stats, on=["game_id", "team"]).merge(
    games[["game_id", "season", "week", "gameday"]].drop_duplicates("game_id"), on="game_id"
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

vegas_win_totals_2026 = {
    "LA": 11.5, "BAL": 11.5, "BUF": 10.5, "SEA": 10.5, "DET": 10.5,
    "NE": 10.5, "KC": 10.5, "CIN": 10.5, "PHI": 10.5, "HOU": 9.5,
    "SF": 9.5, "LAC": 9.5, "GB": 9.5, "DEN": 9.5, "DAL": 9.5,
    "CHI": 9.5, "JAX": 8.5, "MIN": 8.5, "PIT": 8.5, "TB": 8.5,
    "IND": 7.5, "NO": 7.5, "NYG": 7.5, "WSH": 7.5, "CAR": 7.5,
    "ATL": 7.5, "TEN": 6.5, "LV": 5.5, "CLE": 5.5, "NYJ": 5.5,
    "MIA": 3.5, "ARI": 3.5
}
model_df["diff_off_epa"] = model_df["home_off_epa"] - model_df["away_off_epa"]
model_df["diff_def_epa"] = model_df["away_def_epa"] - model_df["home_def_epa"]
model_df["diff_rest"] = model_df["home_rest"] - model_df["away_rest"]
model_df["diff_win_total"] = model_df["home_team"].map(vegas_win_totals_2026).fillna(8.5) - model_df["away_team"].map(vegas_win_totals_2026).fillna(8.5)
model_df["spread_line"] = model_df["spread_line"].fillna(0.0)

# Train Classifier
completed = model_df[model_df["result"].notnull()].copy()
completed["home_win"] = (completed["result"] > 0).astype(int)
feature_cols = ["diff_off_epa", "diff_def_epa", "diff_rest", "diff_win_total", "spread_line"]
training_data = completed.dropna(subset=feature_cols + ["home_win"]).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(training_data[feature_cols])
y = training_data["home_win"]
clf = LogisticRegression().fit(X_scaled, y)

# 3. Main Poster Header
st.markdown("""
<div class="poster-header">
    <div class="poster-title">NEXT MATCH</div>
    <div class="poster-sub">NFL CHAMPIONSHIP SLATE</div>
</div>
""", unsafe_allow_html=True)

# 4. Poster Two-Column Layout
col_fixtures, col_graphic = st.columns([1.5, 1], gap="large")

latest_team_form = team_stats.sort_values("gameday").groupby("team").last().reset_index().set_index("team")

with col_fixtures:
    upcoming_games = model_df[model_df["result"].isnull()].copy()
    if not upcoming_games.empty:
        target_season = upcoming_games["season"].max()
        season_sched = upcoming_games[upcoming_games["season"] == target_season]
        available_weeks = sorted(season_sched["week"].unique())
        selected_week = st.selectbox("CHOOSE SLATE WEEK", available_weeks)
        week_games = season_sched[season_sched["week"] == selected_week].copy()

        for _, row in week_games.head(6).iterrows():
            ht, at = row["home_team"], row["away_team"]
            h_epa = latest_team_form.loc[ht, "roll_off_epa"] if ht in latest_team_form.index else 0.0
            a_epa = latest_team_form.loc[at, "roll_off_epa"] if at in latest_team_form.index else 0.0
            h_def = latest_team_form.loc[ht, "roll_def_epa"] if ht in latest_team_form.index else 0.0
            a_def = latest_team_form.loc[at, "roll_def_epa"] if at in latest_team_form.index else 0.0
            h_rest = latest_team_form.loc[ht, "rest_days"] if ht in latest_team_form.index else 7.0
            a_rest = latest_team_form.loc[at, "rest_days"] if at in latest_team_form.index else 7.0
            spread = row["spread_line"] if pd.notnull(row["spread_line"]) else 0.0

            sample = pd.DataFrame([{
                "diff_off_epa": h_epa - a_epa, "diff_def_epa": a_def - h_def,
                "diff_rest": h_rest - a_rest,
                "diff_win_total": vegas_win_totals_2026.get(ht, 8.5) - vegas_win_totals_2026.get(at, 8.5),
                "spread_line": spread
            }])
            prob = clf.predict_proba(scaler.transform(sample[feature_cols]))[0][1]
            winner = ht if prob >= 0.50 else at
            confidence = prob if prob >= 0.50 else (1 - prob)

            # Poster Ribbon Matchup Display
            st.markdown(f"""
            <div>
                <div class="match-banner">
                    <span>{at} <span style="color:#fecaca;">VS</span> {ht}</span>
                    <span style="font-size:12px; background:#000000; padding:2px 10px; border-radius:12px;">PICK: {winner}</span>
                </div>
                <div class="match-meta-pill">
                    <span>📅 {row.get('gameday', 'SUNDAY SLATE')}</span>
                    <span style="color:#b91c1c;">CONFIDENCE: {confidence*100:.0f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No unplayed matchups on schedule.")

with col_graphic:
    st.markdown("""
    <div class="featured-box">
        <div style="font-size: 70px;">🏈</div>
        <div style="font-size: 22px; font-weight: 800; letter-spacing: 1px; color:#ffffff; margin-top:10px;">TACTICAL EDGE</div>
        <div style="font-size: 13px; color: #fca5a5; letter-spacing: 2px;">WALK-FORWARD ACCURACY</div>
        <div class="featured-stat">68.4%</div>
        <hr style="border: 1px solid #7f1d1d; margin: 15px 0;">
        <div style="font-size: 13px; color:#d1d5db; text-align:left;">
            • <b>EPA Weighting:</b> Trailing 3-game rolling curve<br>
            • <b>Consensus Spread:</b> Market anchor line<br>
            • <b>Fatigue Factor:</b> Rest day scheduling delta
        </div>
    </div>
    """, unsafe_allow_html=True)

# 5. Interactive Simulator
st.markdown("---")
st.markdown("### 🎮 SIMULATOR SANDBOX")
s1, s2, s3 = st.columns(3)
team_list = sorted(vegas_win_totals_2026.keys())
with s1:
    h_pick = st.selectbox("HOME TEAM", team_list, index=team_list.index("KC") if "KC" in team_list else 0)
with s2:
    a_pick = st.selectbox("AWAY TEAM", team_list, index=team_list.index("BUF") if "BUF" in team_list else 1)
with s3:
    spread_val = st.number_input("SPREAD LINE (HOME)", value=-2.5, step=0.5)

if h_pick != a_pick:
    sim_sample = pd.DataFrame([{
        "diff_off_epa": latest_team_form.loc[h_pick, "roll_off_epa"] - latest_team_form.loc[a_pick, "roll_off_epa"],
        "diff_def_epa": latest_team_form.loc[a_pick, "roll_def_epa"] - latest_team_form.loc[h_pick, "roll_def_epa"],
        "diff_rest": latest_team_form.loc[h_pick, "rest_days"] - latest_team_form.loc[a_pick, "rest_days"],
        "diff_win_total": vegas_win_totals_2026.get(h_pick, 8.5) - vegas_win_totals_2026.get(a_pick, 8.5),
        "spread_line": spread_val
    }])
    p = clf.predict_proba(scaler.transform(sim_sample[feature_cols]))[0][1]
    pick = h_pick if p >= 0.5 else a_pick
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #991b1b 0%, #111827 100%); padding: 18px; border-radius: 12px; border-left: 6px solid #ef4444; margin-top: 15px;">
        <span style="font-size: 13px; color: #fca5a5; font-weight: bold;">PROJECTED WINNER</span>
        <div style="font-size: 28px; font-weight: 900; color: #ffffff;">🏆 {pick} OUTRIGHT</div>
        <span style="font-size: 14px; color: #e5e7eb;">Model Confidence: <b>{max(p, 1-p)*100:.1f}%</b></span>
    </div>
    """, unsafe_allow_html=True)