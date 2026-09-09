import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
st.set_page_config(
    page_title="Gridiron Edge | Game Slate",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS Matching the Crimson/Black Athletic Flyer
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Teko:wght@600;700&family=Inter:wght@400;600;800&display=swap');

    /* Global Dark Canvas */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #3b070c 0%, #150204 45%, #080102 100%);
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Poster Slate Container */
    .poster-container {
        background-color: #0b0204;
        border: 2px solid #50070d;
        border-radius: 24px;
        padding: 28px 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8), 0 0 40px rgba(185, 28, 28, 0.15);
        margin-bottom: 25px;
    }

    /* Flyer Typography */
    .flyer-title {
        font-family: 'Teko', sans-serif;
        font-size: 58px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        line-height: 0.9;
        color: #ffffff;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.9);
        margin: 0;
    }
    .flyer-subtitle {
        font-family: 'Teko', sans-serif;
        font-size: 26px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #e2e8f0;
        margin-bottom: 24px;
        border-bottom: 1px solid #450a0a;
        padding-bottom: 8px;
    }

    /* Matchup Row Styles */
    .matchup-pill {
        background: linear-gradient(90deg, #b91c1c 0%, #7f1d1d 70%, #450a0a 100%);
        border-radius: 30px;
        padding: 10px 20px;
        font-weight: 800;
        font-size: 17px;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(185, 28, 28, 0.35);
        margin-top: 14px;
    }
    .matchup-pill-upset {
        background: linear-gradient(90deg, #d97706 0%, #b45309 70%, #78350f 100%);
        box-shadow: 0 4px 12px rgba(217, 119, 6, 0.35);
    }
    
    .meta-row {
        display: flex;
        gap: 10px;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    .white-pill {
        background-color: #ffffff;
        color: #0b0204;
        font-weight: 700;
        font-size: 13px;
        padding: 6px 16px;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    .edge-pill {
        background-color: #260508;
        border: 1px solid #7f1d1d;
        color: #fca5a5;
        font-weight: 600;
        font-size: 12px;
        padding: 6px 14px;
        border-radius: 20px;
    }

    /* Interactive Simulator Panel */
    .sim-panel {
        background: linear-gradient(180deg, #180306 0%, #0c0102 100%);
        border: 1px solid #7f1d1d;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7);
    }

    /* Tab navigation theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a0306;
        border: 1px solid #450a0a;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        color: #e2e8f0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #b91c1c !important;
        border-color: #ef4444 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Data Ingestion & Model Pipeline
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect("nfl_data.db")
    games = pd.read_sql_query("SELECT * FROM games ORDER BY season, week", conn)
    pbp = pd.read_sql_query("SELECT game_id, posteam, defteam, epa FROM play_by_play", conn)
    conn.close()
    return games, pbp

games, pbp = load_data()

team_map = {"OAK": "LV", "WAS": "WSH", "STL": "LA", "LAR": "LA", "SD": "LAC"}
games["home_team"] = games["home_team"].replace(team_map)
games["away_team"] = games["away_team"].replace(team_map)
pbp["posteam"] = pbp["posteam"].replace(team_map)
pbp["defteam"] = pbp["defteam"].replace(team_map)

off_stats = pbp.groupby(["game_id", "posteam"]).agg(off_epa=("epa", "mean")).reset_index().rename(columns={"posteam": "team"})
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

completed_games = model_df[model_df["result"].notnull()].copy()
completed_games["home_win"] = (completed_games["result"] > 0).astype(int)

feature_cols = ["diff_off_epa", "diff_def_epa", "diff_rest", "diff_win_total", "spread_line"]
training_data = completed_games.dropna(subset=feature_cols + ["home_win"]).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(training_data[feature_cols])
y = training_data["home_win"]

clf = LogisticRegression()
clf.fit(X_scaled, y)

latest_team_form = team_stats.sort_values("gameday").groupby("team").last().reset_index().set_index("team")

# 4. Tab Navigation
tab_main, tab_perf, tab_analytics = st.tabs(["🔥 MATCH SLATE & SIMULATOR", "📈 PERFORMANCE TRACKER", "📊 ADVANCED ANALYTICS"])

# TAB 1: POSTER SLATE & SIMULATOR
with tab_main:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown("""
        <div class="poster-container">
            <div class="flyer-title">NEXT MATCH</div>
            <div class="flyer-subtitle">NFL PREDICTIVE SLATE 2026</div>
        """, unsafe_allow_html=True)

        upcoming_games = model_df[model_df["result"].isnull()].copy()
        if not upcoming_games.empty:
            target_season = upcoming_games["season"].max()
            season_sched = upcoming_games[upcoming_games["season"] == target_season]
            available_weeks = sorted(season_sched["week"].unique())
            selected_week = st.selectbox("Select Week", available_weeks, label_visibility="collapsed")
            week_games = season_sched[season_sched["week"] == selected_week].copy()

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
                    "diff_off_epa": h_epa - a_epa, "diff_def_epa": a_def - h_def,
                    "diff_rest": h_rest - a_rest, "diff_win_total": h_wt - a_wt, "spread_line": spread
                }])
                prob = clf.predict_proba(scaler.transform(sample[feature_cols]))[0][1]
                winner = ht if prob >= 0.50 else at
                confidence = prob if prob >= 0.50 else (1 - prob)

                is_upset = (winner == ht and spread > 0) or (winner == at and spread < 0)
                pill_class = "matchup-pill-upset" if is_upset else "matchup-pill"
                pick_badge = f"⚡ UPSET: {winner}" if is_upset else f"PICK: {winner}"

                st.markdown(f"""
                <div class="{pill_class}">
                    <span>{at} VS {ht}</span>
                    <span style="font-size: 13px; font-weight: 800; background: rgba(0,0,0,0.3); padding: 4px 10px; border-radius: 12px;">{pick_badge}</span>
                </div>
                <div class="meta-row">
                    <span class="white-pill">📅 {row.get('gameday', 'Upcoming Slate')}</span>
                    <span class="white-pill">Spread: {spread:+.1f}</span>
                    <span class="edge-pill">Win Confidence: {confidence*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No unplayed games scheduled in current database. Test custom matchups in the simulator.")

        st.markdown("""
            <div style="margin-top: 25px; font-family:'Teko'; font-size:20px; color:#9ca3af; letter-spacing:1px;">
                VENUE / HOME FIELD ADVANTAGE INCLUDED IN EPA REGRESSION
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT COLUMN: HELMET/PLAYER BRANDING & SIMULATOR
    with col_right:
        # High quality transparent background player asset
        img_url = "https://ewscripps.brightspotcdn.com/dims4/default/b611cf8/2147483647/strip/true/crop/5175x2911+0+0/resize/1280x720!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2Fb2%2Fde%2Ffc83212f4759a96c03b6caca7946%2Fap21269635997659.jpg"
        st.image(img_url, use_container_width=True)

        st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
        st.markdown("<h3 style='font-family:Teko; font-size:32px; margin:0 0 10px 0; color:#fff;'>⚡ MATCHUP SANDBOX</h3>", unsafe_allow_html=True)

        team_list = sorted(vegas_win_totals_2026.keys())
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            home_team = st.selectbox("Home Squad", team_list, index=team_list.index("KC") if "KC" in team_list else 0)
        with sim_col2:
            away_team = st.selectbox("Away Squad", team_list, index=team_list.index("BUF") if "BUF" in team_list else 1)
            
        spread_val = st.slider("Vegas Line (Home Team)", -14.0, 14.0, -2.5, step=0.5)

        if home_team != away_team:
            h_form = latest_team_form.loc[home_team]
            a_form = latest_team_form.loc[away_team]
            diff_o = h_form["roll_off_epa"] - a_form["roll_off_epa"]
            diff_d = a_form["roll_def_epa"] - h_form["roll_def_epa"]
            diff_r = h_form["rest_days"] - a_form["rest_days"]
            diff_w = vegas_win_totals_2026.get(home_team, 8.5) - vegas_win_totals_2026.get(away_team, 8.5)

            sim_sample = pd.DataFrame([{
                "diff_off_epa": diff_o, "diff_def_epa": diff_d, "diff_rest": diff_r,
                "diff_win_total": diff_w, "spread_line": spread_val
            }])

            p = clf.predict_proba(scaler.transform(sim_sample[feature_cols]))[0][1]
            sim_winner = home_team if p >= 0.50 else away_team
            sim_conf = p if p >= 0.50 else (1.0 - p)

            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #b91c1c 0%, #450a0a 100%); border-radius: 12px; padding: 16px; margin: 15px 0; text-align: center;">
                <span style="font-size: 13px; text-transform: uppercase; color: #fca5a5; font-weight:700;">Projected Victor</span>
                <div style="font-size: 28px; font-weight: 800; color: #fff; font-family:'Teko'; letter-spacing:1px; margin: 4px 0;">🏆 {sim_winner} OUTRIGHT</div>
                <span style="font-size: 14px; color: #ffffff;">Model Win Expectancy: <b>{sim_conf*100:.1f}%</b></span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(p))

        st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: MODEL ACCURACY TRACKER
with tab_perf:
    st.markdown("<h3 style='font-family:Teko; font-size:36px;'>MACHINE HIT-RATE & PROGRESSION</h3>", unsafe_allow_html=True)
    completed = training_data.copy()
    completed["pred_prob"] = clf.predict_proba(X_scaled)[:, 1]
    completed["pred_home_win"] = (completed["pred_prob"] >= 0.50).astype(int)
    completed["correct_pick"] = (completed["pred_home_win"] == completed["home_win"]).astype(int)

    eval_seasons = sorted(completed["season"].unique(), reverse=True)
    selected_eval_season = st.selectbox("Season Evaluation", eval_seasons)

    season_df = completed[completed["season"] == selected_eval_season].copy()
    total_g = len(season_df)
    correct_p = season_df["correct_pick"].sum()
    acc = (correct_p / total_g) * 100 if total_g > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Straight-Up Win %", f"{acc:.1f}%")
    m2.metric("Total Correct Hits", f"{correct_p} / {total_g}")
    m3.metric("Underdog Pick Hits", f"{len(season_df[(season_df['correct_pick'] == 1) & (season_df['spread_line'] > 0)])}")

    weekly_stats = season_df.groupby("week")["correct_pick"].agg(Total_Games="count", Correct_Calls="sum").reset_index()
    weekly_stats["Weekly Accuracy"] = (weekly_stats["Correct_Calls"] / weekly_stats["Total_Games"]) * 100

    fig_w, ax_w = plt.subplots(figsize=(10, 3.5))
    fig_w.patch.set_facecolor('#0b0204')
    ax_w.set_facecolor('#1a0306')
    ax_w.plot(weekly_stats["week"], weekly_stats["Weekly Accuracy"], marker='o', color='#ef4444', linewidth=2.5, label="Weekly Hit %")
    ax_w.axhline(50, color='#64748b', linestyle=':', label="50% Coin Flip")
    ax_w.tick_params(colors="white")
    ax_w.set_title(f"{selected_eval_season} Accuracy Swing by Week", color="white", weight="bold")
    ax_w.legend(facecolor='#0b0204', edgecolor='none', labelcolor='white')
    st.pyplot(fig_w)

# TAB 3: SEABORN CORRELATIONS
with tab_analytics:
    st.markdown("<h3 style='font-family:Teko; font-size:36px;'>ADVANCED REGRESSION WEIGHTS</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    sns.set_theme(style="darkgrid")
    
    with c1:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        fig1.patch.set_facecolor('#0b0204')
        ax1.set_facecolor('#1a0306')
        corr = training_data[["home_win", "spread_line", "diff_off_epa", "diff_def_epa", "diff_win_total"]].corr()
        sns.heatmap(corr, annot=True, cmap="Reds", center=0, fmt=".2f", linewidths=0.5, ax=ax1, cbar=False)
        ax1.set_title("Correlation Matrix", color="white", weight="bold")
        ax1.tick_params(colors="white")
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor('#0b0204')
        ax2.set_facecolor('#1a0306')
        sns.kdeplot(data=training_data, x="diff_win_total", hue="home_win", fill=True, palette=["#ef4444", "#ffffff"], ax=ax2)
        ax2.set_title("Vegas Win Margin (Wins vs Losses)", color="white", weight="bold")
        ax2.tick_params(colors="white")
        st.pyplot(fig2)