**NFL Game Predictor & Analytics Engine**

An end-to-end sports analytics pipeline and machine learning web application that forecasts weekly NFL matchups, flags outright betting upsets, and tracks walk-forward model performance across seasons.

**Live Web Application:** [Launch Streamlit App](https://nfl-game-predictor-bertyc8.streamlit.app/)

Project Overview
- **Automated Winner Forecasts:** Predicts straight-up weekly NFL game outcomes and model confidence percentages using a calibrated Logistic Regression classifier.
- **Upset  Detection:** Identifies market discrepancies by comparing model win expectancies directly against consensus Vegas spread lines.
- **Historical Backtesting & Tracking:** Evaluates model precision with weekly hit-rate progression charts and team-by-team prediction breakdowns.
- **Interactive Matchup Sandbox:** Allows users to simulate any custom head-to-head matchup with adjustable spread margins.

Tech Stack
- **Language & Modeling:** Python 3.10+, `scikit-learn` (Logistic Regression, StandardScaler), `numpy`
- **Data Ingestion & Storage:** `nflreadpy`, `pandas`, SQLite (`nfl_data.db`)
- **Dashboard & Visualization:** `streamlit`, `seaborn`, `matplotlib`

Feature Engineering & Data Pipeline

To prevent lookahead data leakage, all performance metrics are shifted by one game prior to rolling calculation:

- **Offensive EPA/Play:** 3-game trailing rolling average measuring scoring efficiency.
- **Defensive EPA/Play Suppressed:** 3-game trailing rolling average measuring defensive stop efficiency.
- **Rest Differentials:** Net rest days between opponents (capturing short weeks, bye weeks, and Thursday Night Football).
- **Preseason Anchor Strength:** Consensus preseason Vegas season win totals.
- **Market Baseline:** Vegas spread consensus to account for injuries and external market data.
