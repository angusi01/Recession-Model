import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import THRESHOLDS, URLS
from data_builder import build_feature_matrix, get_current_feature_row
from data_sources import (
    fetch_abs_data, fetch_asic_insolvency, fetch_brent_crude,
    fetch_google_trends, fetch_kalshi_recession_odds, fetch_official_keywords,
    fetch_rba_csv, fetch_real_wage_growth, fetch_westpac_sentiment,
    fetch_yield_curve_spread,
)
from history import get_and_update_history
from model import calculate_total_probability
from model_ml import (
    compute_backtest_metrics,
    get_current_forecast,
    train_final_models,
    walk_forward_predict,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="AU Recession Forecast — ML Leading Indicator Model",
    layout="wide",
)


# ── Gauge helpers ─────────────────────────────────────────────────────────────

def _gauge_color(prob: float) -> str:
    if prob < 30:
        return "green"
    if prob < 55:
        return "#FFC107"
    if prob < 75:
        return "orange"
    return "red"


def display_dual_gauges(p_3m: float | None, p_6m: float | None) -> None:
    """Render two side-by-side recession probability gauges (3m and 6m)."""
    col_a, col_b = st.columns(2)
    for col, prob, title in [
        (col_a, p_3m, "3-Month Recession Probability"),
        (col_b, p_6m, "6-Month Recession Probability"),
    ]:
        with col:
            val = prob if prob is not None else 0.0
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": title, "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkgray"},
                    "steps": [
                        {"range": [0, 30], "color": "green"},
                        {"range": [30, 55], "color": "#FFC107"},
                        {"range": [55, 75], "color": "orange"},
                        {"range": [75, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": val,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
            if prob is None:
                st.caption("⚠️ Model not yet available — insufficient historical data.")


# ── Historical probability chart ──────────────────────────────────────────────

def display_probability_trend(wf_df: pd.DataFrame, recession_series: pd.Series) -> None:
    """Plot the walk-forward OOS probability trend with recession shading."""
    if wf_df.empty:
        st.info("Walk-forward backtest data not yet available.")
        return

    # Convert PeriodIndex to timestamp for Plotly
    def _to_ts(idx):
        if hasattr(idx, "to_timestamp"):
            return idx.to_timestamp()
        return pd.to_datetime(idx.astype(str))

    dates = _to_ts(wf_df.index)
    fig = go.Figure()

    # Recession shading
    if not recession_series.empty:
        rec = recession_series.reindex(wf_df.index).fillna(0)
        in_rec = False
        rec_start = None
        rec_dates = _to_ts(rec.index)
        for i, (d, v) in enumerate(zip(rec_dates, rec.values)):
            if v == 1 and not in_rec:
                rec_start = d
                in_rec = True
            elif v == 0 and in_rec:
                fig.add_vrect(
                    x0=rec_start, x1=d,
                    fillcolor="rgba(220,50,50,0.15)",
                    line_width=0,
                    annotation_text="Recession",
                    annotation_position="top left",
                )
                in_rec = False
        if in_rec and rec_start is not None:
            fig.add_vrect(x0=rec_start, x1=dates[-1], fillcolor="rgba(220,50,50,0.15)", line_width=0)

    # 50% threshold line
    fig.add_hline(y=50, line_dash="dash", line_color="orange",
                  annotation_text="50% threshold", annotation_position="bottom right")

    if "p_ens_3m" in wf_df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=(wf_df["p_ens_3m"] * 100).round(1),
            mode="lines", name="3m Forecast", line=dict(color="#1f77b4", width=2),
        ))
    if "p_ens_6m" in wf_df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=(wf_df["p_ens_6m"] * 100).round(1),
            mode="lines", name="6m Forecast", line=dict(color="#ff7f0e", width=2, dash="dot"),
        ))

    fig.update_layout(
        title="Historical Walk-Forward Recession Probability (Out-of-Sample)",
        xaxis_title="Date",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Feature importance chart ──────────────────────────────────────────────────

def display_feature_importance(importance: dict) -> None:
    """Display LR coefficient bar chart (positive = increases recession risk)."""
    if not importance:
        st.info("Feature importance not available.")
        return

    sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    names = [k.replace("_", " ").title() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="Model Drivers — LR Coefficients (3m, red = increases risk)",
        xaxis_title="Coefficient",
        height=380,
        margin=dict(l=160),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Backtest metrics display ──────────────────────────────────────────────────

def display_backtest_metrics(metrics: dict) -> None:
    """Show ROC-AUC, Brier score and early-detection lead time."""
    m = st.columns(5)

    def _fmt(val, decimals=3, suffix=""):
        return f"{val:.{decimals}f}{suffix}" if val is not None else "N/A"

    m[0].metric("ROC-AUC (3m)", _fmt(metrics.get("roc_auc_3m"), 3))
    m[1].metric("Brier Score (3m)", _fmt(metrics.get("brier_3m"), 4))
    m[2].metric("ROC-AUC (6m)", _fmt(metrics.get("roc_auc_6m"), 3))
    m[3].metric("Brier Score (6m)", _fmt(metrics.get("brier_6m"), 4))
    avg_early = metrics.get("avg_months_early")
    m[4].metric(
        "Avg Early Detection",
        f"{avg_early:.1f} mo" if avg_early is not None else "N/A",
        help="Average months before recession onset that the 3m probability exceeded 50%",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🇦🇺 AU Recession Forecast — ML Leading Indicator Model")
    st.markdown(
        "**True walk-forward model** — all features lag-adjusted for real-time availability. "
        "No lookahead bias. Targets: *recession probability over next 3 months* and *6 months*."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("ℹ️ Model Info")
    st.sidebar.markdown(
        "**Architecture**: Logistic Regression + Gradient Boosting ensemble, "
        "calibrated via isotonic regression on walk-forward out-of-sample predictions.\n\n"
        "**Recession definition**: 2 consecutive quarters of negative GDP growth (ABS).\n\n"
        "**Targets**: y_3m = recession in next 3 months; y_6m = in next 6 months.\n\n"
        "**Data freshness** is shown at the bottom of the page."
    )

    # ── Step 1: Build historical feature matrix (cached 24 h) ────────────────
    with st.spinner("📡 Fetching historical data and building feature matrix…"):
        try:
            matrix = build_feature_matrix()
            features_df = matrix["features"]
            y_3m = matrix["y_3m"]
            y_6m = matrix["y_6m"]
            recession_series = matrix["recession"]
            matrix_ok = not features_df.empty
        except Exception as e:
            st.error(f"Failed to build feature matrix: {e}")
            matrix_ok = False
            features_df = pd.DataFrame()
            y_3m = y_6m = recession_series = pd.Series(dtype=int)

    # ── Step 2: Walk-forward backtest (cached 24 h) ───────────────────────────
    with st.spinner("🔄 Running walk-forward validation…"):
        if matrix_ok:
            try:
                wf_df = walk_forward_predict(features_df, y_3m, y_6m)
            except Exception as e:
                st.warning(f"Walk-forward failed: {e}")
                wf_df = pd.DataFrame()
        else:
            wf_df = pd.DataFrame()

    # ── Step 3: Train final models + calibrate (cached 24 h) ─────────────────
    with st.spinner("🤖 Training final models…"):
        if matrix_ok:
            try:
                models = train_final_models(features_df, y_3m, y_6m, wf_df)
            except Exception as e:
                st.warning(f"Model training failed: {e}")
                models = {}
        else:
            models = {}

    # ── Step 4: Current forecast ──────────────────────────────────────────────
    current_row = get_current_feature_row(features_df) if matrix_ok else None
    if current_row is not None and models:
        forecast = get_current_forecast(current_row, models)
    else:
        forecast = {"p_3m": None, "p_6m": None, "feature_importance": {}}

    # ── Section 1: Dual gauges ────────────────────────────────────────────────
    st.subheader("📊 Current Recession Probability Forecast")
    display_dual_gauges(forecast.get("p_3m"), forecast.get("p_6m"))

    # Probability interpretation text
    if forecast.get("p_3m") is not None:
        p3 = forecast["p_3m"]
        p6 = forecast["p_6m"]
        if p3 >= 60:
            st.error(f"🔴 **High recession risk** — 3m: {p3:.1f}% | 6m: {p6:.1f}%")
        elif p3 >= 35:
            st.warning(f"🟠 **Elevated recession risk** — 3m: {p3:.1f}% | 6m: {p6:.1f}%")
        else:
            st.success(f"🟢 **Low recession risk** — 3m: {p3:.1f}% | 6m: {p6:.1f}%")

    # Show individual model outputs
    with st.expander("🔍 Model component outputs"):
        comp_cols = st.columns(4)
        for i, (label, key) in enumerate([
            ("LR 3m", "p_lr_3m"), ("GB 3m", "p_gb_3m"),
            ("LR 6m", "p_lr_6m"), ("GB 6m", "p_gb_6m"),
        ]):
            val = forecast.get(key)
            comp_cols[i].metric(label, f"{val:.1f}%" if val is not None else "N/A")

    st.divider()

    # ── Section 2: Historical probability trend ───────────────────────────────
    st.subheader("📈 Historical Walk-Forward Probability (Backtest)")
    display_probability_trend(wf_df, recession_series)

    st.divider()

    # ── Section 3: Backtest metrics ───────────────────────────────────────────
    st.subheader("📐 Backtest Metrics (Out-of-Sample)")
    if not wf_df.empty:
        metrics = compute_backtest_metrics(wf_df)
        display_backtest_metrics(metrics)
        st.caption(
            "ROC-AUC > 0.7 = useful. Brier score < 0.1 = well-calibrated. "
            "'Avg Early Detection' = months before recession onset where 3m P > 50%."
        )
    else:
        st.info("Backtest metrics require walk-forward data (loading…).")

    st.divider()

    # ── Section 4: Feature importance ────────────────────────────────────────
    st.subheader("🔑 Top Model Drivers (LR Coefficients — 3m Horizon)")
    display_feature_importance(forecast.get("feature_importance", {}))

    st.divider()

    # ── Section 5: Live data indicators (kept from original system) ───────────
    st.subheader("📡 Live Indicator Readings")

    with st.spinner("Fetching live economic data…"):
        trends_val, trends_err = fetch_google_trends()
        yield_curve_val = fetch_yield_curve_spread(URLS["rba_yield_curve"])

        live_data = {
            "yield_curve": yield_curve_val,
            "iron_ore": fetch_rba_csv(URLS["rba_commodity_prices"], "Iron ore", "iron_ore"),
            "gdp_qq": fetch_abs_data("NA/1.1.1.20.Q", "gdp_qq"),
            "unemployment": fetch_abs_data("LF/1.3.1599.20.M", "unemployment"),
            "cpi_trimmed": fetch_abs_data("CPI/1.10002.10.20.Q", "cpi_trimmed"),
            "real_wage_growth": fetch_real_wage_growth(),
            "insolvency_rate": fetch_asic_insolvency(),
            "brent_crude": fetch_brent_crude(),
            "westpac_sentiment": fetch_westpac_sentiment(),
            "google_trends": trends_val,
            "keyword_hits": fetch_official_keywords(),
            "kalshi_recession": fetch_kalshi_recession_odds(),
            "anz_job_ads": -5.0,
        }

    if trends_err:
        st.warning("⚠️ Google Trends: API rate limit. Using cached value.")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Yield Curve (10y-2y)", f"{live_data['yield_curve']:.2f}%")
    m2.metric("Iron Ore (USD)", f"${live_data['iron_ore']:.0f}")
    m3.metric("GDP q/q", f"{live_data['gdp_qq']:.2f}%")
    m4.metric("Unemployment", f"{live_data['unemployment']:.1f}%")
    m5.metric("Trimmed CPI", f"{live_data['cpi_trimmed']:.1f}%")
    m6.metric("Real Wage Growth", f"{live_data['real_wage_growth']:.2f}%")

    m7, m8, m9, m10 = st.columns(4)
    m7.metric("Insolvency Rate", f"{live_data['insolvency_rate']:.2f}%")
    m8.metric("Brent Crude", f"${live_data['brent_crude']:.0f}")
    m9.metric("Westpac Sentiment", f"{live_data['westpac_sentiment']:.1f}")
    m10.metric("Kalshi US Recession", f"{live_data['kalshi_recession']:.1f}%")

    st.divider()

    # ── Section 6: Legacy scoring model (reference) ───────────────────────────
    with st.expander("📋 Legacy Rule-Based Score (Reference Only)"):
        st.markdown(
            "_The original weighted scoring model is shown here as a reference. "
            "The ML model above is the primary forecast._"
        )
        results_legacy = calculate_total_probability(live_data)
        st.metric(
            "Legacy Score",
            f"{results_legacy['total_probability']:.1f}%",
            help="Weighted indicator scoring model (not ML-based).",
        )
        breakdown = {**results_legacy["contributions"], **results_legacy["overlays"]}
        df_breakdown = pd.DataFrame(
            list(breakdown.items()), columns=["Factor", "Contribution (%)"]
        )
        fig_bar = go.Figure(go.Bar(
            x=df_breakdown["Contribution (%)"],
            y=df_breakdown["Factor"],
            orientation="h",
        ))
        fig_bar.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=350,
            title="Legacy Score Breakdown",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Section 7: Methodology ────────────────────────────────────────────────
    with st.expander("📖 Methodology & Anti-Lookahead Design"):
        st.markdown("""
        ## True Leading Recession Prediction System

        ### No-Lookahead Principle
        At any time **T**, the model uses only data available at T:
        - **GDP** (quarterly) → lagged 1 quarter (ABS releases ~90 days after quarter end)
        - **CPI trimmed** (quarterly) → lagged 1 quarter
        - **Unemployment** (monthly) → lagged 1 month (~6-week release lag)
        - **Market data** (yield curve, iron ore, ASX200) → no lag required

        ### Recession Definition
        Two consecutive quarters of negative GDP growth (q/q %, ABS National Accounts).

        ### Forward Targets
        - **y_3m**: recession occurs in any of the next 3 months
        - **y_6m**: recession occurs in any of the next 6 months

        ### Features
        | Feature | Description |
        |---------|-------------|
        | yield_curve_slope | 10Y CGS − 90-day bank bill rate (inversion = leading signal) |
        | yield_curve_3m_delta | 3-month change in slope |
        | yield_curve_zscore | 5-year rolling z-score |
        | unemployment | ABS unemployment rate (lagged 1 month) |
        | unemployment_3m_change | 3-month change in unemployment (lagged) |
        | unemployment_change_zscore | 5-year rolling z-score of 3m change |
        | gdp_qq | ABS GDP q/q % change (lagged 1 quarter) |
        | gdp_4q_sum | Rolling 4-quarter GDP sum (annualised momentum) |
        | cpi_trimmed | ABS trimmed mean CPI (lagged 1 quarter) |
        | iron_ore | RBA iron ore USD spot price |
        | iron_ore_3m_pct | 3-month % change in iron ore |
        | iron_ore_zscore | 5-year rolling z-score |
        | asx200_drawdown | Rolling 12-month drawdown from peak |
        | asx200_3m_return | 3-month equity market return |

        ### Walk-Forward Validation
        Expanding window: train on [start → t], predict at t+1.
        Minimum 5 years of training data required. Re-trains every 6 months.

        ### Models
        - **Logistic Regression** (L2, class_weight='balanced') — interpretable baseline
        - **Gradient Boosting** (shallow trees, 60 estimators) — nonlinear signals
        - **Ensemble**: simple average of both, then isotonic calibration

        ### Calibration
        Isotonic regression fitted on walk-forward OOS predictions ensures
        probabilities are well-calibrated (meaningful as true probabilities).
        """)

    # ── Data freshness ────────────────────────────────────────────────────────
    st.caption(
        f"🕒 Data fetched: **{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}** | "
        f"Feature matrix as of: **{str(features_df.index[-1]) if not features_df.empty else 'N/A'}** | "
        f"Historical data source: ABS, RBA, yfinance (ASX200)"
    )


if __name__ == "__main__":
    main()

