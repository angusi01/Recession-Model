import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

from config import THRESHOLDS, WEIGHTS, URLS
from data_sources import (
    fetch_abs_data, fetch_rba_csv, fetch_asic_insolvency,
    fetch_brent_crude, fetch_asx_futures, fetch_westpac_sentiment,
    fetch_google_trends, fetch_official_keywords, fetch_kalshi_recession_odds
)
from model import calculate_total_probability
from history import get_and_update_history

st.set_page_config(page_title="Australia Recession Probability Monitor", layout="wide")

def display_gauge(probability):
    """Render the main probability gauge."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        title = {'text': "Recession Probability"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgray"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 55], 'color': "yellow"},
                {'range': [55, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    return fig

def generate_dynamic_advice(probability, raw_data):
    """Generate insight text based on current readings."""
    advice = "### Market Insight\n"
    if probability < 30:
        advice += "The economy is running reliably clear of recessionary thresholds. No immediate action triggers noted."
    elif 30 <= probability < 60:
        advice += "Elevated risks present. Close monitoring of employment and inflation trends is advised."
    else:
        advice += "🔴 **High Recession Risk.** To exit the danger zone:\n"
        # Check which metrics are nearest danger
        if raw_data["gdp_qq"] <= THRESHOLDS["gdp_qq"]["safe"]:
            advice += f"- GDP must print at least {THRESHOLDS['gdp_qq']['safe']}% q/q.\n"
        if raw_data["unemployment"] >= THRESHOLDS["unemployment"]["danger"] - 0.5:
            advice += f"- Unemployment needs to fall back below {THRESHOLDS['unemployment']['safe']}%. \n"
        if raw_data["cpi_headline"] >= THRESHOLDS["cpi_headline"]["danger"] - 1.0:
            advice += f"- Headline CPI must drop toward the RBA {THRESHOLDS['cpi_headline']['safe']}% band.\n"
    
    return advice

def main():
    st.title("Australia Recession Probability Monitor 📉")
    st.write("A live-updating gauge of macroeconomic stability, forward-priced using market signals.")
    
    with st.spinner("Fetching live economic data..."):
        # Fetching all required data
        trends_val, trends_err = fetch_google_trends()
        
        raw_data = {
            "gdp_qq": fetch_abs_data("NA/1.1.1.20.Q", "gdp_qq"),
            "unemployment": fetch_abs_data("LF/1.3.1599.20.M", "unemployment"),
            "cpi_headline": fetch_abs_data("CPI/1.10001.10.20.Q", "cpi_headline"),
            "cpi_trimmed": fetch_abs_data("CPI/1.10002.10.20.Q", "cpi_trimmed"),
            "cash_rate": fetch_rba_csv(URLS["rba_cash_rate"], "Cash Rate Target", "cash_rate"),
            "real_wage_growth": fetch_abs_data("WPI/1.3.999901.20.Q", "real_wage_growth") - fetch_abs_data("CPI/1.10001.10.20.Q", "cpi_headline"),
            "insolvency_rate": fetch_asic_insolvency(),
            "brent_crude": fetch_brent_crude(),
            "asx_cash_rate": fetch_asx_futures(),
            "westpac_sentiment": fetch_westpac_sentiment(),
            "google_trends": trends_val,
            "keyword_hits": fetch_official_keywords(),
            "kalshi_recession": fetch_kalshi_recession_odds(),
        }

    if trends_err:
        st.warning("⚠️ **Google Trends Alert:** API rate limit reached. Using last known cached value for consumer sentiment overlay.")

    # Model Calculation (Base Case)
    results_base = calculate_total_probability(raw_data)
    
    # Save/load history diff
    merged_contribs = {**results_base["contributions"], **results_base["overlays"]}
    diff_data = get_and_update_history(results_base["total_probability"], merged_contribs)
    
    # Scenario Selection
    st.sidebar.header("Scenario Analysis")
    scenario = st.sidebar.select_slider(
        "Market Assumption", 
        options=["Soft Landing", "Base Case", "Hard Landing"], 
        value="Base Case"
    )
    
    adjusted_data = raw_data.copy()
    if scenario == "Soft Landing":
        adjusted_data["gdp_qq"] = max(adjusted_data["gdp_qq"] + 0.3, 0.4)
        adjusted_data["unemployment"] = max(adjusted_data["unemployment"] - 0.3, 3.8)
        adjusted_data["cpi_headline"] = max(adjusted_data["cpi_headline"] - 0.5, 2.5)
        adjusted_data["cpi_trimmed"] = max(adjusted_data["cpi_trimmed"] - 0.5, 2.5)
        adjusted_data["cash_rate"] = max(adjusted_data["cash_rate"] - 0.5, 3.0)
    elif scenario == "Hard Landing":
        adjusted_data["gdp_qq"] -= 0.6
        adjusted_data["unemployment"] += 1.0
        adjusted_data["cpi_headline"] += 1.0
        adjusted_data["cpi_trimmed"] += 1.0
        adjusted_data["brent_crude"] += 20
        
    results = calculate_total_probability(adjusted_data)
    
    # Export Options in Sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Export Data")
    
    # 1. Gauge section
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_gauge = display_gauge(results["total_probability"])
        
        try:
            img_bytes = fig_gauge.to_image(format="png", engine="kaleido")
            st.sidebar.download_button("📸 Download Gauge (PNG)", data=img_bytes, file_name=f"recession_gauge_{datetime.now().strftime('%Y%m%d')}.png", mime="image/png")
        except Exception:
            pass # kaleido may not be installed
            
        csv_bytes = pd.DataFrame([adjusted_data]).to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("📊 Download Raw Data (CSV)", data=csv_bytes, file_name=f"recession_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        
        # Historical Diff Explainer
        if diff_data and scenario == "Base Case":
            diff_abs = diff_data["diff_total"]
            sign = "+" if diff_abs > 0 else ""
            if abs(diff_abs) >= 0.1:
                explainer = f"🧐 **What just moved it?** Recession probability {sign}{diff_abs:.1f} pts vs anchor, mainly due to: "
                reasons = []
                for k, v in diff_data["top_movers"]:
                    rsign = "+" if v > 0 else ""
                    reasons.append(f"**{k}** ({rsign}{v:.1f})")
                explainer += ", ".join(reasons) + "."
                st.info(explainer)

    with col2:
        st.markdown(f"### Latest Run")
        st.write(f"Updated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
        st.write(f"Base Probability: **{results['base_probability']:.1f}%**")
        st.write(f"Total Overlays: **{sum(results['overlays'].values()):.1f}%**")
        
        st.caption(generate_dynamic_advice(results["total_probability"], adjusted_data))
        
        # Tripwire Checklist
        st.markdown("### 🚨 Tripwires")
        gdp_val = adjusted_data["gdp_qq"]
        unemp_val = adjusted_data["unemployment"]
        cpi_val = adjusted_data["cpi_headline"]
        
        def trip_status(val, danger, safe, inverted=False):
            if inverted:
                if val <= danger: return "🔴"
                if val <= safe: return "🟠"
            else:
                if val >= danger: return "🔴"
                if val >= safe: return "🟠"
            return "🟢"
            
        st.markdown(f"{trip_status(gdp_val, -0.2, 0.2, True)} **GDP q/q** ({gdp_val:.1f}%)")
        st.markdown(f"{trip_status(unemp_val, 5.0, 4.5)} **Unemployment** ({unemp_val:.1f}%)")
        st.markdown(f"{trip_status(cpi_val, 4.5, 3.5)} **CPI** ({cpi_val:.1f}%)")
        
    st.divider()

    # 2. Metric Cards
    st.subheader("Current Core Indicators (Live Fetch)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("GDP (q/q)", f"{adjusted_data['gdp_qq']:.2f}%", f"{adjusted_data['gdp_qq'] - raw_data['gdp_qq']:.2f}%" if scenario != "Base Case" else None)
    m2.metric("Unemployment", f"{adjusted_data['unemployment']:.1f}%", f"{adjusted_data['unemployment'] - raw_data['unemployment']:.1f}%" if scenario != "Base Case" else None)
    m3.metric("Inflation (CPI)", f"{adjusted_data['cpi_headline']:.1f}%", f"{adjusted_data['cpi_headline'] - raw_data['cpi_headline']:.1f}%" if scenario != "Base Case" else None)
    m4.metric("Trimmed Mean", f"{adjusted_data['cpi_trimmed']:.1f}%", f"{adjusted_data['cpi_trimmed'] - raw_data['cpi_trimmed']:.1f}%" if scenario != "Base Case" else None)
    m5.metric("Cash Rate", f"{adjusted_data['cash_rate']:.2f}%", f"{adjusted_data['cash_rate'] - raw_data['cash_rate']:.2f}%" if scenario != "Base Case" else None)
    m6.metric("Real Wage Growth", f"{adjusted_data['real_wage_growth']:.2f}%", f"{adjusted_data['real_wage_growth'] - raw_data['real_wage_growth']:.2f}%" if scenario != "Base Case" else None)
    
    st.divider()

    # 3. Bar Chart for Contributions
    col_chart, col_sources = st.columns([1, 1])
    
    with col_chart:
        st.subheader("Score Breakdown")
        breakdown = {**results["contributions"], **results["overlays"]}
        df_breakdown = pd.DataFrame(list(breakdown.items()), columns=["Factor", "Contribution (%)"])
        fig_bar = go.Figure(go.Bar(
            x=df_breakdown["Contribution (%)"],
            y=df_breakdown["Factor"],
            orientation='h'
        ))
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_sources:
        st.subheader("Data Sources")
        source_df = pd.DataFrame({
            "Source": ["ABS", "RBA", "ASIC", "ASX", "Alpha Vantage", "Westpac", "Google Trends", "Gov Media", "Kalshi"],
            "Description": ["Economic Data", "Cash Rate", "Insolvencies", "Futures", "Brent Crude", "Sentiment", "Keyword Volume", "Signals", "US Recession Market"],
            "Status": ["Active"] * 9
        })
        st.dataframe(source_df, hide_index=True, use_container_width=True)

    st.divider()

    # 4. Prediction Market: Market vs Model
    st.subheader("🎯 Prediction Markets: Market vs Model")
    kalshi_odds = raw_data["kalshi_recession"]
    model_prob = results["total_probability"]
    kalshi_overlay_val = results["overlays"].get("US Prediction Market (Kalshi)", 0.0)
    
    pm_col1, pm_col2, pm_col3 = st.columns(3)
    
    with pm_col1:
        st.metric(
            label="🇺🇸 Kalshi: US Recession 2026",
            value=f"{kalshi_odds:.1f}%",
            help="Implied probability from the YES/NO prediction market on Kalshi. "
                 "Midpoint of bid/ask on KXRECSSNBER-26 contract."
        )
        us_signal = "🔴 High" if kalshi_odds >= 50 else ("🟠 Elevated" if kalshi_odds >= 25 else "🟢 Normal")
        st.caption(f"US recession signal: **{us_signal}**")
        st.caption("[View on Kalshi ↗](https://kalshi.com/markets/kxrecssnber-26)")

    with pm_col2:
        st.metric(
            label="🇦🇺 Model: AU Recession Probability",
            value=f"{model_prob:.1f}%",
            help="This model's current output — weighted indicators plus all overlays."
        )
        spread = model_prob - kalshi_odds
        spread_text = f"+{spread:.1f} pts above Kalshi" if spread >= 0 else f"{spread:.1f} pts below Kalshi"
        st.caption(f"Spread vs market: **{spread_text}**")

    with pm_col3:
        st.metric(
            label="📊 Kalshi → AU Overlay Applied",
            value=f"+{kalshi_overlay_val:.1f} pts",
            help="The portion of AU probability added by the Kalshi signal. "
                 "Only activates above 25% threshold. Capped at +15 pts."
        )
        if kalshi_odds < 25:
            st.caption("Below 25% threshold — no overlay currently active.")
        else:
            effective_pct = min(100, (kalshi_overlay_val / 15.0) * 100)
            st.caption(f"Cap utilisation: **{effective_pct:.0f}% of +15 pt max**")

    st.divider()
    with st.expander("📖 Methodology & Data Sources"):
        st.markdown("""
        **Australia Recession Probability Monitor Methodology**
        
        This model utilizes a continuous weighting mechanism rather than a binary threshold. 
        Each core indicator spans from a 'safe' to a 'danger' limit and accumulates its weighted percentage score.
        
        - **GDP (20%)**: Standard contraction limits.
        - **Unemployment (20%)**: RBA NAIRU overshoot.
        - **CPI & Trimmed (30%)**: Central bank mandate breach.
        - **Cash Rate (10%)**: Interest rate restrictiveness.
        - **Real Wage Growth (10%)**: Consumer squeeze.
        - **Insolvencies (10%)**: Corporate stress overlay.
        
        *Overlays*: Brent Crude shocks, Consumer Sentiment crashes, Media Panic signals, and US Prediction Market odds add discrete penalty multipliers to the base score.
        The **Kalshi overlay** is applied only when US recession odds exceed 25% (above normal expansion baseline), scaling at +0.5 AU pts per 1 US pt, capped at +15 pts.
        All signals are derived direct from official endpoints (ABS, RBA, ASIC) and priced daily using market futures.
        """)

if __name__ == "__main__":
    main()
