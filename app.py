import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

from config import THRESHOLDS, WEIGHTS, URLS
from data_sources import (
    fetch_abs_data, fetch_rba_csv, fetch_asic_insolvency,
    fetch_brent_crude, fetch_asx_futures, fetch_westpac_sentiment,
    fetch_google_trends, fetch_official_keywords
)
from model import calculate_total_probability

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
            "google_trends": fetch_google_trends(),
            "keyword_hits": fetch_official_keywords()
        }

    # Model Calculation
    results = calculate_total_probability(raw_data)
    
    # 1. Gauge section
    col1, col2 = st.columns([2, 1])
    with col1:
        display_gauge(results["total_probability"])
    with col2:
        st.markdown(f"### Latest Run")
        st.write(f"Updated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
        st.write(f"Base Probability: **{results['base_probability']:.1f}%**")
        st.write(f"Total Overlays: **{sum(results['overlays'].values()):.1f}%**")
        
        st.markdown(generate_dynamic_advice(results["total_probability"], raw_data))
        
    st.divider()

    # 2. Metric Cards
    st.subheader("Current Core Indicators (Live Fetch)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("GDP (q/q)", f"{raw_data['gdp_qq']:.2f}%")
    m2.metric("Unemployment", f"{raw_data['unemployment']:.1f}%")
    m3.metric("Inflation (CPI)", f"{raw_data['cpi_headline']:.1f}%")
    m4.metric("Trimmed Mean", f"{raw_data['cpi_trimmed']:.1f}%")
    m5.metric("Cash Rate", f"{raw_data['cash_rate']:.2f}%")
    m6.metric("Real Wage Growth", f"{raw_data['real_wage_growth']:.2f}%")
    
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
            "Source": ["ABS", "RBA", "ASIC", "ASX", "Alpha Vantage", "Westpac", "Google Trends", "Gov Media"],
            "Description": ["Economic Data", "Cash Rate", "Insolvencies", "Futures", "Brent Crude", "Sentiment", "Keyword Volume", "Signals"],
            "Status": ["Active"] * 8
        })
        st.dataframe(source_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
