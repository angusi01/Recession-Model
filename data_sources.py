import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
import time
from datetime import datetime
import json
import logging

from config import URLS, KEYWORDS, TOTAL_REGISTERED_COMPANIES, ALPHAVANTAGE_KEY, TTL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback cache in case of API/scrape failure
# This ensures the app never crashes
FALLBACKS = {
    "yield_curve": 0.1,
    "iron_ore": 90.0,
    "gdp_qq": 0.2,
    "unemployment": 4.1,
    "cpi_headline": 4.1,
    "cpi_trimmed": 3.9,
    "cash_rate": 4.35,
    "real_wage_growth": 0.1,
    "insolvency_rate": 0.35,
    "anz_job_ads": -5.0,
    "brent_crude": 80.0,
    "asx_cash_rate": 4.35,
    "westpac_sentiment": 82.0,
    "google_trends": 50.0,
    "keyword_hits": 5,
    "kalshi_recession": 25.0
}

def log_failure(source_name, error=""):
    logger.warning(f"Failed to fetch {source_name}: {error}. Using fallback.")

@st.cache_data(ttl=TTL["daily"], show_spinner=False)
def fetch_abs_data(series_id, fallback_key):
    """Fetch data from the ABS API."""
    url = f"{URLS['abs_base']}{series_id}?format=jsondata"
    try:
        response = requests.get(url, timeout=10, headers={"Accept": "application/vnd.sdmx.data+json"})
        response.raise_for_status()
        data = response.json()
        
        # Parse standard ABS SDMX JSON structure based on prompt
        observations = data["data"]["dataSets"][0]["series"]["0:0:0:0:0"]["observations"]
        # Find the max key (latest observation)
        latest_key = max(observations.keys(), key=lambda x: int(x))
        latest_value = observations[latest_key][0]
        return float(latest_value)
    except Exception as e:
        log_failure(f"ABS ({series_id})", repr(e))
        return FALLBACKS[fallback_key]

@st.cache_data(ttl=TTL["daily"], show_spinner=False)
def fetch_rba_csv(url, target_col, fallback_key):
    """Fetch and parse RBA CSV feeds."""
    try:
        # RBA CSVs usually have 10 rows of metadata
        df = pd.read_csv(url, skiprows=10)
        # Drop rows where the target column is NaN, then get the last value
        df = df.dropna(subset=[target_col])
        if not df.empty:
            latest_val = df[target_col].iloc[-1]
            return float(latest_val)
        return FALLBACKS[fallback_key]
    except Exception as e:
        log_failure(f"RBA CSV ({target_col})", repr(e))
        return FALLBACKS[fallback_key]

@st.cache_data(ttl=TTL["daily"], show_spinner=False)
def fetch_yield_curve_spread(url):
    """Fetch 10Y minus 2Y CGS spread from RBA F2 table.

    Fetches both legs from the same CSV in a single pass.  If either leg is
    missing or the CSV cannot be parsed, the spread falls back to
    FALLBACKS['yield_curve'] so that a partial failure cannot produce an
    invalid spread (e.g. a yield level minus a spread fallback).
    """
    try:
        df = pd.read_csv(url, skiprows=10)
        # Sort by the date column (first column) so iloc[-1] always returns the
        # most recent observation regardless of how the CSV is ordered.
        df = df.sort_values(by=df.columns[0], ascending=True).reset_index(drop=True)
        df_spread = df.dropna(subset=["FCMYGBAG10D", "FCMYGBAG2D"])
        if df_spread.empty:
            log_failure("RBA Yield Curve (spread)", "no rows contain both 10Y and 2Y legs after dropna")
            return FALLBACKS["yield_curve"]
        latest_row = df_spread.iloc[-1]
        cgs_10y = float(latest_row["FCMYGBAG10D"])
        cgs_2y = float(latest_row["FCMYGBAG2D"])
        return cgs_10y - cgs_2y
    except Exception as e:
        log_failure("RBA Yield Curve (spread)", repr(e))
        return FALLBACKS["yield_curve"]

@st.cache_data(ttl=TTL["daily"], show_spinner=False)
def fetch_asic_insolvency():
    """Scrape ASIC insolvency data to calculate rate."""
    url = URLS["asic_insolvency"]
    try:
        # NOTE: ASIC site may require deeper scraping or headless browser.
        # Here we perform a basic text search as a best-effort approach.
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Attempt to find table data for "External administrations"
        # Since the exact DOM might change, we look for text
        tds = soup.find_all("td")
        for i, td in enumerate(tds):
            if "External administrations" in td.get_text():
                # Usually the value is in the next column
                val_text = tds[i+1].get_text().replace(",", "")
                if val_text.isdigit():
                    count = int(val_text)
                    return (count / TOTAL_REGISTERED_COMPANIES) * 100
        
        raise ValueError("Could not find table data")
    except Exception as e:
        log_failure("ASIC Insolvency", repr(e))
        return FALLBACKS["insolvency_rate"]

@st.cache_data(ttl=TTL["hourly"], show_spinner=False)
def fetch_brent_crude():
    """Fetch Brent crude (WTI used as proxy in AlphaVantage free tier)."""
    try:
        if not ALPHAVANTAGE_KEY or ALPHAVANTAGE_KEY == "demo":
            raise ValueError("No valid AlphaVantage key")
            
        url = f"{URLS['alphavantage']}?function=WTI&interval=daily&apikey={ALPHAVANTAGE_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            val = data["data"][0]["value"]
            if val == ".":
                # Current day might not be published yet, grab previous day
                val = data["data"][1]["value"]
            return float(val)
        else:
            raise ValueError(f"Unexpected JSON structure: {str(data)[:100]}")
    except Exception as e:
        log_failure("Brent Crude (Alpha Vantage)", repr(e))
        return FALLBACKS["brent_crude"]

@st.cache_data(ttl=TTL["daily"], show_spinner=False)
def fetch_asx_futures():
    """Scrape implied cash rate from ASX 6 months forward."""
    try:
        url = URLS["asx_rate_tracker"]
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find target rate text
        # If scraping fails due to dynamic JS, we fall back
        for row in soup.find_all('tr'):
            text = row.get_text()
            if 'Implied yield' in text or 'Market Expectation' in text:
                # Naive extraction
                 parts = text.split('%')
                 for p in parts:
                     nums = [s for s in p.split() if s.replace('.','',1).isdigit()]
                     if nums:
                         return float(nums[-1])
        raise ValueError("Could not parse ASX rate")
    except Exception as e:
        log_failure("ASX Futures", repr(e))
        return FALLBACKS["asx_cash_rate"]

@st.cache_data(ttl=TTL["daily"], show_spinner=False)
def fetch_westpac_sentiment():
    """Scrape Westpac sentiment index."""
    try:
        url = URLS["westpac_mics"]
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find latest media release linking to sentiment
        text = soup.get_text()
        # Look for a number near "Consumer Sentiment Index"
        # Highly prone to failure, usually relies on text heuristic
        idx = text.find("Consumer Sentiment")
        if idx != -1:
             snippet = text[idx:idx+100]
             import re
             match = re.search(r'\b\d{2}\.\d\b', snippet)
             if match:
                 return float(match.group())
        raise ValueError("Could not find sentiment index in text")
    except Exception as e:
        log_failure("Westpac Sentiment", repr(e))
        return FALLBACKS["westpac_sentiment"]

def fetch_google_trends():
    """Pull Google Trends data for economic stress keywords. Returns (value, is_error)."""
    import os
    CACHE_FILE = ".trends_cache.json"
    now = time.time()
    
    cache_data = None
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache_data = json.load(f)
        except Exception:
            pass

    # Return valid cache if within TTL
    if cache_data and (now - cache_data["timestamp"] < TTL["twice_daily"]):
        return cache_data["value"], False
        
    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
        kw_list = ["recession australia", "job losses australia", "mortgage stress", "fuel prices australia"]
        pytrends.build_payload(kw_list, cat=0, timeframe='today 3-m', geo='AU')
        data = pytrends.interest_over_time()
        
        if not data.empty:
            # Average the past 90 days for all terms
            data = data.drop(labels=['isPartial'], axis='columns', errors='ignore')
            mean_score = float(data.mean().mean()) # average across time, then average across terms
            
            # Save new valid value to cache
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump({"value": mean_score, "timestamp": now}, f)
            except Exception:
                pass
                
            return mean_score, False
        raise ValueError("Empty trends data")
    except Exception as e:
        log_failure("Google Trends", repr(e))
        
    # Fetch failed, fallback to expired cache if available
    if cache_data:
        return cache_data["value"], True
    return FALLBACKS["google_trends"], True

@st.cache_data(ttl=TTL["hourly"], show_spinner=False)
def fetch_kalshi_recession_odds():
    """Fetch US recession 2026 implied probability from Kalshi's public JSON API.
    
    Uses the midpoint of bid/ask in dollar terms so prices are unauthenticated
    (public endpoint). Returns a percentage 0-100.
    """
    try:
        url = URLS["kalshi_recession"]
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        market = resp.json().get("market", {})
        
        yes_bid = market.get("yes_bid_dollars")
        yes_ask = market.get("yes_ask_dollars")
        
        if yes_bid is None or yes_ask is None:
            raise ValueError(f"Missing price fields. Market keys: {list(market.keys())[:10]}")
        
        midpoint = (float(yes_bid) + float(yes_ask)) / 2.0
        return round(midpoint * 100, 2)  # Convert dollars (0-1) to percentage
    except Exception as e:
        log_failure("Kalshi US Recession Odds", repr(e))
        return FALLBACKS["kalshi_recession"]


@st.cache_data(ttl=TTL["hourly"], show_spinner=False)
def fetch_official_keywords():
    """Scrape official media sites for crisis keywords and return count."""
    hit_count = 0
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for source, url in zip(KEYWORDS.keys(), URLS["official_keywords"]):
        try:
            resp = requests.get(url, timeout=10, headers=headers)
            if resp.status_code == 200:
                text = resp.text.lower()
                for kw in KEYWORDS[source]:
                    hit_count += text.count(kw)
        except Exception as e:
            log_failure(f"Official Keyword ({source})", repr(e))
            
    # If network totally down, return fallback
    if hit_count == 0:
        # It could truly be 0, but as a fallback system we just report 0 hits if empty
        pass
    
    return hit_count
