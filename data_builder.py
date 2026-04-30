"""
data_builder.py — Historical data fetching and feature engineering.

All features are lagged to simulate real-time availability at each point in time:
  - GDP (quarterly)       → lagged 1 quarter (ABS releases ~90 days after reference quarter)
  - CPI trimmed (quarterly)→ lagged 1 quarter
  - Unemployment (monthly) → lagged 1 month (ABS releases ~6 weeks after reference month)
  - Market data (daily)    → no lag required (yield curve, iron ore, ASX200)

Recession definition: 2 consecutive quarters of negative GDP growth (q/q %).
Forward targets:
  y_3m[t] = 1 if recession occurs in any of the 3 months [t+1 … t+3]
  y_6m[t] = 1 if recession occurs in any of the 6 months [t+1 … t+6]
"""

import logging
import warnings

import numpy as np
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
ABS_BASE = "https://data.api.abs.gov.au/rest/data/"
RBA_F1_URL = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
RBA_F2_URL = "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"
RBA_I2_URL = "https://www.rba.gov.au/statistics/tables/csv/i2-data.csv"

# ABS series IDs (new SDMX API format)
GDP_QQ_SERIES = "ANA_AGG/M2.GPM.20.AUS.Q"          # GDP % change q/q, seasonally adjusted
UNEMPLOYMENT_SERIES = "LF/3.1.15.1599.20.M"        # Unemployment rate, monthly, persons, SA
CPI_TRIMMED_SERIES = "CPI/3.999902.20.Q"            # Trimmed mean CPI, YoY %, seasonally adjusted

# RBA column names
RBA_10Y_COL = "FCMYGBAG10D"   # F2: 10-year CGS yield
RBA_2Y_COL = "FCMYGBAG2D"     # F2: 2-year CGS yield (fallback for 3M)
RBA_3M_COL = "FIRMMBAB90D"    # F1: 90-day bank bill rate (3M proxy)
RBA_IRON_ORE_COL = "Iron ore"  # I2: iron ore USD spot

ROLLING_ZSCORE_WINDOW = 60     # 5-year rolling window in months
HISTORY_START = "1989-01-01"   # how far back to pull market data


# ── ABS SDMX helpers ─────────────────────────────────────────────────────────

def _parse_abs_json(data: dict) -> pd.Series:
    """
    Parse ABS SDMX JSON response (new flat observations format) into a pd.Series with PeriodIndex.

    The new ABS Data API (post-November 2024) returns observations in a flat dict at
    dataSets[0]["observations"] when dimensionAtObservation=AllDimensions is used.
    Keys are colon-separated dimension indices (e.g. "0:0:0:0:N") where N is the
    time-period index within the TIME_PERIOD dimension's values list.

    Handles both monthly (e.g. '2023-09') and quarterly (e.g. '2023-Q3') dates.
    """
    dims = data["data"]["structure"]["dimensions"]["observation"]
    time_dim = next((d for d in dims if d["id"] == "TIME_PERIOD"), dims[-1])
    time_pos = time_dim["keyPosition"]
    time_periods = [v["id"] for v in time_dim["values"]]

    observations = data["data"]["dataSets"][0]["observations"]

    records = {}
    for key, val_list in observations.items():
        indices = list(map(int, key.split(":")))
        time_idx = indices[time_pos]
        if time_idx < len(time_periods) and val_list and val_list[0] is not None:
            records[time_periods[time_idx]] = float(val_list[0])

    if not records:
        raise ValueError("No observations found in ABS SDMX response")

    dates, values = [], []
    for date_str in sorted(records):
        try:
            if "-Q" in date_str:
                period = pd.Period(date_str.replace("-Q", "Q"), freq="Q")
            else:
                period = pd.Period(date_str[:7], freq="M")
            dates.append(period)
            values.append(records[date_str])
        except Exception:
            continue

    return pd.Series(values, index=pd.PeriodIndex(dates))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_abs_full_series(series_id: str, start_period: str = "1989-Q1") -> pd.Series:
    """Fetch full historical ABS SDMX series as a pd.Series with PeriodIndex.

    Uses the new ABS Data API format (post-November 2024):
    - Accept: application/json
    - dimensionAtObservation=AllDimensions for consistent flat observations format
    """
    url = f"{ABS_BASE}{series_id}?dimensionAtObservation=AllDimensions&startPeriod={start_period}"
    try:
        resp = requests.get(
            url, timeout=30,
            headers={"Accept": "application/json"}
        )
        resp.raise_for_status()
        return _parse_abs_json(resp.json())
    except Exception as e:
        logger.warning(f"fetch_abs_full_series failed for {series_id}: {e}")
        return pd.Series(dtype=float)


# ── RBA CSV helpers ───────────────────────────────────────────────────────────

def _parse_rba_csv(url: str, cols: list[str]) -> pd.DataFrame:
    """
    Parse an RBA CSV table.
    Returns a DataFrame with a DatetimeIndex and requested columns (where present).
    """
    df = pd.read_csv(url, skiprows=10, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    available = [c for c in cols if c in df.columns]
    if not available:
        raise ValueError(f"None of {cols} found in {url}")
    result = df[available].apply(pd.to_numeric, errors="coerce")
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_rba_yield_history() -> pd.DataFrame:
    """
    Fetch F2 (10Y, 2Y CGS) and F1 (90-day bank bill) to build yield curve history.
    Returns a daily-indexed DataFrame with columns: yield_10y, yield_3m.
    """
    result = pd.DataFrame()

    # F2 — 10Y CGS
    try:
        f2 = _parse_rba_csv(RBA_F2_URL, [RBA_10Y_COL, RBA_2Y_COL])
        result["yield_10y"] = f2[RBA_10Y_COL] if RBA_10Y_COL in f2.columns else np.nan
        result["yield_2y"] = f2[RBA_2Y_COL] if RBA_2Y_COL in f2.columns else np.nan
    except Exception as e:
        logger.warning(f"Failed to fetch RBA F2 yield data: {e}")

    # F1 — 90-day bank bills
    try:
        f1 = _parse_rba_csv(RBA_F1_URL, [RBA_3M_COL])
        result["yield_3m"] = f1[RBA_3M_COL]
    except Exception as e:
        logger.warning(f"Failed to fetch RBA F1 3M rate: {e}")
        # Fall back to 2Y as a proxy if 3M unavailable
        if "yield_2y" in result.columns:
            result["yield_3m"] = result["yield_2y"]

    return result


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_rba_iron_ore_history() -> pd.Series:
    """Fetch iron ore USD price history from RBA I2 commodity table."""
    try:
        df = pd.read_csv(RBA_I2_URL, skiprows=10, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()].sort_index()
        col = next(
            (c for c in df.columns if "iron ore" in c.lower()
             or c == RBA_IRON_ORE_COL
             or "pciron" in c.lower()
             or c.lower().startswith("iron")),
            None,
        )
        if col is None:
            raise ValueError("Iron ore column not found in RBA I2")
        return pd.to_numeric(df[col], errors="coerce").dropna().rename("iron_ore")
    except Exception as e:
        logger.warning(f"Failed to fetch RBA iron ore history: {e}")
        return pd.Series(dtype=float, name="iron_ore")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_asx200_history() -> pd.Series:
    """
    Fetch ASX 200 (^AXJO) daily close history via yfinance.
    Returns a DatetimeIndex Series.
    Falls back to an empty Series if yfinance is unavailable.
    """
    try:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ticker = yf.Ticker("^AXJO")
            hist = ticker.history(start=HISTORY_START, auto_adjust=True)
        if hist.empty:
            raise ValueError("Empty yfinance response for ^AXJO")
        return hist["Close"].rename("asx200")
    except Exception as e:
        logger.warning(f"Failed to fetch ASX200 history: {e}")
        return pd.Series(dtype=float, name="asx200")


# ── Recession labeling ────────────────────────────────────────────────────────

def label_recessions_quarterly(gdp_qq: pd.Series) -> pd.Series:
    """
    Label quarters as in-recession based on 2+ consecutive negative GDP growth quarters.

    When two consecutive quarters both show negative GDP growth, both are marked as
    recession quarters (1). This means quarter i is labelled 1 if gdp[i] < 0 and
    gdp[i-1] < 0, and additionally quarter i-1 is retroactively labelled 1.

    Returns a binary pd.Series with the same quarterly PeriodIndex as gdp_qq.
    """
    gdp = gdp_qq.dropna().sort_index()
    neg = (gdp < 0).astype(int)
    recession = pd.Series(0, index=gdp.index)

    for i in range(1, len(gdp)):
        if neg.iloc[i] == 1 and neg.iloc[i - 1] == 1:
            recession.iloc[i] = 1
            recession.iloc[i - 1] = 1

    return recession


def resample_quarterly_to_monthly(q_series: pd.Series) -> pd.Series:
    """
    Convert a quarterly PeriodIndex series to monthly.

    Each of the three months within a quarter is assigned the quarter's value,
    so the result correctly reflects when that quarterly reading applies.
    (The previous implementation placed values only at the quarter-end month
    and forward-filled, causing the first two months of each quarter to carry
    the preceding quarter's value — contradicting the quarterly labeling intent.)
    """
    if q_series.empty:
        return pd.Series(dtype=float)

    records = {}
    for q_period, val in q_series.items():
        # Assign the quarterly value to all three months within the quarter
        for month_offset in range(3):
            start_month = q_period.asfreq("M", how="S")
            m = start_month + month_offset
            records[m] = val

    if not records:
        return pd.Series(dtype=float)

    monthly_index = pd.PeriodIndex(sorted(records.keys()), freq="M")
    return pd.Series([records[m] for m in monthly_index], index=monthly_index)


def create_forward_targets(recession_monthly: pd.Series,
                           horizons: tuple = (3, 6)) -> dict:
    """
    Create binary forward targets for each horizon H:
      y_Hm[t] = 1 if ANY month in [t+1 … t+H] is a recession month.

    Returns a dict mapping 'y_3m' and 'y_6m' to pd.Series (0/1, monthly).
    """
    targets = {}
    n = len(recession_monthly)
    for h in horizons:
        y = pd.Series(0, index=recession_monthly.index, name=f"y_{h}m", dtype=int)
        for i in range(n - h):
            if recession_monthly.iloc[i + 1: i + h + 1].any():
                y.iloc[i] = 1
        targets[f"y_{h}m"] = y
    return targets


# ── Rolling z-score ───────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = ROLLING_ZSCORE_WINDOW) -> pd.Series:
    """Compute rolling z-score with a given window (in months)."""
    mu = series.rolling(window, min_periods=24).mean()
    sigma = series.rolling(window, min_periods=24).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (series - mu) / sigma
    return z


# ── Feature matrix ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def build_feature_matrix() -> dict:
    """
    Assemble a monthly feature matrix with NO lookahead bias.

    Lags applied:
      - gdp_qq:       +3 months (1 quarter) — data released ~90 days after quarter end
      - cpi_trimmed:  +3 months (1 quarter) — same as GDP
      - unemployment: +1 month              — data released ~6 weeks after reference month

    Returns a dict with keys:
      'features'  : pd.DataFrame of feature columns (monthly PeriodIndex)
      'y_3m'      : pd.Series  binary target (recession in next 3 months)
      'y_6m'      : pd.Series  binary target (recession in next 6 months)
      'recession' : pd.Series  monthly binary recession label (for display)
    """
    # ── 1. Fetch raw series ──────────────────────────────────────────────────
    gdp_qq_q = fetch_abs_full_series(GDP_QQ_SERIES)                        # quarterly
    unemp_m = fetch_abs_full_series(UNEMPLOYMENT_SERIES, "1989-01")        # monthly
    cpi_q = fetch_abs_full_series(CPI_TRIMMED_SERIES)                      # quarterly

    rba_yields = fetch_rba_yield_history()   # daily DataFrame
    iron_ore_d = fetch_rba_iron_ore_history()  # daily/monthly Series
    asx200_d = fetch_asx200_history()          # daily Series

    # ── 2. Recession labels (use UNLAGGED GDP for ground truth) ──────────────
    if gdp_qq_q.empty:
        logger.warning("GDP series empty — recession labels will be all-zero")
        gdp_qq_q_clean = pd.Series(dtype=float)
        recession_q = pd.Series(dtype=float)
    else:
        gdp_qq_q_clean = gdp_qq_q.sort_index()
        recession_q = label_recessions_quarterly(gdp_qq_q_clean)

    recession_m = (
        resample_quarterly_to_monthly(recession_q)
        if not recession_q.empty
        else pd.Series(dtype=float)
    )

    # ── 3. Resample all series to monthly end-of-period ──────────────────────
    def _to_monthly_period(daily: pd.Series, agg: str = "last") -> pd.Series:
        """Resample a daily DatetimeIndex series to monthly PeriodIndex."""
        if daily.empty:
            return pd.Series(dtype=float)
        monthly = daily.resample("ME").agg(agg).dropna()
        monthly.index = monthly.index.to_period("M")
        return monthly

    # Yield series (daily -> monthly mean)
    if not rba_yields.empty:
        has_10y = "yield_10y" in rba_yields.columns
        short_col = "yield_3m" if "yield_3m" in rba_yields.columns else (
            "yield_2y" if "yield_2y" in rba_yields.columns else None
        )
        if has_10y and short_col is not None:
            ten = rba_yields["yield_10y"]
            short = rba_yields[short_col]
            yc_daily = (ten - short).rename("yield_curve_slope").dropna()
            yield_curve_m = _to_monthly_period(yc_daily, agg="mean")
        else:
            logger.warning("Yield curve requires both yield_10y and a short rate — skipping yield features")
            yield_curve_m = pd.Series(dtype=float)
        yc_10y_m = _to_monthly_period(
            rba_yields["yield_10y"].dropna() if has_10y else pd.Series(dtype=float), agg="mean"
        )
        yc_3m_m = _to_monthly_period(
            rba_yields[short_col].dropna() if short_col else pd.Series(dtype=float), agg="mean"
        )
    else:
        yield_curve_m = pd.Series(dtype=float)
        yc_10y_m = pd.Series(dtype=float)
        yc_3m_m = pd.Series(dtype=float)

    iron_ore_m = _to_monthly_period(iron_ore_d, agg="last") if not iron_ore_d.empty else pd.Series(dtype=float)
    asx200_m = _to_monthly_period(asx200_d, agg="last") if not asx200_d.empty else pd.Series(dtype=float)

    # ── 4. Resample quarterly ABS to monthly (forward-fill) ──────────────────
    gdp_qq_m = resample_quarterly_to_monthly(gdp_qq_q_clean) if not gdp_qq_q_clean.empty else pd.Series(dtype=float)
    cpi_m = resample_quarterly_to_monthly(cpi_q.sort_index()) if not cpi_q.empty else pd.Series(dtype=float)

    # Monthly unemployment
    if not unemp_m.empty:
        unemp_m_clean = unemp_m.sort_index()
    else:
        unemp_m_clean = pd.Series(dtype=float)

    # ── 5. Build monthly DataFrame (align on monthly PeriodIndex) ────────────
    all_monthly = [
        gdp_qq_m.rename("gdp_qq_raw"),
        cpi_m.rename("cpi_trimmed_raw"),
        unemp_m_clean.rename("unemployment_raw"),
        yield_curve_m.rename("yield_curve_slope"),
        iron_ore_m.rename("iron_ore"),
        asx200_m.rename("asx200"),
    ]
    # Only use series that have data
    valid_series = [s for s in all_monthly if not s.empty]
    if not valid_series:
        logger.error("No valid data series found — returning empty feature matrix")
        return {"features": pd.DataFrame(), "y_3m": pd.Series(dtype=int),
                "y_6m": pd.Series(dtype=int), "recession": pd.Series(dtype=int)}

    df = pd.concat(valid_series, axis=1)

    # ── 6. Apply publication lags ─────────────────────────────────────────────
    # gdp_qq: lag 3 months (released ~1 quarter late)
    if "gdp_qq_raw" in df.columns:
        df["gdp_qq"] = df["gdp_qq_raw"].shift(3)
    # cpi_trimmed: lag 3 months
    if "cpi_trimmed_raw" in df.columns:
        df["cpi_trimmed"] = df["cpi_trimmed_raw"].shift(3)
    # unemployment: lag 1 month
    if "unemployment_raw" in df.columns:
        df["unemployment"] = df["unemployment_raw"].shift(1)

    # Drop the raw (unlagged) columns now that lagged versions have been created.
    # Keeping them would risk accidental use and wastes memory.
    df.drop(columns=["gdp_qq_raw", "cpi_trimmed_raw", "unemployment_raw"],
            errors="ignore", inplace=True)

    # ── 7. Compute derived features ──────────────────────────────────────────
    if "yield_curve_slope" in df.columns:
        df["yield_curve_3m_delta"] = df["yield_curve_slope"].diff(3)
        df["yield_curve_6m_delta"] = df["yield_curve_slope"].diff(6)
        df["yield_curve_zscore"] = rolling_zscore(df["yield_curve_slope"])

    if "unemployment" in df.columns:
        df["unemployment_3m_change"] = df["unemployment"].diff(3)
        df["unemployment_12m_change"] = df["unemployment"].diff(12)
        df["unemployment_change_zscore"] = rolling_zscore(df["unemployment_3m_change"])

    if "gdp_qq" in df.columns:
        # Annualised 4-quarter GDP momentum (sum of 4 lagged q/q readings)
        df["gdp_4q_sum"] = df["gdp_qq"].rolling(4, min_periods=2).sum()

    if "iron_ore" in df.columns:
        df["iron_ore_3m_pct"] = df["iron_ore"].pct_change(3) * 100
        df["iron_ore_zscore"] = rolling_zscore(df["iron_ore"])

    if "asx200" in df.columns:
        # Rolling 12-month drawdown (0 = at peak, -0.30 = 30% below peak)
        rolling_peak = df["asx200"].rolling(12, min_periods=6).max()
        df["asx200_drawdown"] = (df["asx200"] / rolling_peak) - 1.0
        df["asx200_3m_return"] = df["asx200"].pct_change(3) * 100

    # ── 8. Align recession labels and targets ───────────────────────────────
    if not recession_m.empty:
        recession_aligned = recession_m.reindex(df.index).fillna(0).astype(int)
    else:
        recession_aligned = pd.Series(0, index=df.index)

    targets = create_forward_targets(recession_aligned)

    # ── 9. Final feature columns ─────────────────────────────────────────────
    feature_cols = [
        "yield_curve_slope", "yield_curve_3m_delta", "yield_curve_6m_delta", "yield_curve_zscore",
        "unemployment", "unemployment_3m_change", "unemployment_12m_change", "unemployment_change_zscore",
        "gdp_qq", "gdp_4q_sum",
        "cpi_trimmed",
        "iron_ore", "iron_ore_3m_pct", "iron_ore_zscore",
        "asx200_drawdown", "asx200_3m_return",
    ]
    available_features = [c for c in feature_cols if c in df.columns]
    features_df = df[available_features].copy()

    # Drop rows where more than half of features are NaN (insufficient data)
    min_valid = max(1, len(available_features) // 2)
    features_df = features_df[features_df.notna().sum(axis=1) >= min_valid]

    # Align all outputs to the same index
    common_idx = features_df.index
    y_3m = targets["y_3m"].reindex(common_idx).fillna(0).astype(int)
    y_6m = targets["y_6m"].reindex(common_idx).fillna(0).astype(int)
    recession_out = recession_aligned.reindex(common_idx).fillna(0).astype(int)

    return {
        "features": features_df,
        "y_3m": y_3m,
        "y_6m": y_6m,
        "recession": recession_out,
    }


# ── Current-state feature vector ─────────────────────────────────────────────

def get_current_feature_row(feature_matrix: pd.DataFrame) -> pd.Series | None:
    """
    Return the most recent row of the feature matrix (last non-NaN entry).
    This represents the best available real-time feature vector.
    """
    if feature_matrix.empty:
        return None
    # Take the last row that has at least one non-NaN value
    valid_rows = feature_matrix.dropna(how="all")
    if valid_rows.empty:
        return None
    return valid_rows.iloc[-1]
