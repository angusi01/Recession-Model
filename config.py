import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", "demo")

# Weights for the base probability calculation (must sum to 1.0)
WEIGHTS = {
    "gdp_qq": 0.20,
    "unemployment": 0.20,
    "cpi_headline": 0.18,
    "cpi_trimmed": 0.12,
    "cash_rate": 0.10,
    "real_wage_growth": 0.10,
    "insolvency_rate": 0.10
}

# Thresholds to score indicators from 0 (safe) to 1 (danger)
# "safe": value at or beyond which score is 0
# "danger": value at or beyond which score is 1
# "lower_is_worse": True if dropping values means increasing danger (like GDP)
THRESHOLDS = {
    "gdp_qq": {"safe": 0.6, "danger": -0.2, "lower_is_worse": True},
    "unemployment": {"safe": 4.0, "danger": 5.5, "lower_is_worse": False},
    "cpi_headline": {"safe": 2.5, "danger": 5.0, "lower_is_worse": False},
    "cpi_trimmed": {"safe": 2.5, "danger": 4.5, "lower_is_worse": False},
    "cash_rate": {"safe": 2.5, "danger": 4.75, "lower_is_worse": False},
    "real_wage_growth": {"safe": 0.5, "danger": -2.0, "lower_is_worse": True},
    "insolvency_rate": {"safe": 0.32, "danger": 0.55, "lower_is_worse": False}
}

# Constants for Overlays
PRE_WAR_BRENT_BASELINE = 63.0

# Data Source URLs
URLS = {
    "abs_base": "https://api.data.abs.gov.au/data/",
    "rba_cash_rate": "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv",
    "rba_inflation_exp": "https://www.rba.gov.au/statistics/tables/csv/g3-data.csv",
    "asic_insolvency": "https://asic.gov.au/online-services/search-registries/insolvency-statistics/",
    "asx_rate_tracker": "https://www.asx.com.au/markets/trade-our-derivatives-market/futures-market/rba-rate-tracker",
    "westpac_mics": "https://www.westpac.com.au/about-westpac/media/media-releases/",
    "alphavantage": "https://www.alphavantage.co/query",
    "official_keywords": [
        "https://www.rba.gov.au/media-releases/",
        "https://treasury.gov.au/media-releases",
        "https://www.pm.gov.au/media",
        "https://www.abc.net.au/news/business/economy"
    ]
}

# Official Keyword Targets
KEYWORDS = {
    "rba": ["recession", "contraction", "significant deterioration"],
    "treasury": ["emergency", "intervention", "stimulus"],
    "pm": ["economic crisis", "emergency cabinet"],
    "abc": ["recession", "economic crisis"]
}

# Estimated total registered companies for insolvency rate calculation
TOTAL_REGISTERED_COMPANIES = 3_200_000

# Cache Time-To-Live (seconds)
TTL = {
    "daily": 86400,
    "twice_daily": 43200,
    "hourly": 3600
}
