from config import WEIGHTS, THRESHOLDS, PRE_WAR_BRENT_BASELINE, FORECAST_TARGET, FORECAST_ANNOUNCE

def calculate_indicator_score(value, key):
    """
    Score indicator from 0 (safe) to 1 (danger).
    Linear mapping between safe and danger thresholds.
    """
    safe = THRESHOLDS[key]["safe"]
    danger = THRESHOLDS[key]["danger"]
    lower_is_worse = THRESHOLDS[key]["lower_is_worse"]
    
    if lower_is_worse:
        if value >= safe:
            return 0.0
        elif value <= danger:
            return 1.0
        else:
            return (safe - value) / (safe - danger)
    else:
        if value <= safe:
            return 0.0
        elif value >= danger:
            return 1.0
        else:
            return (value - safe) / (danger - safe)

def calculate_base_probability(raw_data):
    """Calculate the base probability using weighted indicators."""
    base_score = 0.0
    contributions = {}
    
    for key, weight in WEIGHTS.items():
        score = calculate_indicator_score(raw_data[key], key)
        weighted_score = score * weight
        base_score += weighted_score
        contributions[key] = weighted_score * 100 # percentage contribution
        
    return base_score * 100, contributions

def calculate_overlays(raw_data):
    """Calculate the dynamic overlays to add to the base probability."""
    overlays = {}
    
    # Geo overlay: brent crude proxy (oil signal lives here only — no CPI passthrough)
    current_price = raw_data["brent_crude"]
    geo_overlay = max(0, ((current_price - PRE_WAR_BRENT_BASELINE) / PRE_WAR_BRENT_BASELINE) * 15)
    overlays["Geopolitical (Brent Crude)"] = min(20.0, geo_overlay)
    
    # Consumer sentiment overlay
    sentiment = raw_data["westpac_sentiment"]
    overlays["Consumer Sentiment"] = max(0, ((100 - sentiment) / 100) * 5)
    
    # Google Trends overlay
    trends = raw_data["google_trends"]
    overlays["Google Trends Signal"] = (trends / 100) * 4
    
    # Keyword overlay
    keywords = raw_data["keyword_hits"]
    overlays["Official Media Keywords"] = min(5.0, keywords * 0.3)
    
    # Prediction market overlay: Kalshi US recession 2026 odds
    # Threshold: 25% (above normal expansion pricing of 15-20%)
    # Scaling: +0.5 AU pts per 1 US pt above threshold
    # Cap: +12 pts max (prevents runaway if US odds jump to extreme values)
    kalshi_odds = raw_data.get("kalshi_recession", 25.0)
    kalshi_overlay = max(0.0, (kalshi_odds - 25.0) * 0.5)
    overlays["US Prediction Market (Kalshi)"] = min(12.0, kalshi_overlay)
    
    return overlays

def calculate_total_probability(raw_data):
    """
    Master function:
    1. Calculates base probability directly from raw inputs.
    2. Calculates overlays.
    3. Sums and caps at 99%.
    """
    base_prob, contributions = calculate_base_probability(raw_data)
    overlays = calculate_overlays(raw_data)
    
    total_overlay = sum(overlays.values())
    total_prob = min(base_prob + total_overlay, 99.0)
    
    return {
        "total_probability": total_prob,
        "base_probability": base_prob,
        "contributions": contributions,
        "overlays": overlays,
        "forecast_target": FORECAST_TARGET,
        "forecast_announce": FORECAST_ANNOUNCE,
    }
