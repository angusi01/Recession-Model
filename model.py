from config import WEIGHTS, THRESHOLDS, PRE_WAR_BRENT_BASELINE

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

def apply_forward_pricing(raw_data):
    """
    Apply forward-looking adjustments to base economic indicators.
    Returns structurally identical dictionary with adjusted values.
    """
    forward = raw_data.copy()
    
    # GDP: if current < 0.4% q/q, apply -0.3% forward adjustment
    if forward["gdp_qq"] < 0.4:
        forward["gdp_qq"] -= 0.3
        
    # Unemployment: 'Rising trend' heuristic. 
    # If above 4.1% (historical norm), we assume rising pressure 
    if forward["unemployment"] > 4.1:
        forward["unemployment"] += 0.4
        
    # CPI passthrough
    brent_deviation_pct = max(0, (raw_data["brent_crude"] - PRE_WAR_BRENT_BASELINE) / PRE_WAR_BRENT_BASELINE)
    cpi_passthrough = brent_deviation_pct * 0.15
    forward["cpi_headline"] += cpi_passthrough
    forward["cpi_trimmed"] += cpi_passthrough  # Applying structurally to trimmed as well
    
    # Cash rate: override with ASX futures
    forward["cash_rate"] = raw_data["asx_cash_rate"]
    
    return forward

def calculate_base_probability(forward_data):
    """Calculate the base probability using weighted indicators."""
    base_score = 0.0
    contributions = {}
    
    for key, weight in WEIGHTS.items():
        score = calculate_indicator_score(forward_data[key], key)
        weighted_score = score * weight
        base_score += weighted_score
        contributions[key] = weighted_score * 100 # percentage contribution
        
    return base_score * 100, contributions

def calculate_overlays(raw_data):
    """Calculate the dynamic overlays to add to the base probability."""
    overlays = {}
    
    # Geo overlay: brent crude proxy
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
    # Cap: +15 pts max (prevents runaway if US odds jump to extreme values)
    kalshi_odds = raw_data.get("kalshi_recession", 25.0)
    kalshi_overlay = max(0.0, (kalshi_odds - 25.0) * 0.5)
    overlays["US Prediction Market (Kalshi)"] = min(15.0, kalshi_overlay)
    
    return overlays

def calculate_total_probability(raw_data):
    """
    Master function:
    1. Forwards prices the raw inputs.
    2. Calculates base probability.
    3. Calculates overlays.
    4. Sums and caps at 99%.
    """
    forward_data = apply_forward_pricing(raw_data)
    base_prob, contributions = calculate_base_probability(forward_data)
    overlays = calculate_overlays(raw_data)
    
    total_overlay = sum(overlays.values())
    total_prob = min(base_prob + total_overlay, 99.0)
    
    return {
        "total_probability": total_prob,
        "base_probability": base_prob,
        "contributions": contributions,
        "overlays": overlays,
        "forward_data": forward_data
    }
