import json
import os
from datetime import date

HISTORY_FILE = ".history.json"

def get_and_update_history(current_total, current_contribs):
    """
    Reads history, diffs against the 'anchor' (yesterday's final close),
    and updates the 'latest' values for today.
    """
    today_str = str(date.today())
    
    history = {
        "anchor_date": today_str,
        "anchor_total": current_total,
        "anchor_contribs": current_contribs,
        "latest_date": today_str,
        "latest_total": current_total,
        "latest_contribs": current_contribs,
    }
    
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                saved = json.load(f)
            
            # If the saved 'latest_date' is older than today, it becomes the new anchor
            if saved.get("latest_date") and saved["latest_date"] < today_str:
                history["anchor_date"] = saved["latest_date"]
                history["anchor_total"] = saved["latest_total"]
                history["anchor_contribs"] = saved["latest_contribs"]
            else:
                # Same day, keep the existing anchor
                history["anchor_date"] = saved.get("anchor_date", today_str)
                history["anchor_total"] = saved.get("anchor_total", current_total)
                history["anchor_contribs"] = saved.get("anchor_contribs", current_contribs)
                
        except Exception:
            pass # Use defaults if file is corrupted

    # Save the updated history (latest is always now)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except Exception:
        pass
        
    # Compute Diff
    if history["anchor_date"] == today_str:
        return None # No explicit diff if this is the very first run ever
        
    diff_total = current_total - history["anchor_total"]
    
    # Diff contributions
    contrib_diffs = []
    for k, v in current_contribs.items():
        old_v = history["anchor_contribs"].get(k, v)
        delta = v - old_v
        if abs(delta) > 0.1: # Only care about meaningful changes
            contrib_diffs.append((k, delta))
            
    # Sort by absolute magnitude of impact
    contrib_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "days_ago": (date.today() - date.fromisoformat(history["anchor_date"])).days,
        "diff_total": diff_total,
        "top_movers": contrib_diffs[:3] # Top 3 reasons
    }
