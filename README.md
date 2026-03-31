# Australia Recession Probability Monitor

A live-updating macroeconomic dashboard that forward-prices Australia's recession risk based on core economic data, forward market curves, and high-frequency sentiment overlays.

## Overview
This Streamlit application pulls real-time data from:
- **ABS API**: Growth, employment, consumer prices, and wages.
- **RBA**: Cash rate endpoints.
- **ASIC**: Monthly insolvency scraping.
- **ASX Futures**: Implied 6-month forward cash rate.
- **Alpha Vantage**: Brent crude trajectory (geopolitical proxy).
- **Westpac-Melbourne Institute**: Consumer sentiment indexing.
- **Google Trends**: High-frequency search volume (mortgage stress, job losses, etc.)
- **Official Media**: Keyword density on official domains (RBA, Treasury, PM).

## Setup Instructions

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Alpha Vantage API Key**:
   The model requires an Alpha Vantage API key for WTI/Brent crude data.
   - You can get a free tier key (25 calls/day) at [Alpha Vantage](https://www.alphavantage.co/support/#api-key).
   - Your key is currently populated in the `.env` file. If deploying to cloud, add `ALPHAVANTAGE_KEY` to your secrets manager.

4. **Run Locally**:
   ```bash
   streamlit run app.py
   ```

## Deployment to Streamlit Cloud

1. Push this repository to a public or private GitHub repository.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app", select your repository, branch, and `app.py` as the main file path.
4. Expand **Advanced Settings**:
   - Add your secrets (Environment variables), specifically:
     `ALPHAVANTAGE_KEY = "your_key_here"`
5. Click **Deploy**.

## Setting up Cache Refresh via GitHub Actions
To ensure data stays fresh (especially if no users visit the app for a while), you can create a cron job to poke the app daily. 
Create `.github/workflows/refresh.yml`:

```yaml
name: Daily Cache Refresh
on:
  schedule:
    - cron: '0 0 * * *' # Midnight UTC

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Streamlit App
        run: curl -s https://your-streamlit-app-url.streamlit.app > /dev/null
```
