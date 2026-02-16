import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System (Test Version)")

RESAMPLE_RULE = "W"
LOOKBACK_YEARS = 10


# =====================================================
# SAFE DOWNLOAD HELPERS
# =====================================================

@st.cache_data
def download_series(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        st.error(f"{ticker} failed to download.")
        st.stop()

    if isinstance(data, pd.DataFrame):
        if "Close" in data.columns:
            return data["Close"]
        else:
            return data.iloc[:, 0]

    return data


@st.cache_data
def download_spy(start, end):
    spy = yf.download(
        "SPY",
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
    )

    if spy.empty:
        st.error("SPY failed to download.")
        st.stop()

    # Flatten MultiIndex if needed
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # Standardize names
    spy.columns = [c.capitalize() for c in spy.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in spy.columns]

    if missing:
        st.error(f"Missing SPY columns: {missing}")
        st.stop()

    return spy


# =====================================================
# LOAD DATA
# =====================================================

end = datetime.today()
start = datetime(end.year - LOOKBACK_YEARS, end.month, end.day)

spy = download_spy(start, end)

vix = download_series("^VIX", start, end)
hyg = download_series("HYG", start, end)
lqd = download_series("LQD", start, end)
tnx = download_series("^TNX", start, end) / 10
irx = download_series("^IRX", start, end) / 10


# =====================================================
# RESAMPLE
# =====================================================

spy = spy.resample(RESAMPLE_RULE).agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
}).dropna()

vix = vix.resample(RESAMPLE_RULE).last()
hyg = hyg.resample(RESAMPLE_RULE).last()
lqd = lqd.resample(RESAMPLE_RULE).last()
tnx = tnx.resample(RESAMPLE_RULE).last()
irx = irx.resample(RESAMPLE_RULE).last()


# =====================================================
# INDICATORS (FULLY ALIGNED)
# =====================================================

spy["Return"] = spy["Close"].pct_change()

df = pd.DataFrame(index=spy.index)

df["YieldCurve"] = tnx - irx
df["VIX"] = vix
df["CreditRatio"] = hyg / lqd
df["Vol20"] = spy["Return"].rolling(4).std() * np.sqrt(52)

df = df.dropna()

df["Score"] = (
    (df["YieldCurve"] > 0).astype(int)
    + (df["VIX"] < 25).astype(int)
    + (df["CreditRatio"] > df["CreditRatio"].rolling(20).mean()).astype(int)
    + (df["Vol20"] < df["Vol20"].rolling(20).mean()).astype(int)
)

df["Regime"] = np.select(
    [
        df["Score"] >= 3,
        df["Score"] == 2
    ],
    [
        "Risk On",
        "Neutral"
    ],
    default="Risk Off"
)

regime = df["Regime"]


# =====================================================
# CHART
# =====================================================

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=spy.index,
    open=spy["Open"],
    high=spy["High"],
    low=spy["Low"],
    close=spy["Close"],
    increasing=dict(line=dict(color="lime", width=2), fillcolor="rgba(0,0,0,0)"),
    decreasing=dict(line=dict(color="red", width=2), fillcolor="rgba(0,0,0,0)"),
    name="SPY"
))

colors = {
    "Risk On": "rgba(0,200,0,0.30)",
    "Neutral": "rgba(255,200,0,0.30)",
    "Risk Off": "rgba(200,0,0,0.30)"
}

last_regime = regime.iloc[0]
start_date = regime.index[0]

for i in range(1, len(regime)):
    current = regime.iloc[i]
    date = regime.index[i]

    if current != last_regime:
        fig.add_vrect(
            x0=start_date,
            x1=date,
            fillcolor=colors[last_regime],
            layer="below",
            line_width=0
        )
        start_date = date
        last_regime = current

fig.add_vrect(
    x0=start_date,
    x1=regime.index[-1],
    fillcolor=colors[last_regime],
    layer="below",
    line_width=0
)

fig.update_layout(
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False,
    title="SPY with Market Regime Overlay"
)

st.plotly_chart(fig, use_container_width=True)


# =====================================================
# INTERPRETATION GUIDE
# =====================================================

st.markdown("---")
st.header("📊 Interpretation Guide")

st.markdown("""
### 🧬 Immune Score
0–1 → Defensive  
2 → Neutral / Transition  
3–4 → Risk On  

### 📉 VIX
<20 → Calm  
20–30 → Stress building  
>30 → Crisis regime  

### 🏦 Credit Ratio (HYG / LQD)
Rising → Investors favor high-yield (risk appetite strong)  
Falling → Flight to quality (credit stress)

### 📈 Yield Curve (10Y − 2Y)
Positive → Expansionary environment  
Flat → Late cycle  
Negative → Recession probability elevated  

### 📊 SPY 20D Volatility
Low → Stable market  
Moderate → Normal fluctuations  
High → Instability / drawdown environment  

### 🎯 Regime Meaning
Risk On → Favor equities & cyclicals  
Neutral → Balanced allocation  
Risk Off → Defensive assets favored  
""")
