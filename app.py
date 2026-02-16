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


# ================================
# SAFE DOWNLOAD FUNCTION
# ================================

@st.cache_data
def load_data():

    end = datetime.today()
    start = datetime(end.year - LOOKBACK_YEARS, end.month, end.day)

    spy = yf.download(
        "SPY",
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
    )

    if spy.empty:
        st.error("SPY data failed to download.")
        st.stop()

    # Flatten MultiIndex if needed
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # Standardize column names
    spy.columns = [c.capitalize() for c in spy.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in spy.columns]

    if missing:
        st.error(f"Missing columns from SPY data: {missing}")
        st.stop()

    # Download macro data
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
    hyg = yf.download("HYG", start=start, end=end, progress=False)["Close"]
    lqd = yf.download("LQD", start=start, end=end, progress=False)["Close"]
    tnx = yf.download("^TNX", start=start, end=end, progress=False)["Close"] / 10
    irx = yf.download("^IRX", start=start, end=end, progress=False)["Close"] / 10

    return spy, vix, hyg, lqd, tnx, irx


spy, vix, hyg, lqd, tnx, irx = load_data()

# ================================
# RESAMPLE
# ================================

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

# ================================
# INDICATORS
# ================================

spy["Return"] = spy["Close"].pct_change()
spy["Vol20"] = spy["Return"].rolling(4).std() * np.sqrt(52)

yield_curve = tnx - irx
credit_ratio = hyg / lqd

score = (
    (yield_curve > 0).astype(int)
    + (vix < 25).astype(int)
    + (credit_ratio > credit_ratio.rolling(20).mean()).astype(int)
    + (spy["Vol20"] < spy["Vol20"].rolling(20).mean()).astype(int)
)

def classify(x):
    if x >= 3:
        return "Risk On"
    elif x == 2:
        return "Neutral"
    else:
        return "Risk Off"

regime = score.apply(classify)

# ================================
# CHART
# ================================

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

last = regime.iloc[0]
start = regime.index[0]

for i in range(1, len(regime)):
    current = regime.iloc[i]
    date = regime.index[i]

    if current != last:
        fig.add_vrect(
            x0=start,
            x1=date,
            fillcolor=colors[last],
            layer="below",
            line_width=0
        )
        start = date
        last = current

fig.add_vrect(
    x0=start,
    x1=regime.index[-1],
    fillcolor=colors[last],
    layer="below",
    line_width=0
)

fig.update_layout(
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False,
    title="SPY with Regime Overlay"
)

st.plotly_chart(fig, use_container_width=True)

# ================================
# INTERPRETATION
# ================================

st.markdown("---")
st.header("📊 Interpretation Guide")

st.markdown("""
### 🧬 Immune Score
0–1 → Defensive  
2 → Neutral  
3–4 → Risk On  

### 📉 VIX
<20 → Calm  
20–30 → Stress  
>30 → Crisis  

### 🏦 Credit Ratio (HYG / LQD)
Rising → Risk appetite strong  
Falling → Credit stress  

### 📈 Yield Curve (10Y − 2Y)
Positive → Expansion  
Flat → Late cycle  
Negative → Recession risk  

### 📊 SPY 20D Vol
Low → Stable  
High → Instability  

### 🎯 Regimes
Risk On → Favor equities  
Neutral → Balanced  
Risk Off → Defensive posture  
""")
