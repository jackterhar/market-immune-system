import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

st.title("🧬 Market Immune System (Test Version)")

# ================================
# SETTINGS
# ================================

RESAMPLE_RULE = "W"   # Weekly candles
LOOKBACK_YEARS = 10

# ================================
# DATA DOWNLOAD
# ================================

@st.cache_data
def load_data():

    end = datetime.today()
    start = datetime(end.year - LOOKBACK_YEARS, end.month, end.day)

    spy = yf.download("SPY", start=start, end=end, auto_adjust=False)
    vix = yf.download("^VIX", start=start, end=end)["Close"]
    hyg = yf.download("HYG", start=start, end=end)["Close"]
    lqd = yf.download("LQD", start=start, end=end)["Close"]
    dgs10 = yf.download("^TNX", start=start, end=end)["Close"] / 10
    dgs2 = yf.download("^IRX", start=start, end=end)["Close"] / 10

    # Validate columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in spy.columns:
            st.error(f"Missing SPY column: {col}")
            st.stop()

    return spy, vix, hyg, lqd, dgs10, dgs2


spy, vix, hyg, lqd, dgs10, dgs2 = load_data()

# ================================
# RESAMPLE OHLC PROPERLY
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
dgs10 = dgs10.resample(RESAMPLE_RULE).last()
dgs2 = dgs2.resample(RESAMPLE_RULE).last()

# ================================
# INDICATORS
# ================================

spy["Return"] = spy["Close"].pct_change()
spy["20D Vol"] = spy["Return"].rolling(4).std() * np.sqrt(52)

yield_curve = dgs10 - dgs2
credit_ratio = hyg / lqd

score = (
    (yield_curve > 0).astype(int) +
    (vix < 25).astype(int) +
    (credit_ratio > credit_ratio.rolling(20).mean()).astype(int) +
    (spy["20D Vol"] < spy["20D Vol"].rolling(20).mean()).astype(int)
)

# ================================
# REGIME CLASSIFICATION
# ================================

def classify_regime(s):
    if s >= 3:
        return "Risk On"
    elif s == 2:
        return "Neutral"
    else:
        return "Risk Off"

regime = score.apply(classify_regime)

# ================================
# PLOT
# ================================

fig = go.Figure()

# Hollow Candles
fig.add_trace(go.Candlestick(
    x=spy.index,
    open=spy["Open"],
    high=spy["High"],
    low=spy["Low"],
    close=spy["Close"],
    increasing=dict(
        line=dict(color="lime", width=2),
        fillcolor="rgba(0,0,0,0)"
    ),
    decreasing=dict(
        line=dict(color="red", width=2),
        fillcolor="rgba(0,0,0,0)"
    ),
    name="SPY"
))

# ================================
# STRONG REGIME BACKGROUND
# ================================

colors = {
    "Risk On": "rgba(0,200,0,0.25)",
    "Neutral": "rgba(255,200,0,0.25)",
    "Risk Off": "rgba(200,0,0,0.25)"
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

# Add final segment
fig.add_vrect(
    x0=start_date,
    x1=regime.index[-1],
    fillcolor=colors[last_regime],
    layer="below",
    line_width=0
)

fig.update_layout(
    height=700,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    title="SPY with Market Regime Overlay"
)

st.plotly_chart(fig, use_container_width=True)

# ================================
# INTERPRETATION SECTION
# ================================

st.markdown("---")
st.header("📊 Interpretation Guide")

st.markdown("""
### 🧬 Immune Score
- **4** → Strong systemic health (risk assets supported)
- **3** → Healthy but monitor
- **2** → Neutral / Transition
- **1–0** → Defensive posture warranted

### 📉 VIX (Volatility Index)
- **< 20** → Calm markets
- **20–30** → Elevated stress
- **> 30** → Crisis regime

### 🏦 Credit Ratio (HYG / LQD)
- Rising → Investors favor high-yield → Risk appetite strong  
- Falling → Flight to quality → Credit stress building

### 📈 Yield Curve (10Y − 2Y)
- Positive → Expansionary conditions  
- Flat → Late-cycle risk  
- Negative → Recession probability elevated

### 📊 SPY 20D Volatility
- Below 6% → Stable regime  
- 6–12% → Elevated but normal  
- > 12% → Instability / drawdown environment

### 🎯 Regime Labels
- **Risk On** → Favor equities, cyclicals  
- **Neutral** → Balanced allocation  
- **Risk Off** → Defensive assets favored  
""")
# -----------------------
# INDICATORS
# -----------------------
spy["ma20"] = spy["Close"].rolling(20).mean()
spy["vol20"] = spy["Close"].pct_change().rolling(20).std() * np.sqrt(252)

# -----------------------
# REGIME FUNCTION
# -----------------------
def regime(i):
    score = 0

    if spy["Close"].iloc[i] > spy["ma20"].iloc[i]:
        score += 1

    if spy["vol20"].iloc[i] < spy["vol20"].median():
        score += 1

    if score == 2:
        return "RISK-ON", "rgba(0,255,0,0.75)"
    elif score == 1:
        return "NEUTRAL", "rgba(255,165,0,0.75)"
    else:
        return "DEFENSIVE", "rgba(255,0,0,0.8)"

current_regime, _ = regime(-1)

# -----------------------
# PLOT
# -----------------------
fig = go.Figure()

# Create clean regime blocks
last_regime = None
block_start = None

for i in range(len(spy)):
    r, color = regime(i)

    if r != last_regime:
        if block_start is not None:
            fig.add_vrect(
                x0=block_start,
                x1=spy.index[i],
                fillcolor=last_color,
                opacity=0.8,
                layer="below",
                line_width=0
            )
        block_start = spy.index[i]
        last_regime = r
        last_color = color

# Close final block
fig.add_vrect(
    x0=block_start,
    x1=spy.index[-1],
    fillcolor=last_color,
    opacity=0.8,
    layer="below",
    line_width=0
)

# Hollow Candlestick
fig.add_trace(go.Candlestick(
    x=spy.index,
    open=spy["Open"],
    high=spy["High"],
    low=spy["Low"],
    close=spy["Close"],
    increasing_line_color="lime",
    decreasing_line_color="red",
    increasing_fillcolor="rgba(0,0,0,0)",
    decreasing_fillcolor="rgba(0,0,0,0)",
    line_width=2
))

fig.update_layout(
    xaxis_rangeslider_visible=False,
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    height=500
)

# -----------------------
# DISPLAY
# -----------------------
st.subheader("SPY Immune System")
st.write(f"**Current Regime:** {current_regime}")
st.write(f"**20D Volatility:** {spy['vol20'].iloc[-1]:.2%}")

st.plotly_chart(fig, use_container_width=True)    if r != last_regime:
        if block_start is not None:
            fig.add_vrect(
                x0=block_start,
                x1=spy.index[i],
                fillcolor=last_color,
                opacity=0.65,
                layer="below",
                line_width=0
            )
        block_start = spy.index[i]
        last_regime = r
        last_color = color

# close final block
fig.add_vrect(
    x0=block_start,
    x1=spy.index[-1],
    fillcolor=last_color,
    opacity=0.65,
    layer="below",
    line_width=0
)

# Hollow Candlesticks
fig.add_trace(go.Candlestick(
    x=spy.index,
    open=spy["Open"],
    high=spy["High"],
    low=spy["Low"],
    close=spy["Close"],
    increasing_line_color="lime",
    decreasing_line_color="red",
    increasing_fillcolor="rgba(0,0,0,0)",
    decreasing_fillcolor="rgba(0,0,0,0)",
    line_width=2
))

fig.update_layout(
    xaxis_rangeslider_visible=False,
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    height=500
)

st.subheader("SPY Immune System")
st.write(f"**Current Regime:** {current_regime}")
st.plotly_chart(fig, use_container_width=True)
