import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System (Test Version)")

# -------------------
# SETTINGS
# -------------------
LOOKBACK_MONTHS = 24
RESAMPLE_RULE = "2W"
start_date = datetime.today() - timedelta(days=LOOKBACK_MONTHS * 30)

# -------------------
# DATA
# -------------------
spy = yf.download("SPY", start=start_date)

spy = spy.resample(RESAMPLE_RULE).agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
}).dropna()

# Indicators
spy["ma20"] = spy["Close"].rolling(20).mean()
spy["vol20"] = spy["Close"].pct_change().rolling(20).std() * np.sqrt(252)

# Simple regime logic
def regime(i):
    score = 0
    if spy["Close"].iloc[i] > spy["ma20"].iloc[i]:
        score += 1
    if spy["vol20"].iloc[i] < spy["vol20"].median():
        score += 1

    if score == 2:
        return "RISK-ON", "rgba(0,255,0,0.6)"
    elif score == 1:
        return "NEUTRAL", "rgba(255,165,0,0.6)"
    else:
        return "DEFENSIVE", "rgba(255,0,0,0.65)"

current_regime, _ = regime(-1)

# -------------------
# PLOT
# -------------------
fig = go.Figure()

# Add regime blocks (grouped by consecutive regime)
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
