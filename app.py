# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.title("🧬 Market Immune System")

# ----------------------------
# CONFIG
# ----------------------------
LOOKBACK_MONTHS = 24
INTERVAL = "1d"  # we resample manually to bi-weekly
RESAMPLE_RULE = "2W"

start_date = datetime.today() - timedelta(days=LOOKBACK_MONTHS * 30)

# ----------------------------
# DATA FETCH
# ----------------------------
@st.cache_data
def get_data():
    spy = yf.download("SPY", start=start_date, interval=INTERVAL)
    btc = yf.download("BTC-USD", start=start_date, interval=INTERVAL)

    vix = yf.download("^VIX", start=start_date, interval=INTERVAL)
    t10 = yf.download("^TNX", start=start_date, interval=INTERVAL)
    t2 = yf.download("^IRX", start=start_date, interval=INTERVAL)

    return spy, btc, vix, t10, t2


spy, btc, vix, t10, t2 = get_data()

# ----------------------------
# RESAMPLE TO BI-WEEKLY
# ----------------------------
def resample_ohlc(df):
    ohlc = df.resample(RESAMPLE_RULE).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    return ohlc.dropna()


spy = resample_ohlc(spy)
btc = resample_ohlc(btc)
vix = vix.resample(RESAMPLE_RULE).last()
t10 = t10.resample(RESAMPLE_RULE).last()
t2 = t2.resample(RESAMPLE_RULE).last()

# ----------------------------
# INDICATORS
# ----------------------------
def compute_metrics(df):
    df["ma20"] = df["Close"].rolling(20).mean()
    df["vol"] = df["Close"].pct_change().rolling(10).std() * np.sqrt(252)
    return df

spy = compute_metrics(spy)
btc = compute_metrics(btc)

# Yield curve (10Y - 3M proxy)
yield_curve = (t10["Close"] - t2["Close"]).reindex(spy.index)

# ----------------------------
# REGIME LOGIC
# ----------------------------
def compute_regime(df, yield_curve=None, vix=None):
    score = 0

    trend = df["Close"].iloc[-1] > df["ma20"].iloc[-1]
    vol_low = df["vol"].iloc[-1] < df["vol"].median()

    if trend:
        score += 1
    if vol_low:
        score += 1

    if yield_curve is not None:
        if yield_curve.iloc[-1] > 0:
            score += 1

    if vix is not None:
        if vix["Close"].iloc[-1] < 25:
            score += 1

    if score >= 3:
        regime = "RISK-ON"
        color = "rgba(0,200,0,0.25)"
    elif score == 2:
        regime = "NEUTRAL"
        color = "rgba(255,165,0,0.25)"
    else:
        regime = "RISK-OFF"
        color = "rgba(200,0,0,0.30)"

    return regime, score, color


spy_regime, spy_score, spy_color = compute_regime(spy, yield_curve, vix)
btc_regime, btc_score, btc_color = compute_regime(btc)

# ----------------------------
# PLOTTING FUNCTION (HOLLOW CANDLES)
# ----------------------------
def plot_chart(df, regime_color):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="green",
        decreasing_line_color="red",
        increasing_fillcolor="rgba(0,0,0,0)",  # hollow
        decreasing_fillcolor="rgba(0,0,0,0)",
        line_width=2
    ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=450
    )

    # Strong regime background
    fig.add_vrect(
        x0=df.index.min(),
        x1=df.index.max(),
        fillcolor=regime_color,
        opacity=0.35,
        layer="below",
        line_width=0
    )

    return fig


# ----------------------------
# LAYOUT
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("SPY Immune System")
    st.write(f"**Regime:** {spy_regime}")
    st.write(f"**Score:** {spy_score}")
    st.write(f"**Yield Curve (10Y-3M):** {yield_curve.iloc[-1]:.2f}")
    st.write(f"**VIX:** {vix['Close'].iloc[-1]:.2f}")
    st.write(f"**20-Period Volatility:** {spy['vol'].iloc[-1]:.2%}")

    st.plotly_chart(plot_chart(spy, spy_color), use_container_width=True)

with col2:
    st.subheader("BTC Immune System")
    st.write(f"**Regime:** {btc_regime}")
    st.write(f"**Score:** {btc_score}")
    st.write(f"**20-Period Volatility:** {btc['vol'].iloc[-1]:.2%}")

    st.plotly_chart(plot_chart(btc, btc_color), use_container_width=True)

# ----------------------------
# INTERPRETATION SECTION
# ----------------------------
st.markdown("---")
st.header("Interpretation")

st.markdown("### SPY Diagnostics")

if yield_curve.iloc[-1] > 0:
    st.write("• Yield curve is positive → economic expansion bias.")
else:
    st.write("• Yield curve inverted → recessionary risk elevated.")

if vix["Close"].iloc[-1] < 20:
    st.write("• VIX low → complacency / stable volatility regime.")
elif vix["Close"].iloc[-1] < 30:
    st.write("• VIX moderate → controlled stress.")
else:
    st.write("• VIX elevated → market stress.")

if spy["Close"].iloc[-1] > spy["ma20"].iloc[-1]:
    st.write("• Price above MA → bullish trend structure.")
else:
    st.write("• Price below MA → bearish trend structure.")

if spy["vol"].iloc[-1] < spy["vol"].median():
    st.write("• Realized volatility subdued.")
else:
    st.write("• Realized volatility elevated.")

st.markdown("### BTC Diagnostics")

if btc["Close"].iloc[-1] > btc["ma20"].iloc[-1]:
    st.write("• BTC trend positive.")
else:
    st.write("• BTC trend negative.")

if btc["vol"].iloc[-1] < btc["vol"].median():
    st.write("• BTC volatility controlled.")
else:
    st.write("• BTC volatility expanding.")
