# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System")

# -----------------------
# CONFIG
# -----------------------
LOOKBACK_MONTHS = 24
RESAMPLE_RULE = "2W"
start_date = datetime.today() - timedelta(days=LOOKBACK_MONTHS * 30)

# -----------------------
# DATA
# -----------------------
@st.cache_data
def load():
    spy = yf.download("SPY", start=start_date)
    btc = yf.download("BTC-USD", start=start_date)
    vix = yf.download("^VIX", start=start_date)
    t10 = yf.download("^TNX", start=start_date)
    t3m = yf.download("^IRX", start=start_date)
    hyg = yf.download("HYG", start=start_date)
    lqd = yf.download("LQD", start=start_date)
    return spy, btc, vix, t10, t3m, hyg, lqd

spy, btc, vix, t10, t3m, hyg, lqd = load()

def resample(df):
    return df.resample(RESAMPLE_RULE).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

spy = resample(spy)
btc = resample(btc)
vix = vix.resample(RESAMPLE_RULE).last()
t10 = t10.resample(RESAMPLE_RULE).last()
t3m = t3m.resample(RESAMPLE_RULE).last()
hyg = hyg.resample(RESAMPLE_RULE).last()
lqd = lqd.resample(RESAMPLE_RULE).last()

# -----------------------
# METRICS
# -----------------------
spy["ma20"] = spy["Close"].rolling(20).mean()
spy["vol20"] = spy["Close"].pct_change().rolling(20).std() * np.sqrt(252)

btc["ma20"] = btc["Close"].rolling(20).mean()
btc["vol20"] = btc["Close"].pct_change().rolling(20).std() * np.sqrt(252)

yield_curve = (t10["Close"] - t3m["Close"]).reindex(spy.index)
credit_ratio = (hyg["Close"] / lqd["Close"]).reindex(spy.index)

# -----------------------
# REGIME FUNCTION
# -----------------------
def regime_for_bar(i):
    score = 0

    if spy["Close"].iloc[i] > spy["ma20"].iloc[i]:
        score += 1
    if spy["vol20"].iloc[i] < spy["vol20"].median():
        score += 1
    if yield_curve.iloc[i] > 0:
        score += 1
    if credit_ratio.iloc[i] > credit_ratio.median():
        score += 1
    if vix["Close"].iloc[i] < 25:
        score += 1

    if score >= 4:
        return "RISK-ON", "rgba(0,255,0,0.35)"
    elif score >= 2:
        return "NEUTRAL", "rgba(255,165,0,0.35)"
    else:
        return "DEFENSIVE", "rgba(255,0,0,0.40)"

# current regime
current_regime, current_color = regime_for_bar(-1)

# -----------------------
# PLOT FUNCTION
# -----------------------
def plot_candles(df):
    fig = go.Figure()

    # Background bands per regime period
    for i in range(len(df)-1):
        regime, color = regime_for_bar(i)
        fig.add_vrect(
            x0=df.index[i],
            x1=df.index[i+1],
            fillcolor=color,
            opacity=0.45,
            layer="below",
            line_width=0
        )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
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
        height=450
    )

    return fig

# -----------------------
# LAYOUT
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("SPY Immune System")
    st.write(f"**Regime:** {current_regime}")

    score_now = sum([
        spy["Close"].iloc[-1] > spy["ma20"].iloc[-1],
        spy["vol20"].iloc[-1] < spy["vol20"].median(),
        yield_curve.iloc[-1] > 0,
        credit_ratio.iloc[-1] > credit_ratio.median(),
        vix["Close"].iloc[-1] < 25
    ])

    st.write(f"**Score:** {score_now}/5")
    st.write(f"**VIX:** {vix['Close'].iloc[-1]:.2f}")
    st.write(f"**Credit Ratio:** {credit_ratio.iloc[-1]:.3f}")
    st.write(f"**Yield Curve:** {yield_curve.iloc[-1]:.2f}")
    st.write(f"**SPY 20D Vol:** {spy['vol20'].iloc[-1]:.2%}")

    st.plotly_chart(plot_candles(spy), use_container_width=True)

with col2:
    st.subheader("BTC Immune System")

    btc_trend = btc["Close"].iloc[-1] > btc["ma20"].iloc[-1]
    btc_vol = btc["vol20"].iloc[-1] < btc["vol20"].median()
    btc_score = int(btc_trend) + int(btc_vol)

    btc_regime = "RISK-ON" if btc_score == 2 else "DEFENSIVE"

    st.write(f"**Regime:** {btc_regime}")
    st.write(f"**Score:** {btc_score}/2")
    st.write(f"**BTC 20D Vol:** {btc['vol20'].iloc[-1]:.2%}")

    st.plotly_chart(plot_candles(btc), use_container_width=True)

# -----------------------
# INTERPRETATION
# -----------------------
st.markdown("---")
st.header("How to Interpret")

st.write(f"**Score ({score_now}/5):** Composite immune alignment across trend, volatility, credit, yield curve, and VIX.")

st.write("**VIX:** Measures implied volatility. Elevated readings signal systemic stress.")

st.write("**Credit Ratio (HYG/LQD):** Risk appetite gauge. Rising ratio = capital flowing into high yield.")

st.write("**Yield Curve (10Y-3M):** Inversion historically precedes recessionary slowdowns.")

st.write("**SPY 20D Vol:** Realized volatility regime. Expanding vol increases drawdown risk.")
