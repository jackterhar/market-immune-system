# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System")

# -------------------------
# CONFIG
# -------------------------
LOOKBACK_MONTHS = 24
RESAMPLE_RULE = "2W"
start_date = datetime.today() - timedelta(days=LOOKBACK_MONTHS * 30)

# -------------------------
# DATA
# -------------------------
@st.cache_data
def load_data():
    spy = yf.download("SPY", start=start_date)
    btc = yf.download("BTC-USD", start=start_date)
    vix = yf.download("^VIX", start=start_date)
    t10 = yf.download("^TNX", start=start_date)
    t3m = yf.download("^IRX", start=start_date)
    hyg = yf.download("HYG", start=start_date)
    lqd = yf.download("LQD", start=start_date)

    return spy, btc, vix, t10, t3m, hyg, lqd


spy, btc, vix, t10, t3m, hyg, lqd = load_data()

# -------------------------
# RESAMPLE BI-WEEKLY
# -------------------------
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

# -------------------------
# METRICS
# -------------------------
spy["ma20"] = spy["Close"].rolling(20).mean()
spy["vol20"] = spy["Close"].pct_change().rolling(20).std() * np.sqrt(252)

btc["ma20"] = btc["Close"].rolling(20).mean()
btc["vol20"] = btc["Close"].pct_change().rolling(20).std() * np.sqrt(252)

yield_curve = (t10["Close"] - t3m["Close"]).reindex(spy.index)
credit_ratio = (hyg["Close"] / lqd["Close"]).reindex(spy.index)

# -------------------------
# REGIME SCORING
# -------------------------
def compute_regime():
    score = 0

    trend = spy["Close"].iloc[-1] > spy["ma20"].iloc[-1]
    vol_ok = spy["vol20"].iloc[-1] < spy["vol20"].median()
    yc_ok = yield_curve.iloc[-1] > 0
    credit_ok = credit_ratio.iloc[-1] > credit_ratio.median()
    vix_ok = vix["Close"].iloc[-1] < 25

    components = {
        "Trend": trend,
        "Volatility": vol_ok,
        "YieldCurve": yc_ok,
        "Credit": credit_ok,
        "VIX": vix_ok
    }

    score = sum(components.values())

    if score >= 4:
        regime = "RISK-ON"
        color = "rgba(0,255,0,0.55)"   # strong green
    elif score >= 2:
        regime = "NEUTRAL"
        color = "rgba(255,165,0,0.55)" # strong orange
    else:
        regime = "RISK-OFF"
        color = "rgba(255,0,0,0.60)"   # strong red

    return regime, score, color, components


spy_regime, spy_score, spy_color, spy_components = compute_regime()

# -------------------------
# CANDLE PLOT
# -------------------------
def plot_candles(df, regime_color):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="lime",
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

    # Strong full background regime
    fig.add_vrect(
        x0=df.index.min(),
        x1=df.index.max(),
        fillcolor=regime_color,
        opacity=0.6,
        layer="below",
        line_width=0
    )

    return fig


# -------------------------
# LAYOUT
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("SPY Immune System")

    st.write(f"**Regime:** {spy_regime}")
    st.write(f"**Score:** {spy_score} / 5")
    st.write(f"**VIX:** {vix['Close'].iloc[-1]:.2f}")
    st.write(f"**Credit Ratio (HYG/LQD):** {credit_ratio.iloc[-1]:.3f}")
    st.write(f"**Yield Curve (10Y-3M):** {yield_curve.iloc[-1]:.2f}")
    st.write(f"**SPY 20D Vol:** {spy['vol20'].iloc[-1]:.2%}")

    st.plotly_chart(plot_candles(spy, spy_color), use_container_width=True)

with col2:
    st.subheader("BTC Immune System")

    btc_trend = btc["Close"].iloc[-1] > btc["ma20"].iloc[-1]
    btc_vol_ok = btc["vol20"].iloc[-1] < btc["vol20"].median()
    btc_score = int(btc_trend) + int(btc_vol_ok)

    st.write(f"**Trend Positive:** {btc_trend}")
    st.write(f"**Volatility Controlled:** {btc_vol_ok}")
    st.write(f"**Score:** {btc_score} / 2")

    st.plotly_chart(plot_candles(btc, spy_color), use_container_width=True)

# -------------------------
# INTERPRETATION
# -------------------------
st.markdown("---")
st.header("Interpretation")

st.markdown("### Regime Score")

if spy_score >= 4:
    st.write("High systemic alignment across macro + technical factors. Broad risk exposure statistically favored.")
elif spy_score >= 2:
    st.write("Mixed signals. Market internally unstable or transitioning.")
else:
    st.write("Multiple stress indicators active. Defensive posture statistically favored.")

st.markdown("### VIX")

v = vix["Close"].iloc[-1]
if v < 18:
    st.write("Volatility compression regime. Risk appetite strong.")
elif v < 25:
    st.write("Moderate volatility. Normalized risk.")
else:
    st.write("Elevated volatility. Market stress conditions present.")

st.markdown("### Credit Ratio (HYG/LQD)")

if credit_ratio.iloc[-1] > credit_ratio.median():
    st.write("Credit spreads tightening → capital flowing into risk debt.")
else:
    st.write("Credit spreads widening → risk aversion increasing.")

st.markdown("### Yield Curve")

if yield_curve.iloc[-1] > 0:
    st.write("Positive slope → economic expansion bias.")
else:
    st.write("Inversion → recession probability elevated.")

st.markdown("### SPY 20D Realized Volatility")

if spy["vol20"].iloc[-1] < spy["vol20"].median():
    st.write("Volatility subdued → stable market regime.")
else:
    st.write("Volatility elevated → unstable price dynamics.")
