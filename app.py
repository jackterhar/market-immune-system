import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System (Test Version)")

CALC_YEARS = 10
DISPLAY_MONTHS = 12

end = datetime.today()
start = datetime(end.year - CALC_YEARS, end.month, end.day)

# ==========================
# DOWNLOAD
# ==========================

def download_series(ticker):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()

spy = download_series("SPY")
hyg = download_series("HYG")
lqd = download_series("LQD")
vix = download_series("^VIX")
tnx = download_series("^TNX")  # 10Y
irx = download_series("^IRX")  # 3M
btc = download_series("BTC-USD")

# ==========================
# ALIGN CORE SYSTEM
# ==========================

core_index = spy.index.intersection(hyg.index).intersection(lqd.index)
spy = spy.loc[core_index]
hyg = hyg.loc[core_index]
lqd = lqd.loc[core_index]

df = pd.DataFrame(index=core_index)
df["SPY"] = spy
df["CreditRatio"] = hyg / lqd

# ==========================
# INDICATORS
# ==========================

# SPY Trend
df["SPY_MA"] = df["SPY"].rolling(20).mean()
df["SPY_Trend"] = np.where(df["SPY"] > df["SPY_MA"], 1, -1)

# Credit Trend
df["Credit_MA"] = df["CreditRatio"].rolling(20).mean()
df["Credit_Trend"] = np.where(df["CreditRatio"] > df["Credit_MA"], 1, -1)

# VIX Regime
vix = vix.loc[df.index]
df["VIX"] = vix
df["VIX_Regime"] = np.where(df["VIX"] < 20, 1, -1)

# Yield Curve
yc_index = tnx.index.intersection(irx.index).intersection(df.index)
yield_curve = (tnx.loc[yc_index] - irx.loc[yc_index])
df["YieldCurve"] = yield_curve
df["Yield_Regime"] = np.where(df["YieldCurve"] > 0, 1, -1)

# SPY 20D Volatility
returns = df["SPY"].pct_change()
df["SPY_Vol"] = returns.rolling(20).std() * np.sqrt(252)
df["Vol_Regime"] = np.where(df["SPY_Vol"] < df["SPY_Vol"].rolling(50).mean(), 1, -1)

# ==========================
# COMPOSITE SCORE
# ==========================

df["MacroScore"] = (
    df["SPY_Trend"] +
    df["Credit_Trend"] +
    df["VIX_Regime"] +
    df["Yield_Regime"] +
    df["Vol_Regime"]
)

def classify(score):
    if score >= 3:
        return "Risk-On"
    elif score <= -3:
        return "Risk-Off"
    else:
        return "Transition"

df["Regime"] = df["MacroScore"].apply(classify)

# ==========================
# BTC SYSTEM
# ==========================

btc_df = pd.DataFrame(index=btc.index)
btc_df["BTC"] = btc
btc_df["BTC_MA"] = btc_df["BTC"].rolling(20).mean()
btc_df["BTC_Regime"] = np.where(btc_df["BTC"] > btc_df["BTC_MA"], "Bull", "Bear")

# ==========================
# DISPLAY WINDOWS
# ==========================

display_spy = df.loc[df.index.max() - pd.DateOffset(months=DISPLAY_MONTHS):]
display_btc = btc_df.loc[btc_df.index.max() - pd.DateOffset(months=DISPLAY_MONTHS):]

# ==========================
# CONTINUOUS REGIME SHADING
# ==========================

def add_continuous_shading(fig, data, regime_col, color_map):
    for i in range(1, len(data)):
        regime = data[regime_col].iloc[i]
        fig.add_vrect(
            x0=data.index[i-1],
            x1=data.index[i],
            fillcolor=color_map[regime],
            opacity=0.15,
            line_width=0
        )

# Color maps
spy_colors = {
    "Risk-On": "green",
    "Transition": "gold",
    "Risk-Off": "red"
}

btc_colors = {
    "Bull": "green",
    "Bear": "red"
}

# ==========================
# SPY CHART
# ==========================

spy_fig = go.Figure()

add_continuous_shading(spy_fig, display_spy, "Regime", spy_colors)

spy_fig.add_trace(go.Scatter(
    x=display_spy.index,
    y=display_spy["SPY"],
    mode="lines",
    line=dict(color="white", width=2),
    name="SPY"
))

spy_fig.update_layout(
    template="plotly_dark",
    height=550,
    title="SPY (12M Macro Regime)"
)

st.plotly_chart(spy_fig, use_container_width=True)

# ==========================
# BTC CHART
# ==========================

st.markdown("---")
st.subheader("₿ Bitcoin")

btc_fig = go.Figure()

add_continuous_shading(btc_fig, display_btc, "BTC_Regime", btc_colors)

btc_fig.add_trace(go.Scatter(
    x=display_btc.index,
    y=display_btc["BTC"],
    mode="lines",
    line=dict(color="white", width=2),
    name="BTC"
))

btc_fig.update_layout(
    template="plotly_dark",
    height=500,
    title="BTC (12M Regime)"
)

st.plotly_chart(btc_fig, use_container_width=True)

# ==========================
# INTERPRETATION PANEL
# ==========================

st.markdown("---")
st.subheader("Macro Diagnostics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Macro Regime", display_spy["Regime"].iloc[-1])
c2.metric("VIX", round(display_spy["VIX"].iloc[-1],2))
c3.metric("Yield Curve", round(display_spy["YieldCurve"].iloc[-1],2))
c4.metric("SPY 20D Vol", round(display_spy["SPY_Vol"].iloc[-1],2))

st.write("Composite Macro Score:", display_spy["MacroScore"].iloc[-1])
