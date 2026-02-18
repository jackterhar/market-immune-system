import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System (Test Version)")

# ==========================
# PARAMETERS
# ==========================

CALC_YEARS = 10
DISPLAY_MONTHS = 12
RESAMPLE_RULE = "W-FRI"

end = datetime.today()
start = datetime(end.year - CALC_YEARS, end.month, end.day)

# ==========================
# SAFE DOWNLOAD
# ==========================

def download_series(ticker):
    df = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    series = df["Close"].copy()
    series.name = ticker
    return series.resample(RESAMPLE_RULE).last().dropna()

# ==========================
# SPY + CREDIT SYSTEM (CORE)
# ==========================

spy = download_series("SPY")
hyg = download_series("HYG")
lqd = download_series("LQD")

# Align credit only to SPY
credit_index = spy.index.intersection(hyg.index).intersection(lqd.index)

spy = spy.loc[credit_index]
hyg = hyg.loc[credit_index]
lqd = lqd.loc[credit_index]

df = pd.DataFrame(index=credit_index)
df["SPY"] = spy
df["CreditRatio"] = hyg / lqd

# Indicators
df["SPY_MA"] = df["SPY"].rolling(20).mean()
df["SPY_Trend"] = np.where(df["SPY"] > df["SPY_MA"], 1, -1)

df["Credit_MA"] = df["CreditRatio"].rolling(20).mean()
df["Credit_Trend"] = np.where(df["CreditRatio"] > df["Credit_MA"], 1, -1)

df["Score"] = df["SPY_Trend"] + df["Credit_Trend"]

def classify(x):
    if x == 2:
        return "Risk-On"
    elif x == -2:
        return "Risk-Off"
    else:
        return "Transition"

df["Regime"] = df["Score"].apply(classify)

# Liquidity trap
delta = df["SPY"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + rs))

df["LiquidityTrap"] = (
    (df["Regime"] == "Risk-Off") &
    (df["RSI"] < 30)
)

# ==========================
# BTC (INDEPENDENT SYSTEM)
# ==========================

btc = download_series("BTC-USD")

btc_df = pd.DataFrame(index=btc.index)
btc_df["BTC"] = btc
btc_df["BTC_MA"] = btc_df["BTC"].rolling(20).mean()
btc_df["BTC_Trend"] = np.where(btc_df["BTC"] > btc_df["BTC_MA"], 1, -1)
btc_df["BTC_Regime"] = np.where(btc_df["BTC_Trend"] == 1, "Bull", "Bear")

# ==========================
# DISPLAY WINDOWS
# ==========================

display_start_spy = df.index.max() - pd.DateOffset(months=DISPLAY_MONTHS)
display_spy = df.loc[display_start_spy:]

display_start_btc = btc_df.index.max() - pd.DateOffset(months=DISPLAY_MONTHS)
display_btc = btc_df.loc[display_start_btc:]

# ==========================
# SPY CHART (RESTORED FEEL)
# ==========================

spy_fig = go.Figure()

spy_fig.add_trace(go.Scatter(
    x=display_spy.index,
    y=display_spy["SPY"],
    mode="lines",
    line=dict(width=2),
    name="SPY"
))

for i in range(1, len(display_spy)):
    if display_spy["Regime"].iloc[i] == "Risk-Off":
        spy_fig.add_vrect(
            x0=display_spy.index[i-1],
            x1=display_spy.index[i],
            fillcolor="red",
            opacity=0.12,
            line_width=0
        )

trap = display_spy[display_spy["LiquidityTrap"]]

spy_fig.add_trace(go.Scatter(
    x=trap.index,
    y=trap["SPY"],
    mode="markers",
    marker=dict(size=6),
    name="Liquidity Trap"
))

spy_fig.update_layout(
    template="plotly_dark",
    height=550,
    title="SPY (12M Regime View)"
)

st.plotly_chart(spy_fig, use_container_width=True)

# ==========================
# BTC CHART (CLEAN & SEPARATE)
# ==========================

st.markdown("---")
st.subheader("₿ Bitcoin")

btc_fig = go.Figure()

btc_fig.add_trace(go.Scatter(
    x=display_btc.index,
    y=display_btc["BTC"],
    mode="lines",
    line=dict(width=2),
    name="BTC"
))

for i in range(1, len(display_btc)):
    if display_btc["BTC_Regime"].iloc[i] == "Bear":
        btc_fig.add_vrect(
            x0=display_btc.index[i-1],
            x1=display_btc.index[i],
            fillcolor="orange",
            opacity=0.08,
            line_width=0
        )

btc_fig.update_layout(
    template="plotly_dark",
    height=500,
    title="BTC (12M View)"
)

st.plotly_chart(btc_fig, use_container_width=True)

# ==========================
# METRICS
# ==========================

st.markdown("---")
c1, c2 = st.columns(2)

c1.metric("SPY Regime", display_spy["Regime"].iloc[-1])
c2.metric("BTC Regime", display_btc["BTC_Regime"].iloc[-1])
