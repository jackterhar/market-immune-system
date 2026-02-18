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
# DOWNLOAD FUNCTION (DAILY)
# ==========================

def download_series(ticker):
    df = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df["Close"].dropna()

# ==========================
# CORE SYSTEM (DAILY)
# ==========================

spy = download_series("SPY")
hyg = download_series("HYG")
lqd = download_series("LQD")

common = spy.index.intersection(hyg.index).intersection(lqd.index)

spy = spy.loc[common]
hyg = hyg.loc[common]
lqd = lqd.loc[common]

df = pd.DataFrame(index=common)
df["SPY"] = spy
df["CreditRatio"] = hyg / lqd

# Indicators (Daily)
df["SPY_MA"] = df["SPY"].rolling(20).mean()
df["Credit_MA"] = df["CreditRatio"].rolling(20).mean()

df["SPY_Trend"] = np.where(df["SPY"] > df["SPY_MA"], 1, -1)
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

# Liquidity Trap (RSI 14D)
delta = df["SPY"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + rs))

df["LiquidityTrap"] = (
    (df["Regime"] == "Risk-Off") &
    (df["RSI"] < 30)
)

# ==========================
# BTC SYSTEM (DAILY)
# ==========================

btc = download_series("BTC-USD")

btc_df = pd.DataFrame(index=btc.index)
btc_df["BTC"] = btc
btc_df["BTC_MA"] = btc_df["BTC"].rolling(20).mean()
btc_df["BTC_Regime"] = np.where(
    btc_df["BTC"] > btc_df["BTC_MA"],
    "Bull",
    "Bear"
)

# ==========================
# DISPLAY WINDOWS (12M)
# ==========================

display_start_spy = df.index.max() - pd.DateOffset(months=DISPLAY_MONTHS)
display_spy = df.loc[display_start_spy:].copy()

display_start_btc = btc_df.index.max() - pd.DateOffset(months=DISPLAY_MONTHS)
display_btc = btc_df.loc[display_start_btc:].copy()

# ==========================
# CLEAN REGIME BLOCK SHADING
# ==========================

def add_regime_blocks(fig, data, regime_col, color):
    current_regime = data[regime_col].iloc[0]
    start_idx = data.index[0]

    for i in range(1, len(data)):
        if data[regime_col].iloc[i] != current_regime:
            if current_regime in ["Risk-Off", "Bear"]:
                fig.add_vrect(
                    x0=start_idx,
                    x1=data.index[i],
                    fillcolor=color,
                    opacity=0.12,
                    line_width=0
                )
            start_idx = data.index[i]
            current_regime = data[regime_col].iloc[i]

    # Final block
    if current_regime in ["Risk-Off", "Bear"]:
        fig.add_vrect(
            x0=start_idx,
            x1=data.index[-1],
            fillcolor=color,
            opacity=0.12,
            line_width=0
        )

# ==========================
# SPY CHART
# ==========================

spy_fig = go.Figure()

spy_fig.add_trace(go.Scatter(
    x=display_spy.index,
    y=display_spy["SPY"],
    mode="lines",
    line=dict(width=2),
    name="SPY"
))

add_regime_blocks(spy_fig, display_spy, "Regime", "red")

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
# BTC CHART
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

add_regime_blocks(btc_fig, display_btc, "BTC_Regime", "orange")

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
