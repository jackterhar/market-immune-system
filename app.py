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
# SAFE DOWNLOAD FUNCTION
# ==========================

def download_clean_series(ticker):
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        st.error(f"No data returned for {ticker}")
        st.stop()

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        st.error(f"{ticker} missing Close column")
        st.stop()

    series = df["Close"].copy()
    series.name = ticker

    # Force to true Series
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    return series

def resample_series(series):
    return series.resample(RESAMPLE_RULE).last().dropna()

# ==========================
# DOWNLOAD DATA
# ==========================

spy = resample_series(download_clean_series("SPY"))
btc = resample_series(download_clean_series("BTC-USD"))
hyg = resample_series(download_clean_series("HYG"))
lqd = resample_series(download_clean_series("LQD"))

# ==========================
# ALIGN INDEX SAFELY
# ==========================

common_index = spy.index
common_index = common_index.intersection(btc.index)
common_index = common_index.intersection(hyg.index)
common_index = common_index.intersection(lqd.index)

spy = spy.loc[common_index]
btc = btc.loc[common_index]
hyg = hyg.loc[common_index]
lqd = lqd.loc[common_index]

# ==========================
# BUILD MASTER DATAFRAME
# ==========================

df = pd.DataFrame(index=common_index)
df["SPY"] = spy.values
df["BTC"] = btc.values
df["HYG"] = hyg.values
df["LQD"] = lqd.values

df["CreditRatio"] = df["HYG"] / df["LQD"]

# ==========================
# INDICATORS
# ==========================

df["SPY_MA"] = df["SPY"].rolling(20).mean()
df["SPY_Trend"] = np.where(df["SPY"] > df["SPY_MA"], 1, -1)

df["Credit_MA"] = df["CreditRatio"].rolling(20).mean()
df["Credit_Trend"] = np.where(df["CreditRatio"] > df["Credit_MA"], 1, -1)

df["BTC_MA"] = df["BTC"].rolling(20).mean()
df["BTC_Trend"] = np.where(df["BTC"] > df["BTC_MA"], 1, -1)

# ==========================
# REGIME CLASSIFICATION
# ==========================

df["Score"] = df["SPY_Trend"] + df["Credit_Trend"]

def classify(score):
    if score == 2:
        return "Risk-On"
    elif score == -2:
        return "Risk-Off"
    else:
        return "Transition"

df["Regime"] = df["Score"].apply(classify)
df["BTC_Regime"] = np.where(df["BTC_Trend"] == 1, "Bull", "Bear")

# ==========================
# LIQUIDITY TRAP DETECTOR
# ==========================

delta = df["SPY"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

df["LiquidityTrap"] = (
    (df["Regime"] == "Risk-Off") &
    (df["RSI"] < 30)
)

# ==========================
# CRYPTOQUANT SPREAD PLACEHOLDER
# ==========================

np.random.seed(42)
df["MCapGrowth"] = np.random.normal(0, 0.02, len(df)).cumsum()
df["RealizedCapGrowth"] = np.random.normal(0, 0.015, len(df)).cumsum()
df["MCapSpread"] = df["MCapGrowth"] - df["RealizedCapGrowth"]

# ==========================
# DISPLAY WINDOW
# ==========================

display_start = df.index.max() - pd.DateOffset(months=DISPLAY_MONTHS)
display = df.loc[display_start:].copy()

# ==========================
# SPY CHART
# ==========================

spy_fig = go.Figure()

spy_fig.add_trace(go.Scatter(
    x=display.index,
    y=display["SPY"],
    mode="lines",
    name="SPY"
))

for i in range(1, len(display)):
    if display["Regime"].iloc[i] == "Risk-Off":
        spy_fig.add_vrect(
            x0=display.index[i-1],
            x1=display.index[i],
            fillcolor="red",
            opacity=0.08,
            line_width=0
        )

trap = display[display["LiquidityTrap"]]

spy_fig.add_trace(go.Scatter(
    x=trap.index,
    y=trap["SPY"],
    mode="markers",
    name="Liquidity Trap"
))

spy_fig.update_layout(
    template="plotly_dark",
    height=500,
    title="SPY (12M Regime View)"
)

st.plotly_chart(spy_fig, use_container_width=True)

# ==========================
# BTC CHART
# ==========================

btc_fig = go.Figure()

btc_fig.add_trace(go.Scatter(
    x=display.index,
    y=display["BTC"],
    mode="lines",
    name="BTC"
))

for i in range(1, len(display)):
    if display["BTC_Regime"].iloc[i] == "Bear":
        btc_fig.add_vrect(
            x0=display.index[i-1],
            x1=display.index[i],
            fillcolor="orange",
            opacity=0.08,
            line_width=0
        )

btc_fig.update_layout(
    template="plotly_dark",
    height=500,
    title="BTC (12M Regime View)"
)

st.plotly_chart(btc_fig, use_container_width=True)

# ==========================
# SPREAD CHART
# ==========================

spread_fig = go.Figure()

spread_fig.add_trace(go.Scatter(
    x=display.index,
    y=display["MCapSpread"],
    mode="lines",
    name="MCap vs Realized Spread"
))

spread_fig.update_layout(
    template="plotly_dark",
    height=400,
    title="BTC Market Cap vs Realized Cap Growth Spread"
)

st.plotly_chart(spread_fig, use_container_width=True)

# ==========================
# METRICS PANEL
# ==========================

st.markdown("---")

c1, c2, c3 = st.columns(3)

c1.metric("SPY Regime", display["Regime"].iloc[-1])
c2.metric("BTC Regime", display["BTC_Regime"].iloc[-1])
c3.metric("MCap Spread", round(display["MCapSpread"].iloc[-1], 4))
