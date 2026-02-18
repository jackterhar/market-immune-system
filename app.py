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

def download_series(ticker):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        st.error(f"No data for {ticker}")
        st.stop()
    if "Close" not in df.columns:
        st.error(f"{ticker} missing Close column")
        st.stop()
    return df["Close"]

def resample_series(series):
    return series.resample(RESAMPLE_RULE).last().dropna()

# ==========================
# DOWNLOAD DATA
# ==========================

spy = resample_series(download_series("SPY"))
btc = resample_series(download_series("BTC-USD"))
hyg = resample_series(download_series("HYG"))
lqd = resample_series(download_series("LQD"))

# Align index safely
common_index = spy.index.intersection(btc.index)
common_index = common_index.intersection(hyg.index)
common_index = common_index.intersection(lqd.index)

spy = spy.reindex(common_index)
btc = btc.reindex(common_index)
hyg = hyg.reindex(common_index)
lqd = lqd.reindex(common_index)

# ==========================
# BUILD DATAFRAME
# ==========================

df = pd.DataFrame(index=common_index)
df["SPY"] = spy
df["BTC"] = btc
df["CreditRatio"] = hyg / lqd

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
# COMPOSITE SCORE
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

def classify_btc(trend):
    return "Bull" if trend == 1 else "Bear"

df["BTC_Regime"] = df["BTC_Trend"].apply(classify_btc)

# ==========================
# LIQUIDITY TRAP DETECTOR
# ==========================

df["SPY_RSI"] = (
    df["SPY"].diff()
    .pipe(lambda x: x.where(x > 0, 0).rolling(14).mean()) /
    df["SPY"].diff().abs().rolling(14).mean()
) * 100

df["LiquidityTrap"] = (
    (df["Regime"] == "Risk-Off") &
    (df["SPY_RSI"] < 30)
)

# ==========================
# CRYPTOQUANT PLACEHOLDER
# ==========================

# Replace this with real API pull later
np.random.seed(42)
df["MCapGrowth"] = np.random.normal(0, 0.02, len(df)).cumsum()
df["RealizedCapGrowth"] = np.random.normal(0, 0.015, len(df)).cumsum()

df["MCapSpread"] = df["MCapGrowth"] - df["RealizedCapGrowth"]

# ==========================
# DISPLAY WINDOW
# ==========================

display_start = df.index.max() - pd.DateOffset(months=DISPLAY_MONTHS)
display = df.loc[display_start:]

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

# Regime shading
for i in range(1, len(display)):
    if display["Regime"].iloc[i] == "Risk-Off":
        spy_fig.add_vrect(
            x0=display.index[i-1],
            x1=display.index[i],
            fillcolor="red",
            opacity=0.08,
            line_width=0
        )

# Liquidity trap markers
trap_points = display[display["LiquidityTrap"]]

spy_fig.add_trace(go.Scatter(
    x=trap_points.index,
    y=trap_points["SPY"],
    mode="markers",
    marker=dict(size=8),
    name="Liquidity Trap"
))

spy_fig.update_layout(
    template="plotly_dark",
    height=500,
    title="SPY (12 Month Regime View)"
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
    title="BTC (12 Month Regime View)"
)

st.plotly_chart(btc_fig, use_container_width=True)

# ==========================
# MCap vs Realized Cap Spread
# ==========================

spread_fig = go.Figure()

spread_fig.add_trace(go.Scatter(
    x=display.index,
    y=display["MCapSpread"],
    mode="lines",
    name="Market Cap vs Realized Cap Spread"
))

spread_fig.update_layout(
    template="plotly_dark",
    height=400,
    title="BTC Market Cap vs Realized Cap Growth Spread"
)

st.plotly_chart(spread_fig, use_container_width=True)

# ==========================
# SUMMARY PANEL
# ==========================

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Current SPY Regime", display["Regime"].iloc[-1])
col2.metric("Current BTC Regime", display["BTC_Regime"].iloc[-1])
col3.metric("MCap Spread", round(display["MCapSpread"].iloc[-1], 4))
