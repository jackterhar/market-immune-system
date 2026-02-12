# ==========================================
# MARKET IMMUNE SYSTEM — v15
# Independent SPY + BTC Regime Engines
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System — Independent Regimes")

# ==========================================
# SAFE DOWNLOAD
# ==========================================

def get_close(ticker, period="3y"):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    if "Close" in df.columns:
        return df["Close"]

    return df.iloc[:, 0]


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data(show_spinner=False)
def load_data():

    spy = get_close("SPY")
    btc = get_close("BTC-USD")
    vix = get_close("^VIX")
    hyg = get_close("HYG")
    ief = get_close("IEF")
    t10 = get_close("^TNX")
    t2 = get_close("^IRX")

    df = pd.DataFrame(index=spy.index)

    df["SPY"] = spy
    df["BTC"] = btc.reindex(df.index)
    df["VIX"] = vix.reindex(df.index)
    df["HYG"] = hyg.reindex(df.index)
    df["IEF"] = ief.reindex(df.index)
    df["10Y"] = t10.reindex(df.index)
    df["2Y"] = t2.reindex(df.index)

    df.dropna(inplace=True)

    # Credit ratio
    df["Credit_Ratio"] = df["HYG"] / df["IEF"]

    # Yield curve
    df["Curve"] = df["10Y"] - df["2Y"]

    # Returns
    df["SPY_ret"] = df["SPY"].pct_change()
    df["BTC_ret"] = df["BTC"].pct_change()

    # MAs
    df["SPY_50"] = df["SPY"].rolling(50).mean()
    df["SPY_200"] = df["SPY"].rolling(200).mean()
    df["BTC_50"] = df["BTC"].rolling(50).mean()
    df["BTC_200"] = df["BTC"].rolling(200).mean()

    # Vol
    df["SPY_vol"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    df["BTC_vol"] = df["BTC_ret"].rolling(20).std() * np.sqrt(252)

    df["SPY_vol_med"] = df["SPY_vol"].rolling(252).median()
    df["BTC_vol_med"] = df["BTC_vol"].rolling(252).median()

    return df


df = load_data()

if df.empty:
    st.error("Data failed to load.")
    st.stop()

# ==========================================
# SPY REGIME ENGINE (MACRO INTEGRATED)
# ==========================================

def score_spy(row):

    score = 0

    # Trend
    if row["SPY"] < row["SPY_200"]:
        score += 1
    if row["SPY_50"] < row["SPY_200"]:
        score += 1

    # Volatility
    if row["SPY_vol"] > row["SPY_vol_med"]:
        score += 1

    # Credit deterioration
    if row["Credit_Ratio"] < df["Credit_Ratio"].rolling(60).mean().iloc[-1]:
        score += 1

    # Yield curve inversion
    if row["Curve"] < 0:
        score += 1

    # VIX stress
    if row["VIX"] > 25:
        score += 1

    return score  # max 6


df["SPY_score"] = df.apply(score_spy, axis=1)


def classify_spy(score):
    if score <= 2:
        return "RISK-ON"
    elif score <= 4:
        return "CAUTION"
    else:
        return "RISK-OFF"


df["SPY_regime"] = df["SPY_score"].apply(classify_spy)

# ==========================================
# BTC REGIME ENGINE (SEPARATE)
# ==========================================

def score_btc(row):

    score = 0

    # Trend
    if row["BTC"] < row["BTC_200"]:
        score += 1
    if row["BTC_50"] < row["BTC_200"]:
        score += 1

    # Volatility
    if row["BTC_vol"] > row["BTC_vol_med"]:
        score += 1

    # Risk spillover
    if row["VIX"] > 25:
        score += 1

    return score  # max 4


df["BTC_score"] = df.apply(score_btc, axis=1)


def classify_btc(score):
    if score <= 1:
        return "RISK-ON"
    elif score <= 2:
        return "CAUTION"
    else:
        return "RISK-OFF"


df["BTC_regime"] = df["BTC_score"].apply(classify_btc)

# ==========================================
# LATEST VALUES
# ==========================================

latest = df.iloc[-1]

# ==========================================
# DISPLAY REGIMES
# ==========================================

col1, col2 = st.columns(2)

col1.metric("SPY Regime", latest["SPY_regime"])
col2.metric("BTC Regime", latest["BTC_regime"])

st.write(f"SPY Score: {latest['SPY_score']} / 6")
st.write(f"BTC Score: {latest['BTC_score']} / 4")

st.divider()

# ==========================================
# COLOR MAP
# ==========================================

color_map = {
    "RISK-ON": "green",
    "CAUTION": "yellow",
    "RISK-OFF": "red"
}

# ==========================================
# SHADING FUNCTION
# ==========================================

def shade(ax, regime_col):
    for i in range(1, len(df)):
        regime = df.iloc[i][regime_col]
        ax.axvspan(
            df.index[i-1],
            df.index[i],
            color=color_map[regime],
            alpha=0.08
        )

# ==========================================
# SPY CHART
# ==========================================

st.subheader("SPY — Independent Regime")

fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(df.index, df["SPY"], linewidth=2)
shade(ax1, "SPY_regime")
st.pyplot(fig1)

# ==========================================
# BTC CHART
# ==========================================

st.subheader("BTC — Independent Regime")

fig2, ax2 = plt.subplots(figsize=(14,6))
ax2.plot(df.index, df["BTC"], linewidth=2)
shade(ax2, "BTC_regime")
st.pyplot(fig2)

st.divider()

# ==========================================
# SUMMARY
# ==========================================

st.subheader("Summary")

st.write(f"""
As of {datetime.today().strftime('%Y-%m-%d')}:

SPY Regime: **{latest['SPY_regime']}**  
BTC Regime: **{latest['BTC_regime']}**

SPY integrates macro structure (credit, yield curve, volatility, VIX).  
BTC reflects crypto-specific trend + volatility with risk spillover.

These regimes are now fully independent.
""")
