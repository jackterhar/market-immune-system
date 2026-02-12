# ==========================================
# MARKET IMMUNE SYSTEM — Independent SPY + BTC
# Full Scoring Above Each Chart
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System")

# ==========================================
# DATA DOWNLOAD
# ==========================================

def get_close(ticker, period="3y"):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    return df["Close"]


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

    # Macro metrics
    df["Credit_Ratio"] = df["HYG"] / df["IEF"]
    df["Curve"] = df["10Y"] - df["2Y"]

    # Returns
    df["SPY_ret"] = df["SPY"].pct_change()
    df["BTC_ret"] = df["BTC"].pct_change()

    # MAs
    df["SPY_50"] = df["SPY"].rolling(50).mean()
    df["SPY_200"] = df["SPY"].rolling(200).mean()
    df["BTC_50"] = df["BTC"].rolling(50).mean()
    df["BTC_200"] = df["BTC"].rolling(200).mean()

    # Vol (20d annualized)
    df["SPY_vol"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    df["BTC_vol"] = df["BTC_ret"].rolling(20).std() * np.sqrt(252)

    return df


df = load_data()

if df.empty:
    st.error("Data failed.")
    st.stop()

# ==========================================
# SCORING ENGINES
# ==========================================

def score_spy(row):

    score = 0

    # Trend (2)
    if row["SPY"] < row["SPY_200"]:
        score += 1
    if row["SPY_50"] < row["SPY_200"]:
        score += 1

    # Vol (1)
    if row["SPY_vol"] > df["SPY_vol"].rolling(252).median().iloc[-1]:
        score += 1

    # Credit (1)
    if row["Credit_Ratio"] < df["Credit_Ratio"].rolling(60).mean().iloc[-1]:
        score += 1

    # Curve (1)
    if row["Curve"] < 0:
        score += 1

    # VIX (1)
    if row["VIX"] > 25:
        score += 1

    return score  # max 6


def score_btc(row):

    score = 0

    # Trend (2)
    if row["BTC"] < row["BTC_200"]:
        score += 1
    if row["BTC_50"] < row["BTC_200"]:
        score += 1

    # Vol (1)
    if row["BTC_vol"] > df["BTC_vol"].rolling(252).median().iloc[-1]:
        score += 1

    return score  # max 3


df["SPY_score"] = df.apply(score_spy, axis=1)
df["BTC_score"] = df.apply(score_btc, axis=1)

# ==========================================
# CLASSIFICATION
# ==========================================

def classify_spy(score):
    if score <= 2:
        return "RISK-ON"
    elif score <= 4:
        return "CAUTION"
    else:
        return "RISK-OFF"


def classify_btc(score):
    if score <= 1:
        return "RISK-ON"
    elif score == 2:
        return "CAUTION"
    else:
        return "RISK-OFF"


df["SPY_regime"] = df["SPY_score"].apply(classify_spy)
df["BTC_regime"] = df["BTC_score"].apply(classify_btc)

latest = df.iloc[-1]

# ==========================================
# EXPOSURE + CONFIDENCE
# ==========================================

def exposure_spy(regime):
    return {"RISK-ON": 100, "CAUTION": 60, "RISK-OFF": 20}[regime]


def exposure_btc(regime):
    return {"RISK-ON": 100, "CAUTION": 50, "RISK-OFF": 10}[regime]


def confidence(score, max_score):
    return round((score / max_score) * 100, 1)


color_map = {
    "RISK-ON": "green",
    "CAUTION": "orange",
    "RISK-OFF": "red"
}

# ==========================================
# SPY SECTION
# ==========================================

st.subheader("SPY Regime")

spy_regime = latest["SPY_regime"]
spy_score = latest["SPY_score"]

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"### :{color_map[spy_regime]}[{spy_regime}]")
col2.metric("Confidence", f"{confidence(spy_score,6)}%")
col3.metric("Suggested Exposure", f"{exposure_spy(spy_regime)}%")
col4.metric("VIX", round(latest["VIX"],2))

col5, col6, col7 = st.columns(3)
col5.metric("Credit Ratio", round(latest["Credit_Ratio"],3))
col6.metric("Yield Curve", round(latest["Curve"],2))
col7.metric("SPY 20d Vol", f"{round(latest['SPY_vol']*100,1)}%")

# Chart
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(df.index, df["SPY"], linewidth=2)

for i in range(1, len(df)):
    ax1.axvspan(
        df.index[i-1],
        df.index[i],
        color=color_map[df.iloc[i]["SPY_regime"]],
        alpha=0.08
    )

st.pyplot(fig1)

st.divider()

# ==========================================
# BTC SECTION
# ==========================================

st.subheader("BTC Regime")

btc_regime = latest["BTC_regime"]
btc_score = latest["BTC_score"]

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"### :{color_map[btc_regime]}[{btc_regime}]")
col2.metric("Confidence", f"{confidence(btc_score,3)}%")
col3.metric("Suggested Exposure", f"{exposure_btc(btc_regime)}%")
col4.metric("BTC 20d Vol", f"{round(latest['BTC_vol']*100,1)}%")

# Chart
fig2, ax2 = plt.subplots(figsize=(14,6))
ax2.plot(df.index, df["BTC"], linewidth=2)

for i in range(1, len(df)):
    ax2.axvspan(
        df.index[i-1],
        df.index[i],
        color=color_map[df.iloc[i]["BTC_regime"]],
        alpha=0.08
    )

st.pyplot(fig2)

st.caption(f"Last Updated: {datetime.today().strftime('%Y-%m-%d')}")
