# ================================
# IMPORTS (MUST BE FIRST LINES)
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(layout="wide")
st.title("🧬 Market Immune System — v12")

# ================================
# DATA LOADER
# ================================
@st.cache_data(show_spinner=False)
def load_data():

    spy = yf.download("SPY", period="3y", auto_adjust=True)
    btc = yf.download("BTC-USD", period="3y", auto_adjust=True)
    vix = yf.download("^VIX", period="3y", auto_adjust=True)

    df = pd.DataFrame()
    df["SPY"] = spy["Close"]
    df["BTC"] = btc["Close"]
    df["VIX"] = vix["Close"]

    df.dropna(inplace=True)

    # Returns
    df["SPY_ret"] = df["SPY"].pct_change()
    df["BTC_ret"] = df["BTC"].pct_change()

    # Moving averages
    df["SPY_50"] = df["SPY"].rolling(50).mean()
    df["SPY_200"] = df["SPY"].rolling(200).mean()

    # Volatility
    df["SPY_vol"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)

    # BTC stress proxy
    df["BTC_stress"] = df["BTC_ret"].rolling(20).std()

    return df

df = load_data()

# ================================
# REGIME ENGINE (SPY FIRST)
# ================================
latest = df.iloc[-1]

risk_score = 0

# Trend
if latest["SPY"] < latest["SPY_200"]:
    risk_score += 1

if latest["SPY"] < latest["SPY_50"]:
    risk_score += 1

# Volatility
if latest["SPY_vol"] > df["SPY_vol"].median():
    risk_score += 1

# VIX
if latest["VIX"] > df["VIX"].median():
    risk_score += 1

# BTC stress confirmation
if latest["BTC_stress"] > df["BTC_stress"].median():
    risk_score += 1

# ================================
# REGIME CLASSIFICATION
# ================================
if risk_score >= 4:
    regime = "RISK-OFF"
    exposure = 20
elif risk_score >= 2:
    regime = "CAUTION"
    exposure = 50
else:
    regime = "RISK-ON"
    exposure = 80

confidence = round(risk_score / 5, 2)

# ================================
# TOP METRICS
# ================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Regime", regime)
col2.metric("Confidence", confidence)
col3.metric("Suggested Exposure", f"{exposure}%")
col4.metric("SPY 20d Vol (ann)", f"{latest['SPY_vol']:.2%}")

st.divider()

# ================================
# CHART 1 — SPY WITH REGIME BACKDROP
# ================================
st.subheader("SPY with Regime Overlay")

fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(df.index, df["SPY"], label="SPY")

# Shade based on regime score historically
for i in range(len(df)):
    if df["SPY"].iloc[i] < df["SPY_200"].iloc[i]:
        ax1.axvspan(df.index[i], df.index[i], alpha=0.02)

ax1.set_title("SPY Price")
ax1.legend()
st.pyplot(fig1)

# ================================
# CHART 2 — BTC (Complementary Risk Barometer)
# ================================
st.subheader("BTC Risk Proxy")

fig2, ax2 = plt.subplots(figsize=(14,6))
ax2.plot(df.index, df["BTC"], label="BTC")

ax2.set_title("BTC Price")
ax2.legend()
st.pyplot(fig2)

st.divider()

# ================================
# WHY TODAY SECTION
# ================================
st.subheader("Why Today?")

reasons = []

if latest["SPY"] < latest["SPY_200"]:
    reasons.append("SPY below 200-day moving average (macro trend weak).")

if latest["SPY"] < latest["SPY_50"]:
    reasons.append("SPY below 50-day moving average (short-term weakness).")

if latest["SPY_vol"] > df["SPY_vol"].median():
    reasons.append("Volatility elevated vs median.")

if latest["VIX"] > df["VIX"].median():
    reasons.append("VIX elevated vs median.")

if latest["BTC_stress"] > df["BTC_stress"].median():
    reasons.append("BTC volatility elevated (cross-asset stress).")

if len(reasons) == 0:
    st.write("Conditions supportive of risk appetite.")
else:
    for r in reasons:
        st.write("- " + r)

# ================================
# DAILY SUMMARY
# ================================
st.subheader("Daily Summary")

summary = f"""
As of {datetime.today().strftime('%Y-%m-%d')}, the model classifies the market as **{regime}**
with a confidence score of {confidence}. The system is primarily driven by SPY trend,
volatility regime, and cross-asset confirmation via BTC and VIX.
Suggested equity exposure is **{exposure}%** based on current risk conditions.
"""

st.write(summary)
