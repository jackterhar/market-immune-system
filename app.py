# ================================
# IMPORTS
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ================================
# CONFIG
# ================================
st.set_page_config(layout="wide")
st.title("🧬 Market Immune System — v13")

# ================================
# DATA
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

    df["SPY_ret"] = df["SPY"].pct_change()
    df["BTC_ret"] = df["BTC"].pct_change()

    df["SPY_50"] = df["SPY"].rolling(50).mean()
    df["SPY_200"] = df["SPY"].rolling(200).mean()

    df["SPY_vol"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    df["BTC_vol"] = df["BTC_ret"].rolling(20).std() * np.sqrt(252)

    return df

df = load_data()

# ================================
# REGIME ENGINE (SPY PRIMARY)
# ================================
def compute_risk_score(row, df):

    score = 0

    if row["SPY"] < row["SPY_200"]:
        score += 1
    if row["SPY"] < row["SPY_50"]:
        score += 1
    if row["SPY_vol"] > df["SPY_vol"].median():
        score += 1
    if row["VIX"] > df["VIX"].median():
        score += 1
    if row["BTC_vol"] > df["BTC_vol"].median():
        score += 1

    return score

df["risk_score"] = df.apply(lambda row: compute_risk_score(row, df), axis=1)

def classify(score):
    if score >= 4:
        return "RISK-OFF"
    elif score >= 2:
        return "CAUTION"
    else:
        return "RISK-ON"

df["regime"] = df["risk_score"].apply(classify)

latest = df.iloc[-1]

# ================================
# COLOR MAP
# ================================
color_map = {
    "RISK-ON": "#2ecc71",
    "CAUTION": "#f1c40f",
    "RISK-OFF": "#e74c3c"
}

regime = latest["regime"]
confidence = round(latest["risk_score"] / 5, 2)

if regime == "RISK-ON":
    exposure = 80
elif regime == "CAUTION":
    exposure = 50
else:
    exposure = 20

# ================================
# REGIME HEADER
# ================================
st.markdown(
    f"""
    ### Current Regime: 
    <span style='color:{color_map[regime]}; font-weight:bold'>
    ● {regime}
    </span>
    """,
    unsafe_allow_html=True
)

# ================================
# METRICS — SPY
# ================================
st.subheader("SPY Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk Score", int(latest["risk_score"]))
c2.metric("Confidence", confidence)
c3.metric("Suggested Exposure", f"{exposure}%")
c4.metric("20d Vol (ann)", f"{latest['SPY_vol']:.2%}")

# ================================
# METRICS — BTC
# ================================
st.subheader("BTC Metrics")

btc_score = 1 if latest["BTC_vol"] > df["BTC_vol"].median() else 0
btc_conf = round(btc_score, 2)
btc_exposure = 70 if btc_score == 0 else 30

b1, b2, b3, b4 = st.columns(4)
b1.metric("Risk Signal", btc_score)
b2.metric("Confidence", btc_conf)
b3.metric("Suggested Exposure", f"{btc_exposure}%")
b4.metric("20d Vol (ann)", f"{latest['BTC_vol']:.2%}")

st.divider()

# ================================
# SHADING FUNCTION
# ================================
def shade_chart(ax, df):
    for i in range(1, len(df)):
        regime = df["regime"].iloc[i]
        ax.axvspan(
            df.index[i-1],
            df.index[i],
            color=color_map[regime],
            alpha=0.06
        )

# ================================
# CHART 1 — SPY
# ================================
st.subheader("SPY with Risk Regime Overlay")

fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(df.index, df["SPY"], linewidth=2)
shade_chart(ax1, df)
ax1.set_title("SPY Price")
st.pyplot(fig1)

# ================================
# CHART 2 — BTC
# ================================
st.subheader("BTC with Risk Regime Overlay")

fig2, ax2 = plt.subplots(figsize=(14,6))
ax2.plot(df.index, df["BTC"], linewidth=2)
shade_chart(ax2, df)
ax2.set_title("BTC Price")
st.pyplot(fig2)

st.divider()

# ================================
# DAILY SUMMARY
# ================================
st.subheader("Daily Summary")

summary = f"""
As of {datetime.today().strftime('%Y-%m-%d')}, the system classifies the market as 
**{regime}** with a confidence score of {confidence}. 

SPY trend, volatility regime, VIX level, and BTC volatility collectively determine 
the current state. Suggested equity exposure is **{exposure}%** based on risk conditions.

Shaded chart regions correspond to:
- Green → Risk-On
- Yellow → Caution
- Red → Risk-Off
"""

st.write(summary)
