# ==========================================
# MARKET IMMUNE SYSTEM — v14.2 (ROBUST)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System — v14.2 (Macro Integrated Robust)")

# ==========================================
# SAFE DOWNLOAD FUNCTION
# ==========================================

def get_close(ticker, period="3y"):

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Flatten multi-index if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # If Close exists
    if "Close" in df.columns:
        return df["Close"]

    # If only one column exists, use it
    if len(df.columns) == 1:
        return df.iloc[:, 0]

    # Otherwise fallback safely
    return df.iloc[:, 0]


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data(show_spinner=False)
def load_data():

    spy = get_close("SPY")
    btc = get_close("BTC-USD")
    vix = get_close("^VIX")
    t10 = get_close("^TNX")
    t2 = get_close("^IRX")
    hyg = get_close("HYG")
    ief = get_close("IEF")

    # Align everything to SPY index
    df = pd.DataFrame(index=spy.index)

    df["SPY"] = spy
    df["BTC"] = btc.reindex(df.index)
    df["VIX"] = vix.reindex(df.index)
    df["10Y"] = t10.reindex(df.index)
    df["2Y_proxy"] = t2.reindex(df.index)

    df["Credit_Ratio"] = (
        hyg.reindex(df.index) / ief.reindex(df.index)
    )

    df.dropna(inplace=True)

    # Returns
    df["SPY_ret"] = df["SPY"].pct_change()
    df["BTC_ret"] = df["BTC"].pct_change()

    # Trend
    df["SPY_50"] = df["SPY"].rolling(50).mean()
    df["SPY_200"] = df["SPY"].rolling(200).mean()

    # Vol
    df["SPY_vol"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    df["BTC_vol"] = df["BTC_ret"].rolling(20).std() * np.sqrt(252)

    # Yield curve
    df["Curve"] = df["10Y"] - df["2Y_proxy"]

    return df


df = load_data()

if df.empty:
    st.error("Data failed to load. Likely Yahoo rate limit. Refresh in 60 seconds.")
    st.stop()

# ==========================================
# RISK ENGINE
# ==========================================

def compute_score(row):

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

    if row["Curve"] < 0:
        score += 1

    if row["Credit_Ratio"] < df["Credit_Ratio"].rolling(60).mean().iloc[-1]:
        score += 1

    return score


df["risk_score"] = df.apply(compute_score, axis=1)

def classify(score):
    if score >= 5:
        return "RISK-OFF"
    elif score >= 3:
        return "CAUTION"
    else:
        return "RISK-ON"

df["regime"] = df["risk_score"].apply(classify)

latest = df.iloc[-1]

confidence = round(latest["risk_score"] / 7, 2)

if latest["regime"] == "RISK-ON":
    exposure = 80
elif latest["regime"] == "CAUTION":
    exposure = 50
else:
    exposure = 20

# ==========================================
# COLORS
# ==========================================

color_map = {
    "RISK-ON": "#2ecc71",
    "CAUTION": "#f1c40f",
    "RISK-OFF": "#e74c3c"
}

# ==========================================
# HEADER
# ==========================================

st.markdown(
    f"""
    ### Current Regime:
    <span style='color:{color_map[latest["regime"]]}; font-weight:bold'>
    ● {latest["regime"]}
    </span>
    """,
    unsafe_allow_html=True
)

# ==========================================
# METRICS
# ==========================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk Score", int(latest["risk_score"]))
c2.metric("Confidence", confidence)
c3.metric("Suggested Exposure", f"{exposure}%")
c4.metric("SPY 20d Vol", f"{latest['SPY_vol']:.2%}")

m1, m2, m3 = st.columns(3)
m1.metric("Yield Curve (10Y-2Y)", f"{latest['Curve']:.2f}")
m2.metric("Credit Ratio (HYG/IEF)", f"{latest['Credit_Ratio']:.2f}")
m3.metric("VIX", f"{latest['VIX']:.2f}")

st.divider()

# ==========================================
# SHADING
# ==========================================

def shade(ax):
    for i in range(1, len(df)):
        ax.axvspan(
            df.index[i-1],
            df.index[i],
            color=color_map[df["regime"].iloc[i]],
            alpha=0.06
        )

# ==========================================
# SPY CHART
# ==========================================

st.subheader("SPY with Regime Overlay")

fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(df.index, df["SPY"], linewidth=2)
shade(ax1)
st.pyplot(fig1)

# ==========================================
# BTC CHART
# ==========================================

st.subheader("BTC with Regime Overlay")

fig2, ax2 = plt.subplots(figsize=(14,6))
ax2.plot(df.index, df["BTC"], linewidth=2)
shade(ax2)
st.pyplot(fig2)

st.divider()

# ==========================================
# DAILY SUMMARY
# ==========================================

summary = f"""
As of {datetime.today().strftime('%Y-%m-%d')}, the system classifies the market as 
**{latest['regime']}** with confidence {confidence}.

Suggested exposure: **{exposure}%**.

Risk score integrates:
• Trend structure  
• Volatility regime  
• VIX stress  
• BTC volatility  
• Yield curve slope  
• Credit conditions  
"""

st.write(summary)
