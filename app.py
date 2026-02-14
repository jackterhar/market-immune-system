import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

############################################
# SAFE DATA LOADER
############################################

def get_data(ticker):
    df = yf.download(ticker, start="2010-01-01", auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
    else:
        df = df[["Close"]]

    df = df.dropna()
    return df

############################################
# DOWNLOAD DATA
############################################

spy = get_data("SPY")
btc = get_data("BTC-USD")
vix = get_data("^VIX")
hyg = get_data("HYG")
lqd = get_data("LQD")
tnx = get_data("^TNX")
irx = get_data("^IRX")

############################################
# ALIGN MACRO DATA TO SPY INDEX
############################################

macro = pd.concat([
    vix["Close"],
    hyg["Close"],
    lqd["Close"],
    tnx["Close"],
    irx["Close"]
], axis=1)

macro.columns = ["VIX","HYG","LQD","TNX","IRX"]
macro = macro.reindex(spy.index).ffill()

############################################
# SPY IMMUNE SYSTEM
############################################

spy["returns"] = spy["Close"].pct_change()
spy["vol20"] = spy["returns"].rolling(20).std() * np.sqrt(252)
spy["ma200"] = spy["Close"].rolling(200).mean()

spy = spy.dropna()

macro = macro.reindex(spy.index)

credit_ratio = macro["HYG"] / macro["LQD"]
yield_curve = macro["TNX"] - macro["IRX"]

# Score components
trend_score = np.where(spy["Close"] > spy["ma200"], 1, -1)
vol_score = np.where(spy["vol20"] < spy["vol20"].rolling(252).median(), 1, -1)
credit_score = np.where(credit_ratio > credit_ratio.rolling(60).mean(), 1, -1)
curve_score = np.where(yield_curve > 0, 1, -1)

spy_score = trend_score + vol_score + credit_score + curve_score

############################################
# BTC IMMUNE SYSTEM
############################################

btc["returns"] = btc["Close"].pct_change()
btc["vol30"] = btc["returns"].rolling(30).std() * np.sqrt(365)
btc["ma200"] = btc["Close"].rolling(200).mean()
btc = btc.dropna()

btc_score = (
    np.where(btc["Close"] > btc["ma200"], 1, -1)
    + np.where(btc["vol30"] < btc["vol30"].rolling(365).median(), 1, -1)
)

############################################
# REGIME CLASSIFIER
############################################

def classify(score):
    if score >= 3:
        return "RISK ON", "green", 100
    elif score >= 1:
        return "ACCUMULATION", "gold", 70
    elif score >= -1:
        return "NEUTRAL", "orange", 50
    else:
        return "DEFENSIVE", "red", 20

############################################
# LATEST VALUES
############################################

spy_latest = spy_score[-1]
btc_latest = btc_score[-1]

spy_regime, spy_color, spy_exposure = classify(spy_latest)
btc_regime, btc_color, btc_exposure = classify(btc_latest)

############################################
# DASHBOARD
############################################

st.title("🧬 Market Immune System")

col1, col2 = st.columns(2)

############################################
# SPY PANEL
############################################

with col1:
    st.subheader("SPY Immune System")

    st.markdown(f"**Regime:** :{spy_color}[{spy_regime}]")
    st.markdown(f"**Score:** {spy_latest}")
    st.markdown(f"**Suggested Exposure:** {spy_exposure}%")
    st.markdown(f"**VIX:** {round(macro['VIX'][-1],2)}")
    st.markdown(f"**Credit Ratio:** {round(credit_ratio[-1],3)}")
    st.markdown(f"**Yield Curve:** {round(yield_curve[-1],2)}")
    st.markdown(f"**SPY 20D Vol:** {round(spy['vol20'][-1]*100,2)}%")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(spy.index, spy["Close"], linewidth=1.5)

    for i in range(len(spy_score)):
        if spy_score[i] >= 3:
            ax.axvspan(spy.index[i], spy.index[i], color='green', alpha=0.03)
        elif spy_score[i] <= -2:
            ax.axvspan(spy.index[i], spy.index[i], color='red', alpha=0.03)

    st.pyplot(fig)

############################################
# BTC PANEL
############################################

with col2:
    st.subheader("BTC Immune System")

    st.markdown(f"**Regime:** :{btc_color}[{btc_regime}]")
    st.markdown(f"**Score:** {btc_latest}")
    st.markdown(f"**Suggested Exposure:** {btc_exposure}%")
    st.markdown(f"**BTC 30D Vol:** {round(btc['vol30'][-1]*100,2)}%")

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(btc.index, btc["Close"], linewidth=1.5)

    for i in range(len(btc_score)):
        if btc_score[i] >= 2:
            ax2.axvspan(btc.index[i], btc.index[i], color='green', alpha=0.03)
        elif btc_score[i] <= -1:
            ax2.axvspan(btc.index[i], btc.index[i], color='red', alpha=0.03)

    st.pyplot(fig2)

############################################
# INTERPRETATION
############################################

st.markdown("---")
st.subheader("How to Interpret the Immune System")

st.markdown("""
**Trend (200D MA):** Determines structural bull vs bear regime.  
**Volatility:** Rising realized vol indicates stress conditions.  
**Credit Ratio (HYG/LQD):** Risk appetite gauge. Rising = expansion.  
**Yield Curve (10Y–3M):** Inversion signals tightening / recession risk.  
**VIX:** Equity fear gauge. Elevated VIX confirms risk-off.  
**Score:** Composite of all components.  
**Suggested Exposure:** Capital allocation guidance based on total alignment.  
""")
