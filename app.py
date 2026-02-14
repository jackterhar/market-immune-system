import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

############################################
# SAFE LOADER
############################################

def get_data(ticker):
    df = yf.download(ticker, period="5y", auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
    else:
        df = df[["Close"]]

    return df.dropna()

############################################
# DOWNLOAD
############################################

spy = get_data("SPY")
btc = get_data("BTC-USD")
vix = get_data("^VIX")
hyg = get_data("HYG")
lqd = get_data("LQD")
tnx = get_data("^TNX")
irx = get_data("^IRX")

############################################
# MACRO ALIGNMENT
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

spy["ret"] = spy["Close"].pct_change()
spy["vol20"] = spy["ret"].rolling(20).std() * np.sqrt(252)
spy["ma200"] = spy["Close"].rolling(200).mean()
spy = spy.dropna()

macro = macro.reindex(spy.index)

credit_ratio = macro["HYG"] / macro["LQD"]
yield_curve = macro["TNX"] - macro["IRX"]

trend = np.where(spy["Close"] > spy["ma200"], 1, -1)
vol = np.where(spy["vol20"] < spy["vol20"].rolling(252).median(), 1, -1)
credit = np.where(credit_ratio > credit_ratio.rolling(60).mean(), 1, -1)
curve = np.where(yield_curve > 0, 1, -1)

spy_score = trend + vol + credit + curve

############################################
# BTC IMMUNE SYSTEM
############################################

btc["ret"] = btc["Close"].pct_change()
btc["vol30"] = btc["ret"].rolling(30).std() * np.sqrt(365)
btc["ma200"] = btc["Close"].rolling(200).mean()
btc = btc.dropna()

btc_score = (
    np.where(btc["Close"] > btc["ma200"], 1, -1)
    + np.where(btc["vol30"] < btc["vol30"].rolling(365).median(), 1, -1)
)

############################################
# CLASSIFICATION
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

spy_latest = spy_score[-1]
btc_latest = btc_score[-1]

spy_regime, spy_color, spy_exposure = classify(spy_latest)
btc_regime, btc_color, btc_exposure = classify(btc_latest)

############################################
# TRIM TO LAST 24 MONTHS + BIWEEKLY
############################################

cutoff = spy.index[-1] - pd.DateOffset(months=24)

spy_24 = spy[spy.index >= cutoff].resample("2W").last()
btc_24 = btc[btc.index >= cutoff].resample("2W").last()

spy_score_24 = pd.Series(spy_score, index=spy.index)
spy_score_24 = spy_score_24[spy_score_24.index >= cutoff].resample("2W").last()

btc_score_24 = pd.Series(btc_score, index=btc.index)
btc_score_24 = btc_score_24[btc_score_24.index >= cutoff].resample("2W").last()

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
    ax.plot(spy_24.index, spy_24["Close"], linewidth=2)

    for i in range(len(spy_score_24)):
        regime, color, _ = classify(spy_score_24.iloc[i])
        ax.axvspan(spy_24.index[i], spy_24.index[i], color=color, alpha=0.08)

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
    ax2.plot(btc_24.index, btc_24["Close"], linewidth=2)

    for i in range(len(btc_score_24)):
        regime, color, _ = classify(btc_score_24.iloc[i])
        ax2.axvspan(btc_24.index[i], btc_24.index[i], color=color, alpha=0.08)

    st.pyplot(fig2)

############################################
# INTERPRETATION
############################################

st.markdown("---")
st.subheader("How to Interpret")

st.markdown("""
**RISK ON** – Trend, volatility, and macro conditions aligned positively.  
**ACCUMULATION** – Improving conditions but not fully confirmed.  
**NEUTRAL** – Mixed signals. Reduce leverage.  
**DEFENSIVE** – Macro or volatility stress. Capital preservation phase.

Bi-weekly chart shows the last 24 months only for clarity.
""")
