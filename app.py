import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")

############################################
# Helper Functions
############################################

def zscore(series, window=252):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def compute_regime(score):
    if score >= 1:
        return "RISK ON", "green", 100
    elif score >= 0.25:
        return "ACCUMULATION", "gold", 70
    elif score >= -0.25:
        return "NEUTRAL", "orange", 50
    else:
        return "DEFENSIVE", "red", 20

############################################
# DATA DOWNLOAD
############################################

spy = yf.download("SPY", start="2010-01-01")
btc = yf.download("BTC-USD", start="2015-01-01")
vix = yf.download("^VIX", start="2010-01-01")
hyg = yf.download("HYG", start="2010-01-01")
lqd = yf.download("LQD", start="2010-01-01")
dgs10 = yf.download("^TNX", start="2010-01-01")
dgs3m = yf.download("^IRX", start="2010-01-01")

############################################
# SPY IMMUNE SYSTEM
############################################

spy['returns'] = spy['Close'].pct_change()
spy['vol20'] = spy['returns'].rolling(20).std() * np.sqrt(252)
spy['ma200'] = spy['Close'].rolling(200).mean()
spy['trend'] = spy['Close'] > spy['ma200']

credit_ratio = hyg['Close'] / lqd['Close']
yield_curve = dgs10['Close'] - dgs3m['Close']

spy_score = (
    zscore(spy['trend'].astype(int)) +
    -zscore(spy['vol20']) +
    zscore(credit_ratio) +
    zscore(yield_curve)
)

spy_latest_score = spy_score.dropna().iloc[-1]
spy_regime, spy_color, spy_exposure = compute_regime(spy_latest_score)
spy_confidence = round(abs(spy_latest_score) * 25, 1)

############################################
# BTC IMMUNE SYSTEM
############################################

btc['returns'] = btc['Close'].pct_change()
btc['vol30'] = btc['returns'].rolling(30).std() * np.sqrt(365)
btc['ma200'] = btc['Close'].rolling(200).mean()
btc['trend'] = btc['Close'] > btc['ma200']

btc_score = (
    zscore(btc['trend'].astype(int)) +
    -zscore(btc['vol30'])
)

btc_latest_score = btc_score.dropna().iloc[-1]
btc_regime, btc_color, btc_exposure = compute_regime(btc_latest_score)
btc_confidence = round(abs(btc_latest_score) * 25, 1)

############################################
# DASHBOARD DISPLAY
############################################

st.title("Market Immune System")

col1, col2 = st.columns(2)

############################################
# SPY PANEL
############################################

with col1:
    st.markdown("## SPY Immune System")

    st.markdown(f"**Regime:** :{spy_color}[{spy_regime}]")
    st.markdown(f"**Confidence:** {spy_confidence}%")
    st.markdown(f"**Suggested Exposure:** {spy_exposure}%")
    st.markdown(f"**VIX:** {round(vix['Close'].iloc[-1],2)}")
    st.markdown(f"**Credit Ratio (HYG/LQD):** {round(credit_ratio.iloc[-1],3)}")
    st.markdown(f"**Yield Curve (10Y-3M):** {round(yield_curve.iloc[-1],2)}")
    st.markdown(f"**SPY 20D Vol:** {round(spy['vol20'].iloc[-1]*100,2)}%")

    fig, ax = plt.subplots()
    ax.plot(spy['Close'], label="SPY")
    
    for i in range(len(spy_score)):
        if spy_score.iloc[i] >= 1:
            ax.axvspan(spy.index[i], spy.index[i], color='green', alpha=0.02)
        elif spy_score.iloc[i] < -0.25:
            ax.axvspan(spy.index[i], spy.index[i], color='red', alpha=0.02)

    ax.set_title("SPY Price with Regime Shading")
    st.pyplot(fig)

############################################
# BTC PANEL
############################################

with col2:
    st.markdown("## BTC Immune System")

    st.markdown(f"**Regime:** :{btc_color}[{btc_regime}]")
    st.markdown(f"**Confidence:** {btc_confidence}%")
    st.markdown(f"**Suggested Exposure:** {btc_exposure}%")
    st.markdown(f"**BTC 30D Vol:** {round(btc['vol30'].iloc[-1]*100,2)}%")

    fig2, ax2 = plt.subplots()
    ax2.plot(btc['Close'], label="BTC")

    for i in range(len(btc_score)):
        if btc_score.iloc[i] >= 1:
            ax2.axvspan(btc.index[i], btc.index[i], color='green', alpha=0.02)
        elif btc_score.iloc[i] < -0.25:
            ax2.axvspan(btc.index[i], btc.index[i], color='red', alpha=0.02)

    ax2.set_title("BTC Price with Regime Shading")
    st.pyplot(fig2)

############################################
# INTERPRETATION SECTION
############################################

st.markdown("---")
st.markdown("## How to Interpret the Immune System")

st.markdown("""
### Regime
- **Risk On**: Strong macro + trend + volatility alignment. Favor aggressive exposure.
- **Accumulation**: Improving conditions but not full confirmation.
- **Neutral**: Mixed signals. Reduce sizing.
- **Defensive**: Elevated risk. Capital preservation prioritized.

### Confidence
Derived from magnitude of composite z-score. Higher = stronger signal alignment.

### Suggested Exposure
Systematic allocation guidance based on regime strength.

### VIX (SPY only)
Measures equity fear. Rising VIX = stress environment.

### Credit Ratio (HYG/LQD)
High yield vs investment grade. Rising ratio = risk appetite.

### Yield Curve (10Y–3M)
Positive = expansionary backdrop. Inversion = recession risk.

### SPY 20D Vol
Short-term realized volatility of SPY.

### BTC 30D Vol
Realized volatility specific to BTC market structure.

### Trend (200D MA)
Primary structural regime filter. Above = expansion phase.
""")
