import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# =========================================================
# LOAD (SPY-ANCHORED)
# =========================================================

@st.cache_data
def load_data():

    tickers = [
        "SPY",
        "BTC-USD",
        "^VIX",
        "^TNX",
        "DX-Y.NYB",
        "HYG",
        "IEF"
    ]

    raw = yf.download(
        tickers,
        start="2022-01-01",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    closes = raw["Close"]

    closes = closes.dropna(subset=["SPY"])   # Anchor to SPY trading days
    closes = closes.ffill()                  # Fill BTC weekends

    return closes


df = load_data()

df.columns = [
    "SPY",
    "BTC",
    "VIX",
    "TNX",
    "DXY",
    "HYG",
    "IEF"
]

data = df.copy()
ret = data.pct_change()

# =========================================================
# FACTORS (unchanged logic)
# =========================================================

# Trend
data["MA50"] = data["SPY"].rolling(50).mean()
data["MA200"] = data["SPY"].rolling(200).mean()
trend_raw = (data["MA50"] - data["MA200"]) / data["SPY"]
trend_z = (trend_raw - trend_raw.rolling(252).mean()) / trend_raw.rolling(252).std()

# Volatility
vol_20 = ret["SPY"].rolling(20).std()
vol_z = (vol_20 - vol_20.rolling(252).mean()) / vol_20.rolling(252).std()
vix_z = (data["VIX"] - data["VIX"].rolling(252).mean()) / data["VIX"].rolling(252).std()
vol_comp = -(vol_z + vix_z) / 2

# Credit
credit_ratio = data["HYG"] / data["IEF"]
credit_z = (credit_ratio - credit_ratio.rolling(252).mean()) / credit_ratio.rolling(252).std()

# Liquidity
rates_z = (data["TNX"] - data["TNX"].rolling(252).mean()) / data["TNX"].rolling(252).std()
dxy_z = (data["DXY"] - data["DXY"].rolling(252).mean()) / data["DXY"].rolling(252).std()
liq_comp = -(rates_z + dxy_z) / 2

# BTC stress (complementary)
btc_trend = data["BTC"].rolling(50).mean() - data["BTC"].rolling(200).mean()
btc_trend_z = (btc_trend - btc_trend.rolling(252).mean()) / btc_trend.rolling(252).std()

btc_vol = ret["BTC"].rolling(20).std()
btc_vol_z = (btc_vol - btc_vol.rolling(252).mean()) / btc_vol.rolling(252).std()

btc_comp = btc_trend_z - btc_vol_z

# =========================================================
# COMPOSITE
# =========================================================

composite = (
    0.25*trend_z +
    0.20*vol_comp +
    0.20*credit_z +
    0.20*liq_comp +
    0.15*btc_comp
)

composite = composite.dropna()
data = data.loc[composite.index]

# =========================================================
# REGIME SERIES
# =========================================================

regime_series = pd.Series(index=composite.index)

regime_series[composite > 0.75] = 1
regime_series[composite < -0.75] = -1
regime_series[(composite >= -0.75) & (composite <= 0.75)] = 0

latest_score = float(composite.iloc[-1])
latest_regime = regime_series.iloc[-1]

if latest_regime == 1:
    regime = "RISK-ON"
elif latest_regime == -1:
    regime = "RISK-OFF"
else:
    regime = "CAUTION"

# =========================================================
# REGIME BLOCK SHADING FUNCTION
# =========================================================

def add_regime_blocks(ax, regime_series):
    start = regime_series.index[0]
    current = regime_series.iloc[0]

    for i in range(1, len(regime_series)):
        if regime_series.iloc[i] != current:
            end = regime_series.index[i]
            color = (
                "green" if current == 1
                else "red" if current == -1
                else "yellow"
            )
            ax.axvspan(start, end, color=color, alpha=0.12)
            start = end
            current = regime_series.iloc[i]

    # final block
    end = regime_series.index[-1]
    color = (
        "green" if current == 1
        else "red" if current == -1
        else "yellow"
    )
    ax.axvspan(start, end, color=color, alpha=0.12)

# =========================================================
# CHART 1 — SPY (PRIMARY)
# =========================================================

st.subheader("SPY with Regime Overlay")

fig1, ax1 = plt.subplots(figsize=(15,6))

ax1.plot(data.index, data["SPY"], linewidth=2)
add_regime_blocks(ax1, regime_series)

ax2 = ax1.twinx()
ax2.plot(composite.index, composite, linestyle="--", alpha=0.6)

ax1.set_title("SPY (Primary Regime Structure)")
st.pyplot(fig1)

# =========================================================
# CHART 2 — BTC (COMPLEMENTARY)
# =========================================================

st.subheader("BTC with Same Regime Overlay")

fig2, ax3 = plt.subplots(figsize=(15,6))

ax3.plot(data.index, data["BTC"], linewidth=2)
add_regime_blocks(ax3, regime_series)

ax4 = ax3.twinx()
ax4.plot(btc_comp.index, btc_comp, linestyle="--", alpha=0.6)

ax3.set_title("BTC (Crypto Stress Context)")
st.pyplot(fig2)
