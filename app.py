import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Market Immune System v4")

# =========================================================
# CONFIG
# =========================================================

SPX_TICKER = "^GSPC"

AI_SEMI_TICKERS = {
    "NVDA": 2.20e12,
    "AMD": 2.80e11,
    "AVGO": 6.20e11,
    "TSM": 7.50e11,
    "ASML": 3.80e11,
}

START_DATE = "2023-01-01"

# =========================================================
# DATA LOADERS (SAFE ENDPOINTS ONLY)
# =========================================================

@st.cache_data(ttl=3600)
def load_prices(tickers):
    df = yf.download(
        list(tickers),
        start=START_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]
    return df.dropna()

# =========================================================
# CAP-WEIGHTED AI / SEMIS INDEX
# =========================================================

def build_cap_weighted_index(price_df, caps):
    caps = pd.Series(caps, dtype="float64")
    weights = caps / caps.sum()
    weighted = price_df.mul(weights, axis=1)
    index = weighted.sum(axis=1)
    return index, weights

# =========================================================
# SIGNAL ENGINE
# =========================================================

def zscore(series, window=60):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def compute_signals(spx, ai_index):
    df = pd.DataFrame({
        "SPX": spx,
        "AI": ai_index
    })

    # Relative Strength
    rs = df["AI"] / df["SPX"]
    rs_z = zscore(rs)

    # Volatility (proxy: absolute returns)
    vol = df["SPX"].pct_change().abs().rolling(20).mean()
    vol_z = zscore(vol)

    # Trend
    trend = df["SPX"].pct_change(50)
    trend_z = zscore(trend)

    signals = pd.DataFrame({
        "RS_Z": rs_z,
        "VOL_Z": vol_z,
        "TREND_Z": trend_z
    })

    return df, signals.dropna()

# =========================================================
# REGIME LOGIC
# =========================================================

def classify_regime(latest):
    signal_flags = {
        "rs": latest["RS_Z"] > 0,
        "vol": latest["VOL_Z"] < 0,
        "trend": latest["TREND_Z"] > 0,
    }

    confidence = sum(signal_flags.values()) / len(signal_flags)

    if confidence >= 0.67:
        regime = "🟢 RISK-ON"
    elif confidence >= 0.40:
        regime = "🟡 CAUTION"
    else:
        regime = "🔴 RISK-OFF"

    return regime, confidence, signal_flags

# =========================================================
# LOAD DATA
# =========================================================

price_data = load_prices([SPX_TICKER] + list(AI_SEMI_TICKERS.keys()))

spx = price_data[SPX_TICKER]
ai_prices = price_data[list(AI_SEMI_TICKERS.keys())]

ai_index, weights = build_cap_weighted_index(ai_prices, AI_SEMI_TICKERS)

df, signals = compute_signals(spx, ai_index)

latest = signals.iloc[-1]
regime, confidence, flags = classify_regime(latest)

# =========================================================
# HEADER
# =========================================================

st.title("🧬 Market Immune System — v4")
st.caption(f"Last close: {signals.index[-1].date()}")

col1, col2, col3 = st.columns(3)
col1.metric("Regime", regime)
col2.metric("Confidence", f"{confidence:.2f}")
col3.metric("SPX Level", f"{spx.iloc[-1]:,.0f}")

# =========================================================
# REGIME SHADING PLOT
# =========================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(spx.index, spx, label="SPX", linewidth=2)
ax.plot(spx.index, ai_index * (spx.iloc[0] / ai_index.iloc[0]),
        label="AI/Semis (Cap-Weighted, Normalized)", linestyle="--")

for i in range(1, len(signals)):
    row = signals.iloc[i]
    prev_date = signals.index[i - 1]
    curr_date = signals.index[i]

    _, conf, _ = classify_regime(row)

    if conf >= 0.67:
        color = "green"
        alpha = 0.08
    elif conf >= 0.40:
        color = "yellow"
        alpha = 0.10
    else:
        color = "red"
        alpha = 0.12

    ax.axvspan(prev_date, curr_date, color=color, alpha=alpha)

ax.set_title("SPX vs Cap-Weighted AI/Semis with Regime Shading")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =========================================================
# SIGNAL DIAGNOSTICS
# =========================================================

st.subheader("Signal Diagnostics")

diag = pd.DataFrame({
    "Signal": ["Relative Strength", "Volatility", "Trend"],
    "Z-Score": [
        latest["RS_Z"],
        latest["VOL_Z"],
        latest["TREND_Z"]
    ],
    "Bullish": [
        flags["rs"],
        flags["vol"],
        flags["trend"]
    ]
})

st.dataframe(diag, use_container_width=True)

# =========================================================
# CAP WEIGHTS DISPLAY
# =========================================================

st.subheader("AI / Semiconductor Cap Weights")

cap_df = pd.DataFrame({
    "Ticker": weights.index,
    "Weight": weights.values
}).sort_values("Weight", ascending=False)

st.dataframe(cap_df, use_container_width=True)

# =========================================================
# DAILY SNAPSHOT EXPORT
# =========================================================

snapshot = pd.DataFrame([{
    "date": signals.index[-1].date(),
    "regime": regime,
    "confidence": confidence,
    "rs_z": latest["RS_Z"],
    "vol_z": latest["VOL_Z"],
    "trend_z": latest["TREND_Z"],
    "spx": spx.iloc[-1]
}])

st.download_button(
    "Download Daily Snapshot",
    snapshot.to_csv(index=False),
    "market_immune_system_snapshot.csv",
)
