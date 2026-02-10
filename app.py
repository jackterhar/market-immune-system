import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import json
import requests
from pathlib import Path

# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(layout="wide", page_title="Market Immune System v5")

SPX_TICKER = "^GSPC"
START_DATE = "2023-01-01"

AI_SEMI_TICKERS = {
    "NVDA": 2.20e12,
    "AMD": 2.80e11,
    "AVGO": 6.20e11,
    "TSM": 7.50e11,
    "ASML": 3.80e11,
}

STATE_FILE = Path("state.json")

# =========================================================
# DATA LOADERS (SAFE ENDPOINTS ONLY)
# =========================================================

@st.cache_data(ttl=3600)
def load_prices(tickers):
    df = yf.download(
        tickers,
        start=START_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]
    return df.dropna()

@st.cache_data(ttl=3600)
def load_btc():
    btc = yf.download(
        "BTC-USD",
        start=START_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]
    return btc.dropna()

# =========================================================
# SIGNAL ENGINE
# =========================================================

def zscore(series, window=60):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def build_cap_weighted_index(price_df, caps):
    caps = pd.Series(caps, dtype="float64")
    weights = caps / caps.sum()
    index = price_df.mul(weights, axis=1).sum(axis=1)
    return index, weights

def compute_signals(spx, ai_index):
    rs = ai_index / spx
    vol = spx.pct_change().abs().rolling(20).mean()
    trend = spx.pct_change(50)

    signals = pd.DataFrame({
        "RS_Z": zscore(rs),
        "VOL_Z": zscore(vol),
        "TREND_Z": zscore(trend),
    })

    return signals.dropna()

def compute_btc_stress(btc):
    mcap_growth = btc.pct_change(30)
    rcap_proxy = btc.rolling(180).mean()
    rcap_growth = rcap_proxy.pct_change(30)

    stress = mcap_growth - rcap_growth
    stress_z = zscore(stress, 90)

    return stress_z.dropna()

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
# STATE + ALERTS
# =========================================================

def load_previous_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def send_alert(message):
    webhook = st.secrets.get("ALERT_WEBHOOK", None)
    if webhook:
        requests.post(webhook, json={"text": message})

# =========================================================
# LOAD & COMPUTE
# =========================================================

prices = load_prices([SPX_TICKER] + list(AI_SEMI_TICKERS.keys()))
spx = prices[SPX_TICKER]
ai_prices = prices[list(AI_SEMI_TICKERS.keys())]

ai_index, weights = build_cap_weighted_index(ai_prices, AI_SEMI_TICKERS)
signals = compute_signals(spx, ai_index)

latest = signals.iloc[-1]
regime, confidence, flags = classify_regime(latest)

# =========================================================
# BTC MODIFIER
# =========================================================

btc = load_btc()
btc_stress = compute_btc_stress(btc)
btc_latest = btc_stress.loc[signals.index[-1]]

btc_penalty = 0
if btc_latest > 1:
    btc_penalty = 0.15
elif btc_latest < -1:
    btc_penalty = -0.10

confidence = max(0, min(1, confidence - btc_penalty))

# =========================================================
# ALERT ON REGIME CHANGE
# =========================================================

prev_state = load_previous_state()
prev_regime = prev_state.get("regime")

if prev_regime and prev_regime != regime:
    send_alert(
        f"🚨 Market Regime Change\n"
        f"From: {prev_regime}\n"
        f"To: {regime}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Date: {signals.index[-1].date()}"
    )

save_state({
    "regime": regime,
    "confidence": confidence,
    "date": str(signals.index[-1].date())
})

# =========================================================
# UI — HEADER
# =========================================================

st.title("🧬 Market Immune System — v5")
st.caption(f"Last close: {signals.index[-1].date()}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", regime)
c2.metric("Confidence", f"{confidence:.2f}")
c3.metric("SPX", f"{spx.iloc[-1]:,.0f}")
c4.metric("BTC Stress Z", f"{btc_latest:.2f}")

# =========================================================
# REGIME SHADING PLOT
# =========================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(spx.index, spx, label="SPX", linewidth=2)
ax.plot(
    spx.index,
    ai_index * (spx.iloc[0] / ai_index.iloc[0]),
    label="AI/Semis (Cap-Weighted)",
    linestyle="--"
)

for i in range(1, len(signals)):
    row = signals.iloc[i]
    prev_date = signals.index[i - 1]
    curr_date = signals.index[i]

    _, conf, _ = classify_regime(row)

    if conf >= 0.67:
        ax.axvspan(prev_date, curr_date, color="green", alpha=0.07)
    elif conf >= 0.40:
        ax.axvspan(prev_date, curr_date, color="yellow", alpha=0.10)
    else:
        ax.axvspan(prev_date, curr_date, color="red", alpha=0.12)

ax.set_title("SPX vs AI/Semis with Regime Shading")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =========================================================
# DIAGNOSTICS
# =========================================================

st.subheader("Signal Diagnostics")

diag = pd.DataFrame({
    "Signal": ["Relative Strength", "Volatility", "Trend"],
    "Z-Score": [latest["RS_Z"], latest["VOL_Z"], latest["TREND_Z"]],
    "Bullish": [flags["rs"], flags["vol"], flags["trend"]],
})

st.dataframe(diag, use_container_width=True)

# =========================================================
# CAP WEIGHTS
# =========================================================

st.subheader("AI / Semiconductor Cap Weights")

cap_df = pd.DataFrame({
    "Ticker": weights.index,
    "Weight": weights.values
}).sort_values("Weight", ascending=False)

st.dataframe(cap_df, use_container_width=True)

# =========================================================
# SNAPSHOT EXPORT
# =========================================================

snapshot = pd.DataFrame([{
    "date": signals.index[-1].date(),
    "regime": regime,
    "confidence": confidence,
    "rs_z": latest["RS_Z"],
    "vol_z": latest["VOL_Z"],
    "trend_z": latest["TREND_Z"],
    "btc_stress_z": btc_latest,
    "spx": spx.iloc[-1],
}])

st.download_button(
    "Download Daily Snapshot",
    snapshot.to_csv(index=False),
    "market_immune_system_snapshot.csv"
)
