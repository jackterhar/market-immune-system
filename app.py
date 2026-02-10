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

st.set_page_config(layout="wide", page_title="Market Immune System v7")

SPX_TICKER = "^GSPC"
START_DATE = "2022-01-01"

AI_SEMI_TICKERS = {
    "NVDA": 2.20e12,
    "AMD": 2.80e11,
    "AVGO": 6.20e11,
    "TSM": 7.50e11,
    "ASML": 3.80e11,
}

STATE_FILE = Path("state.json")
HISTORY_FILE = Path("regime_history.csv")

# =========================================================
# DATA LOADERS
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
# UTILITIES
# =========================================================

def zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def build_cap_weighted_index(price_df, caps):
    caps = pd.Series(caps, dtype="float64")
    weights = caps / caps.sum()
    index = price_df.mul(weights, axis=1).sum(axis=1)
    return index, weights

# =========================================================
# SIGNAL ENGINE
# =========================================================

def compute_signals(spx, ai_index):
    rs = ai_index / spx
    vol = spx.pct_change().abs().rolling(20).mean()
    trend = spx.pct_change(50)

    signals = pd.DataFrame({
        "RS_Z": zscore(rs, 60),
        "VOL_Z": zscore(vol, 60),
        "TREND_Z": zscore(trend, 60),
    })

    return signals.dropna()

def compute_btc_stress(btc):
    mcap_growth = btc.pct_change(30)
    rcap_proxy = btc.rolling(180).mean()
    rcap_growth = rcap_proxy.pct_change(30)
    stress = mcap_growth - rcap_growth
    return zscore(stress, 90).dropna()

# =========================================================
# REGIME LOGIC
# =========================================================

def classify_regime(row):
    flags = {
        "rs": row["RS_Z"] > 0,
        "vol": row["VOL_Z"] < 0,
        "trend": row["TREND_Z"] > 0,
    }
    confidence = sum(flags.values()) / len(flags)

    if confidence >= 0.67:
        return "🟢 RISK-ON", confidence
    elif confidence >= 0.40:
        return "🟡 CAUTION", confidence
    else:
        return "🔴 RISK-OFF", confidence

# =========================================================
# STATE / HISTORY
# =========================================================

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state))

def update_history(row):
    if HISTORY_FILE.exists():
        hist = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
        if hist.iloc[-1]["date"].date() == row["date"]:
            return hist
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])

    hist.to_csv(HISTORY_FILE, index=False)
    return hist

# =========================================================
# LOAD DATA
# =========================================================

prices = load_prices([SPX_TICKER] + list(AI_SEMI_TICKERS.keys()))
spx = prices[SPX_TICKER]
ai_prices = prices[list(AI_SEMI_TICKERS.keys())]

ai_index, weights = build_cap_weighted_index(ai_prices, AI_SEMI_TICKERS)
signals = compute_signals(spx, ai_index)

# =========================================================
# DATA GUARD (CRITICAL)
# =========================================================

if signals.empty or len(signals) < 120:
    st.warning("Insufficient historical data to compute stable regimes.")
    st.stop()

# =========================================================
# BTC MODIFIER
# =========================================================

btc = load_btc()
btc_stress = compute_btc_stress(btc)
btc_latest = btc_stress.reindex(signals.index).iloc[-1]

# =========================================================
# CURRENT REGIME
# =========================================================

latest = signals.iloc[-1]
regime, confidence = classify_regime(latest)

if btc_latest > 1:
    confidence -= 0.15
elif btc_latest < -1:
    confidence += 0.10

confidence = float(np.clip(confidence, 0, 1))

# =========================================================
# FORWARD RETURNS (V7 CORE)
# =========================================================

forward_returns = pd.DataFrame({
    "fwd_5d": spx.pct_change(5).shift(-5),
    "fwd_20d": spx.pct_change(20).shift(-20),
    "fwd_60d": spx.pct_change(60).shift(-60),
})

signal_hist = signals.join(forward_returns).dropna()

def regime_forward_stats(label):
    subset = signal_hist[signal_hist["regime"] == label]
    return subset[["fwd_5d", "fwd_20d", "fwd_60d"]].mean()

# =========================================================
# BUILD HISTORY ROW
# =========================================================

row = {
    "date": signals.index[-1].date(),
    "regime": regime,
    "confidence": confidence,
    "btc_stress": btc_latest,
}

history = update_history(row)

# =========================================================
# ADD REGIME COLUMN TO SIGNAL HISTORY
# =========================================================

signal_hist["regime"] = signal_hist.apply(
    lambda r: classify_regime(r)[0], axis=1
)

# =========================================================
# HIT RATE VALIDATION
# =========================================================

drawdown = spx.pct_change(20).shift(-20) < -0.07
signal_hist["drawdown"] = drawdown

hit_rate = (
    signal_hist[signal_hist["regime"] == "🔴 RISK-OFF"]["drawdown"].mean()
)

# =========================================================
# POSITION SIZING (NOT A SIGNAL)
# =========================================================

if regime == "🟢 RISK-ON":
    exposure = 1.00
elif regime == "🟡 CAUTION":
    exposure = 0.55
else:
    exposure = 0.20

# =========================================================
# UI
# =========================================================

st.title("🧬 Market Immune System — v7")
st.caption(f"Last close: {row['date']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", regime)
c2.metric("Confidence", f"{confidence:.2f}")
c3.metric("Suggested Exposure", f"{int(exposure*100)}%")
c4.metric("Risk-Off Hit Rate", f"{hit_rate:.0%}")

# =========================================================
# PRICE + REGIME SHADING
# =========================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(spx.index, spx, label="SPX", linewidth=2)

for i in range(1, len(signals)):
    reg, conf = classify_regime(signals.iloc[i])
    color = "green" if conf >= 0.67 else "yellow" if conf >= 0.40 else "red"
    ax.axvspan(signals.index[i-1], signals.index[i], color=color, alpha=0.07)

ax.legend()
ax.grid(True)
ax.set_title("SPX with Regime Overlay")

st.pyplot(fig)

# =========================================================
# FORWARD RETURN TABLE
# =========================================================

st.subheader("Regime-Conditioned Forward Returns (Mean)")
table = signal_hist.groupby("regime")[["fwd_5d", "fwd_20d", "fwd_60d"]].mean()
st.dataframe(table.style.format("{:.2%}"), use_container_width=True)

# =========================================================
# RECENT HISTORY
# =========================================================

st.subheader("Recent Regime History")
st.dataframe(history.tail(25), use_container_width=True)
