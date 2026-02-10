import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
import json

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(layout="wide", page_title="Market Immune System v8")

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
# HELPERS
# =========================================================

def zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def build_cap_weighted_index(price_df, caps):
    caps = pd.Series(caps, dtype="float64")
    weights = caps / caps.sum()
    return price_df.mul(weights, axis=1).sum(axis=1), weights

# =========================================================
# SIGNAL ENGINE
# =========================================================

def compute_signals(spx, ai_index):
    rs = ai_index / spx
    vol = spx.pct_change().abs().rolling(20).mean()
    trend = spx.pct_change(50)

    df = pd.DataFrame({
        "RS_Z": zscore(rs, 60),
        "VOL_Z": zscore(vol, 60),
        "TREND_Z": zscore(trend, 60),
    })

    return df.dropna()

def compute_btc_stress(btc):
    mcap = btc.pct_change(30)
    rcap = btc.rolling(180).mean().pct_change(30)
    stress = mcap - rcap
    return zscore(stress, 90)

# =========================================================
# REGIME CLASSIFIER
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
# HISTORY
# =========================================================

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

if signals.empty or len(signals) < 150:
    st.warning("Insufficient data for regime inference.")
    st.stop()

# =========================================================
# BTC STRESS (BULLETPROOF SCALAR EXTRACTION)
# =========================================================

btc = load_btc()
btc_stress = compute_btc_stress(btc)

btc_aligned = btc_stress.reindex(signals.index)

if len(btc_aligned) == 0:
    btc_latest = 0.0
else:
    btc_latest = btc_aligned.iloc[-1]

    # Force scalar extraction
    if isinstance(btc_latest, (pd.Series, np.ndarray)):
        btc_latest = btc_latest.squeeze()

    btc_latest = float(btc_latest) if not pd.isna(btc_latest) else 0.0

# =========================================================
# CURRENT REGIME
# =========================================================

latest = signals.iloc[-1]
regime, confidence = classify_regime(latest)

# BTC modifier (safe scalar logic)
if btc_latest > 1.0:
    confidence -= 0.15
elif btc_latest < -1.0:
    confidence += 0.10

confidence = float(np.clip(confidence, 0, 1))

# =========================================================
# FORWARD RETURNS
# =========================================================

forward = pd.DataFrame({
    "fwd_5d": spx.pct_change(5).shift(-5),
    "fwd_20d": spx.pct_change(20).shift(-20),
    "fwd_60d": spx.pct_change(60).shift(-60),
})

signal_hist = signals.join(forward).dropna()
signal_hist["regime"] = signal_hist.apply(lambda r: classify_regime(r)[0], axis=1)

# =========================================================
# DRAWDOWN PROBABILITY
# =========================================================

drawdown = spx.pct_change(20).shift(-20) < -0.07
signal_hist["drawdown"] = drawdown

drawdown_prob = (
    signal_hist[signal_hist["regime"] == regime]["drawdown"].mean()
)

# =========================================================
# REGIME DURATION
# =========================================================

runs = (signal_hist["regime"] != signal_hist["regime"].shift()).cumsum()
durations = signal_hist.groupby(runs).size()
current_duration = durations.iloc[-1]

# =========================================================
# HISTORY ROW
# =========================================================

row = {
    "date": signals.index[-1].date(),
    "regime": regime,
    "confidence": confidence,
    "btc_stress": btc_latest,
}

history = update_history(row)

# =========================================================
# EXPOSURE GUIDANCE
# =========================================================

exposure = {
    "🟢 RISK-ON": 1.00,
    "🟡 CAUTION": 0.55,
    "🔴 RISK-OFF": 0.20
}[regime]

# =========================================================
# UI
# =========================================================

st.title("🧬 Market Immune System — v8")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Regime", regime)
c2.metric("Confidence", f"{confidence:.2f}")
c3.metric("Suggested Exposure", f"{int(exposure*100)}%")
c4.metric("20d Drawdown Risk", f"{drawdown_prob:.0%}")
c5.metric("Regime Duration (days)", current_duration)

# =========================================================
# CHART
# =========================================================

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(spx.index, spx, label="SPX", linewidth=2)

for i in range(1, len(signals)):
    reg, conf = classify_regime(signals.iloc[i])
    color = "green" if conf >= 0.67 else "yellow" if conf >= 0.40 else "red"
    ax.axvspan(signals.index[i-1], signals.index[i], color=color, alpha=0.06)

ax.set_title("SPX with Regime Overlay")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# =========================================================
# TABLES
# =========================================================

st.subheader("Regime-Conditioned Forward Returns")
st.dataframe(
    signal_hist.groupby("regime")[["fwd_5d", "fwd_20d", "fwd_60d"]]
    .mean()
    .style.format("{:.2%}"),
    use_container_width=True
)

st.subheader("Recent Regime History")
st.dataframe(history.tail(30), use_container_width=True)
