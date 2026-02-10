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

st.set_page_config(layout="wide", page_title="Market Immune System v6")

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
HISTORY_FILE = Path("regime_history.csv")

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

def zscore(series, window):
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
# STATE / HISTORY / ALERTS
# =========================================================

def load_previous_state():
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

def send_alert(message):
    webhook = st.secrets.get("ALERT_WEBHOOK", None)
    if webhook:
        requests.post(webhook, json={"text": message})

# =========================================================
# LOAD DATA
# =========================================================

prices = load_prices([SPX_TICKER] + list(AI_SEMI_TICKERS.keys()))
spx = prices[SPX_TICKER]
ai_prices = prices[list(AI_SEMI_TICKERS.keys())]

ai_index, weights = build_cap_weighted_index(ai_prices, AI_SEMI_TICKERS)
signals = compute_signals(spx, ai_index)

# =========================================================
# 🔒 DATA READINESS GUARD (FIXES YOUR ERROR)
# =========================================================

if signals.empty:
    st.warning(
        "Not enough historical data yet to compute signals. "
        "Waiting for sufficient lookback window."
    )
    st.stop()

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
# HISTORY + ALERTS
# =========================================================

row = {
    "date": signals.index[-1].date(),
    "regime": regime,
    "confidence": confidence,
    "btc_stress": btc_latest,
    "rs_z": latest["RS_Z"],
    "vol_z": latest["VOL_Z"],
    "trend_z": latest["TREND_Z"],
}

history = update_history(row)

prev_state = load_previous_state()
if prev_state.get("regime") and prev_state["regime"] != regime:
    send_alert(
        f"🚨 Market Regime Change\n"
        f"{prev_state['regime']} → {regime}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Date: {row['date']}"
    )

save_state({"regime": regime, "date": str(row["date"])})

# =========================================================
# V6 ANALYTICS
# =========================================================

history["drawdown_risk"] = (
    (history["regime"] == "🔴 RISK-OFF").astype(int)
    * (1 - history["confidence"])
)

regime_duration = (
    history.iloc[::-1]["regime"]
    .ne(history.iloc[::-1]["regime"].shift())
    .cumsum()
    .value_counts()
    .iloc[0]
)

drawdown_prob = history["drawdown_risk"].rolling(60).mean().iloc[-1]

# =========================================================
# UI
# =========================================================

st.title("🧬 Market Immune System — v6")
st.caption(f"Last close: {row['date']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", regime)
c2.metric("Confidence", f"{confidence:.2f}")
c3.metric("Regime Duration (days)", regime_duration)
c4.metric("5–10% Drawdown Risk", f"{drawdown_prob:.0%}")

# =========================================================
# REGIME SHADING CHART
# =========================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(spx.index, spx, label="SPX", linewidth=2)
ax.plot(
    spx.index,
    ai_index * (spx.iloc[0] / ai_index.iloc[0]),
    linestyle="--",
    label="AI/Semis (Cap-Weighted)"
)

for i in range(1, len(signals)):
    _, conf, _ = classify_regime(signals.iloc[i])
    color = "green" if conf >= 0.67 else "yellow" if conf >= 0.40 else "red"
    ax.axvspan(signals.index[i-1], signals.index[i], color=color, alpha=0.08)

ax.legend()
ax.grid(True)
ax.set_title("SPX vs AI/Semis with Regime Shading")

st.pyplot(fig)

# =========================================================
# HISTORY TABLE
# =========================================================

st.subheader("Regime History")
st.dataframe(history.tail(30), use_container_width=True)
