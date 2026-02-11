import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# =========================================================
# DATA LOADERS
# =========================================================

@st.cache_data
def load_data():
    spx = yf.download("^GSPC", start="2022-01-01")
    btc = yf.download("BTC-USD", start="2022-01-01")

    spx["ret"] = spx["Close"].pct_change()
    btc["ret"] = btc["Close"].pct_change()

    return spx, btc


spx, btc = load_data()

# =========================================================
# FEATURE ENGINEERING
# =========================================================

data = pd.DataFrame(index=spx.index)
data["SPX"] = spx["Close"]
data["SPX_ret"] = spx["ret"]

data["BTC"] = btc["Close"]
data["BTC_ret"] = btc["ret"]

# Relative strength (BTC vs SPX)
data["RS"] = data["BTC_ret"].rolling(20).mean()
data["RS_Z"] = (data["RS"] - data["RS"].rolling(252).mean()) / data["RS"].rolling(252).std()

# Volatility regime
data["VOL"] = data["SPX_ret"].rolling(20).std()
data["VOL_Z"] = (data["VOL"] - data["VOL"].rolling(252).mean()) / data["VOL"].rolling(252).std()

# Trend regime
data["TREND"] = data["SPX"].rolling(50).mean() - data["SPX"].rolling(200).mean()
data["TREND_Z"] = (data["TREND"] - data["TREND"].rolling(252).mean()) / data["TREND"].rolling(252).std()

# BTC Stress (absolute vol spike proxy)
data["BTC_STRESS"] = data["BTC_ret"].rolling(10).std()
data["BTC_STRESS_Z"] = (
    data["BTC_STRESS"] - data["BTC_STRESS"].rolling(252).mean()
) / data["BTC_STRESS"].rolling(252).std()

data = data.dropna()

latest = data.iloc[-1]

# =========================================================
# SIGNAL ENGINE
# =========================================================

score = 0

score += 1 if latest["RS_Z"] > 0 else -1
score += 1 if latest["VOL_Z"] < 0 else -1
score += 1 if latest["TREND_Z"] > 0 else -1
score += -1 if latest["BTC_STRESS_Z"] > 1 else 0

confidence = abs(score) / 4

if score >= 2:
    regime = "RISK-ON"
    exposure = 1.0
elif score <= -2:
    regime = "RISK-OFF"
    exposure = 0.2
else:
    regime = "CAUTION"
    exposure = 0.5

btc_latest = latest["BTC_STRESS_Z"]

# =========================================================
# REGIME SERIES (FOR PLOTTING)
# =========================================================

def classify_row(row):
    s = 0
    s += 1 if row["RS_Z"] > 0 else -1
    s += 1 if row["VOL_Z"] < 0 else -1
    s += 1 if row["TREND_Z"] > 0 else -1
    s += -1 if row["BTC_STRESS_Z"] > 1 else 0

    if s >= 2:
        return "RISK-ON"
    elif s <= -2:
        return "RISK-OFF"
    else:
        return "CAUTION"


data["Regime"] = data.apply(classify_row, axis=1)
regime_series = data["Regime"]

# Current regime duration
current_duration = (
    regime_series[::-1].eq(regime).cumprod().sum()
)

# 20-day drawdown proxy
future_returns = data["SPX"].pct_change(20).shift(-20)
drawdown_prob = (future_returns < -0.07).mean()

# =========================================================
# WHY TODAY
# =========================================================

rs_flag = latest["RS_Z"] > 0
vol_flag = latest["VOL_Z"] < 0
trend_flag = latest["TREND_Z"] > 0

why_today = pd.DataFrame({
    "Signal": [
        "Relative Strength (BTC vs SPX)",
        "Volatility Regime",
        "Trend",
        "BTC Stress"
    ],
    "Status": [
        "Positive" if rs_flag else "Negative",
        "Supportive" if vol_flag else "Elevated",
        "Positive" if trend_flag else "Negative",
        "Elevated" if latest["BTC_STRESS_Z"] > 1 else "Normal"
    ],
    "Z-Score": [
        round(latest["RS_Z"], 2),
        round(latest["VOL_Z"], 2),
        round(latest["TREND_Z"], 2),
        round(latest["BTC_STRESS_Z"], 2),
    ],
})

# =========================================================
# DAILY SUMMARY
# =========================================================

summary = (
    f"As of the latest close, the Market Immune System is in a "
    f"**{regime}** regime with **confidence {confidence:.2f}**. "
    f"Relative strength is {'supportive' if rs_flag else 'weak'}, "
    f"volatility conditions are {'benign' if vol_flag else 'elevated'}, "
    f"and trend signals remain {'constructive' if trend_flag else 'fragile'}. "
    f"BTC-derived stress is {'a headwind' if btc_latest > 1 else 'neutral to supportive'}, "
    f"and the current regime has persisted for **{current_duration} days**. "
    f"Historically, similar conditions have been associated with a "
    f"**{drawdown_prob:.0%} probability of a >7% drawdown over 20 trading days**, "
    f"supporting a **{int(exposure*100)}% risk exposure posture**."
)

# =========================================================
# UI
# =========================================================

st.title("Market Immune System — V9")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Regime", regime)
col2.metric("Confidence", f"{confidence:.2f}")
col3.metric("Suggested Exposure", f"{int(exposure*100)}%")
col4.metric("20d Drawdown Risk", f"{drawdown_prob:.0%}")
col5.metric("Regime Duration (days)", current_duration)

# Plot SPX with regime shading
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(data.index, data["SPX"], label="SPX")

for i in range(len(data)-1):
    color = {
        "RISK-ON": "green",
        "CAUTION": "yellow",
        "RISK-OFF": "red"
    }[regime_series.iloc[i]]
    ax.axvspan(data.index[i], data.index[i+1], color=color, alpha=0.1)

ax.legend()
st.pyplot(fig)

# Why today
st.subheader("Why Today?")
st.dataframe(why_today, use_container_width=True)

# Daily summary
st.subheader("Daily Summary")
st.markdown(summary)
