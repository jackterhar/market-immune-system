import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# =========================================================
# DATA LOAD
# =========================================================

@st.cache_data
def load_data():
    tickers = {
        "SPX": "^GSPC",
        "SPY": "SPY",
        "BTC": "BTC-USD",
        "VIX": "^VIX",
        "TNX": "^TNX",          # 10Y yield
        "DXY": "DX-Y.NYB",      # Dollar index
        "HYG": "HYG",           # High yield ETF
        "IEF": "IEF",           # Treasuries
    }

    data = {}
    for k, t in tickers.items():
        data[k] = yf.download(t, start="2022-01-01")["Close"]

    df = pd.DataFrame(data)
    return df.dropna()

df = load_data()

# =========================================================
# FEATURE ENGINEERING
# =========================================================

data = df.copy()

# Returns
ret = data.pct_change()

# ----------------------
# TREND
# ----------------------
data["SPX_50"] = data["SPX"].rolling(50).mean()
data["SPX_200"] = data["SPX"].rolling(200).mean()
trend_signal = (data["SPX_50"] - data["SPX_200"]) / data["SPX"]

trend_z = (trend_signal - trend_signal.rolling(252).mean()) / trend_signal.rolling(252).std()

# ----------------------
# VOLATILITY
# ----------------------
vol_20 = ret["SPX"].rolling(20).std()
vol_z = (vol_20 - vol_20.rolling(252).mean()) / vol_20.rolling(252).std()

vix_z = (data["VIX"] - data["VIX"].rolling(252).mean()) / data["VIX"].rolling(252).std()

vol_composite = -(vol_z + vix_z) / 2  # negative vol = bullish

# ----------------------
# CREDIT
# ----------------------
credit_ratio = data["HYG"] / data["IEF"]
credit_z = (credit_ratio - credit_ratio.rolling(252).mean()) / credit_ratio.rolling(252).std()

# ----------------------
# LIQUIDITY
# ----------------------
rates_z = (data["TNX"] - data["TNX"].rolling(252).mean()) / data["TNX"].rolling(252).std()
dxy_z = (data["DXY"] - data["DXY"].rolling(252).mean()) / data["DXY"].rolling(252).std()

liquidity_composite = -(rates_z + dxy_z) / 2

# ----------------------
# BTC STRUCTURE
# ----------------------
btc_trend = data["BTC"].rolling(50).mean() - data["BTC"].rolling(200).mean()
btc_trend_z = (btc_trend - btc_trend.rolling(252).mean()) / btc_trend.rolling(252).std()

btc_vol = ret["BTC"].rolling(20).std()
btc_vol_z = (btc_vol - btc_vol.rolling(252).mean()) / btc_vol.rolling(252).std()

btc_composite = (btc_trend_z - btc_vol_z)

# =========================================================
# COMPOSITE SCORE
# =========================================================

composite = (
    0.25 * trend_z +
    0.20 * vol_composite +
    0.20 * credit_z +
    0.20 * liquidity_composite +
    0.15 * btc_composite
)

composite = composite.dropna()

latest_score = composite.iloc[-1]

# =========================================================
# REGIME CLASSIFICATION
# =========================================================

if latest_score > 0.75:
    regime = "RISK-ON"
    exposure = 1.0
elif latest_score < -0.75:
    regime = "RISK-OFF"
    exposure = 0.2
else:
    regime = "CAUTION"
    exposure = 0.5

confidence = min(abs(latest_score) / 2, 1.0)

# =========================================================
# FORWARD STATISTICS
# =========================================================

fwd_20 = data["SPX"].pct_change(20).shift(-20)
drawdown_prob = (fwd_20 < -0.07).mean()

exp_return = fwd_20.mean()
exp_vol = fwd_20.std()

# =========================================================
# WHY TODAY DECOMPOSITION
# =========================================================

latest = composite.index[-1]

why_today = pd.DataFrame({
    "Factor": [
        "Trend",
        "Volatility",
        "Credit",
        "Liquidity",
        "Crypto"
    ],
    "Contribution": [
        0.25 * trend_z.loc[latest],
        0.20 * vol_composite.loc[latest],
        0.20 * credit_z.loc[latest],
        0.20 * liquidity_composite.loc[latest],
        0.15 * btc_composite.loc[latest],
    ]
})

# =========================================================
# DAILY SUMMARY
# =========================================================

summary = (
    f"The composite cross-asset regime score is {latest_score:.2f}, "
    f"placing the market in a **{regime}** posture with {confidence:.0%} confidence. "
    f"Trend and credit conditions are currently "
    f"{'supportive' if trend_z.iloc[-1] > 0 else 'fragile'}, "
    f"while volatility dynamics are "
    f"{'benign' if vol_composite.iloc[-1] > 0 else 'elevated'}. "
    f"Liquidity pressure from rates and dollar is "
    f"{'accommodative' if liquidity_composite.iloc[-1] > 0 else 'tightening'}. "
    f"BTC structure is "
    f"{'risk-confirming' if btc_composite.iloc[-1] > 0 else 'risk-diverging'}. "
    f"Forward 20-day expectancy is {exp_return:.2%} with a "
    f"{drawdown_prob:.0%} probability of >7% drawdown."
)

# =========================================================
# UI
# =========================================================

st.title("Market Immune System — V10 (Institutional)")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Regime", regime)
col2.metric("Confidence", f"{confidence:.2f}")
col3.metric("Suggested Exposure", f"{int(exposure*100)}%")
col4.metric("20d Drawdown Risk", f"{drawdown_prob:.0%}")

# Plot SPX + composite
fig, ax1 = plt.subplots(figsize=(14,6))

ax1.plot(data.index, data["SPX"], label="SPX")
ax2 = ax1.twinx()
ax2.plot(composite.index, composite, linestyle="--")

st.pyplot(fig)

st.subheader("Why Today?")
st.dataframe(why_today, use_container_width=True)

st.subheader("Forward Risk Metrics")
st.write(f"20d Expected Return: {exp_return:.2%}")
st.write(f"20d Expected Volatility: {exp_vol:.2%}")
st.write(f"Probability of >7% Drawdown: {drawdown_prob:.0%}")

st.subheader("Daily Summary")
st.markdown(summary)
