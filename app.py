import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# =========================================================
# ROBUST DATA LOADER
# =========================================================

@st.cache_data
def load_data():

    tickers = [
        "^GSPC",
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

    closes = raw["Close"].dropna(how="all")

    return closes.dropna()

df = load_data()

df.columns = [
    "SPX",
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
# FACTORS
# =========================================================

# Trend
data["MA50"] = data["SPX"].rolling(50).mean()
data["MA200"] = data["SPX"].rolling(200).mean()
trend_raw = (data["MA50"] - data["MA200"]) / data["SPX"]
trend_z = (trend_raw - trend_raw.rolling(252).mean()) / trend_raw.rolling(252).std()

# Vol
vol_20 = ret["SPX"].rolling(20).std()
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

# BTC
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
).dropna()

data = data.loc[composite.index]

# =========================================================
# REGIME SERIES
# =========================================================

regime_series = pd.Series(index=composite.index)

regime_series[composite > 0.75] = 1
regime_series[composite < -0.75] = -1
regime_series[(composite >= -0.75) & (composite <= 0.75)] = 0

latest_regime = regime_series.iloc[-1]
latest_score = float(composite.iloc[-1])

if latest_regime == 1:
    regime = "RISK-ON"
    exposure = 1.0
elif latest_regime == -1:
    regime = "RISK-OFF"
    exposure = 0.2
else:
    regime = "CAUTION"
    exposure = 0.5

confidence = min(abs(latest_score)/2,1.0)

# Regime duration
duration = (regime_series[::-1] == latest_regime).cumprod().sum()

# =========================================================
# FORWARD RISK
# =========================================================

fwd_20 = data["SPX"].pct_change(20).shift(-20)
drawdown_prob = float((fwd_20 < -0.07).mean())

# =========================================================
# UI
# =========================================================

st.title("Market Immune System — V11")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Regime", regime)
c2.metric("Confidence", f"{confidence:.2f}")
c3.metric("Exposure", f"{int(exposure*100)}%")
c4.metric("20d Drawdown Risk", f"{drawdown_prob:.0%}")

# =========================================================
# CLEAN REGIME SHADING
# =========================================================

fig, ax = plt.subplots(figsize=(15,7))

ax.plot(data.index, data["SPX"], linewidth=2)

for i in range(1,len(regime_series)):
    if regime_series.iloc[i] != regime_series.iloc[i-1]:
        start = regime_series.index[i]
        break

for date, value in regime_series.items():
    if value == 1:
        ax.axvspan(date, date, color="green", alpha=0.08)
    elif value == -1:
        ax.axvspan(date, date, color="red", alpha=0.08)
    else:
        ax.axvspan(date, date, color="yellow", alpha=0.05)

ax.set_title("SPX with Regime Overlay")
st.pyplot(fig)

# =========================================================
# WHY TODAY
# =========================================================

latest_idx = composite.index[-1]

why = pd.DataFrame({
    "Factor":["Trend","Volatility","Credit","Liquidity","Crypto"],
    "Z-Score":[
        float(trend_z.loc[latest_idx]),
        float(vol_comp.loc[latest_idx]),
        float(credit_z.loc[latest_idx]),
        float(liq_comp.loc[latest_idx]),
        float(btc_comp.loc[latest_idx])
    ]
})

st.subheader("Why Today?")
st.dataframe(why, use_container_width=True)

st.subheader("Regime Duration")
st.write(f"{duration} trading days")

st.subheader("Daily Summary")
st.markdown(
    f"The system is in **{regime}** with {confidence:.0%} confidence. "
    f"Cross-asset composite score is {latest_score:.2f}. "
    f"Probability of >7% drawdown over 20 days: {drawdown_prob:.0%}."
)
